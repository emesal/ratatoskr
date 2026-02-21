//! LLM crate wrapper implementing ChatProvider and GenerateProvider traits.
//!
//! This module provides [`LlmChatProvider`], which stores LLM configuration and
//! builds providers per-request because the llm crate requires tools to be
//! specified at build time. Also implements [`ChatProvider::fetch_metadata()`]
//! for OpenRouter (fetches `/api/v1/models` and converts to [`ModelMetadata`](crate::ModelMetadata)).

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use llm::LLMProvider;
use llm::builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder};
use llm::completion::CompletionRequest;
use tracing::instrument;

use crate::convert::{from_llm_tool_calls, from_llm_usage, to_llm_messages};
use crate::providers::workarounds;
use crate::types::{
    ChatEvent, ChatOptions, ChatResponse, FinishReason, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, ParameterName, ToolChoice, ToolDefinition,
};
use crate::{RatatoskrError, Result};

use super::traits::{ChatProvider, GenerateProvider};

/// Wraps llm crate provider configuration to implement our traits.
///
/// Unlike a direct wrapper, this stores configuration and builds providers
/// per-request because the llm crate requires tools to be specified at
/// build time.
///
/// # Example
///
/// ```ignore
/// use llm::builder::LLMBackend;
/// use ratatoskr::providers::LlmChatProvider;
///
/// let provider = LlmChatProvider::new(LLMBackend::OpenRouter, Some("your-key"), "openrouter");
/// ```
pub struct LlmChatProvider {
    backend: LLMBackend,
    api_key: Option<String>,
    name: String,
    /// Ollama base URL (only used for Ollama backend)
    ollama_url: Option<String>,
    /// Default timeout in seconds
    timeout_secs: u64,
    /// Shared HTTP client for metadata fetches.
    http_client: reqwest::Client,
    /// Override base URL for model metadata endpoint (testing).
    models_base_url: Option<String>,
}

impl LlmChatProvider {
    /// Create a new LlmChatProvider with the given backend and API key.
    ///
    /// # Arguments
    ///
    /// * `backend` - The LLM backend to use (OpenRouter, Anthropic, etc.)
    /// * `api_key` - API key for the backend (`None` for keyless access)
    /// * `name` - Human-readable name for logging/debugging (e.g., "openrouter", "anthropic")
    pub fn new(
        backend: LLMBackend,
        api_key: Option<impl Into<String>>,
        name: impl Into<String>,
    ) -> Self {
        Self::with_http_client(backend, api_key, name, reqwest::Client::new())
    }

    /// Create a new LlmChatProvider with a shared HTTP client.
    ///
    /// Prefer this over [`new`](Self::new) when multiple providers should
    /// share a connection pool (e.g. from the builder).
    pub fn with_http_client(
        backend: LLMBackend,
        api_key: Option<impl Into<String>>,
        name: impl Into<String>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            backend,
            api_key: api_key.map(|k| k.into()),
            name: name.into(),
            ollama_url: None,
            timeout_secs: 120,
            http_client,
            models_base_url: None,
        }
    }

    /// Set the Ollama base URL (only relevant for Ollama backend).
    pub fn ollama_url(mut self, url: impl Into<String>) -> Self {
        self.ollama_url = Some(url.into());
        self
    }

    /// Set the timeout in seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Override the base URL for the models metadata endpoint.
    ///
    /// Used for testing with wiremock. The full URL is `{base}/api/v1/models`.
    pub fn models_base_url(mut self, url: impl Into<String>) -> Self {
        self.models_base_url = Some(url.into());
        self
    }

    /// Build an llm provider configured for the given options and tools.
    fn build_provider(
        &self,
        options: &ChatOptions,
        system_prompt: Option<&str>,
        tools: Option<&[ToolDefinition]>,
    ) -> Result<Box<dyn LLMProvider>> {
        // compute provider-specific adjustments (parallel_tool_calls, raw_provider_options)
        let adjustments = workarounds::compute_adjustments(&self.backend, options)?;

        let mut builder = LLMBuilder::new()
            .backend(self.backend.clone())
            .model(&options.model)
            .timeout_seconds(self.timeout_secs);
        if let Some(ref key) = self.api_key {
            builder = builder.api_key(key);
        }

        // Add system prompt if present
        if let Some(sys) = system_prompt {
            builder = builder.system(sys);
        }

        if let Some(temp) = options.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(max) = options.max_tokens {
            builder = builder.max_tokens(max as u32);
        }
        if let Some(p) = options.top_p {
            builder = builder.top_p(p);
        }

        // Handle reasoning config
        if let Some(ref reasoning) = options.reasoning {
            if let Some(ref effort) = reasoning.effort {
                // Map extended effort levels to llm crate's Low/Medium/High
                match effort {
                    crate::ReasoningEffort::None => {
                        // Don't enable reasoning at all
                    }
                    _ => {
                        let llm_effort = match effort {
                            crate::ReasoningEffort::Minimal | crate::ReasoningEffort::Low => {
                                llm::chat::ReasoningEffort::Low
                            }
                            crate::ReasoningEffort::Medium => llm::chat::ReasoningEffort::Medium,
                            crate::ReasoningEffort::High | crate::ReasoningEffort::XHigh => {
                                llm::chat::ReasoningEffort::High
                            }
                            crate::ReasoningEffort::None => unreachable!(),
                        };
                        builder = builder.reasoning_effort(llm_effort);
                        builder = builder.reasoning(true);
                    }
                }
            }
            if let Some(max_tokens) = reasoning.max_tokens {
                builder = builder.reasoning_budget_tokens(max_tokens as u32);
                builder = builder.reasoning(true);
            }
        }

        // Handle tool_choice passthrough
        if let Some(ref tc) = options.tool_choice {
            let llm_tc = match tc {
                ToolChoice::Auto => llm::chat::ToolChoice::Auto,
                ToolChoice::None => llm::chat::ToolChoice::None,
                ToolChoice::Required => llm::chat::ToolChoice::Any,
                ToolChoice::Function { name } => llm::chat::ToolChoice::Tool(name.clone()),
            };
            builder = builder.tool_choice(llm_tc);
        }

        // Apply workaround adjustments
        if let Some(extra) = adjustments.extra_body {
            builder = builder.extra_body(extra);
        }
        if let Some(ptc) = adjustments.native_parallel_tool_calls {
            builder = builder.enable_parallel_tool_use(ptc);
        }

        // Handle Ollama URL
        if self.backend == LLMBackend::Ollama
            && let Some(ref url) = self.ollama_url
        {
            builder = builder.base_url(url.clone());
        }

        // Add tools via FunctionBuilder
        if let Some(tools) = tools {
            for tool in tools {
                let mut func_builder =
                    FunctionBuilder::new(&tool.name).description(&tool.description);

                // Extract parameters from JSON schema
                if let Some(props) = tool.parameters.get("properties")
                    && let Some(props_obj) = props.as_object()
                {
                    for (name, schema) in props_obj {
                        let type_str = schema
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("string");
                        let desc = schema
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");

                        func_builder = func_builder
                            .param(ParamBuilder::new(name).type_of(type_str).description(desc));
                    }
                }

                // Mark required fields
                if let Some(required) = tool.parameters.get("required")
                    && let Some(req_arr) = required.as_array()
                {
                    let req_vec: Vec<String> = req_arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    func_builder = func_builder.required(req_vec);
                }

                // Wire cache_control for prompt caching (Anthropic, Bedrock)
                if let Some(cc) = &tool.cache_control {
                    func_builder = func_builder.cache_control(cc.clone());
                }

                builder = builder.function(func_builder);
            }
        }

        builder
            .build()
            .map_err(|e| RatatoskrError::Llm(e.to_string()))
    }
}

#[async_trait]
impl ChatProvider for LlmChatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(name = "llm.chat", skip(self, messages, tools, options), fields(model = %options.model, provider = %self.name))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let (system_prompt, llm_messages) = to_llm_messages(messages)?;
        let provider = self.build_provider(options, system_prompt.as_deref(), tools)?;

        let response = if tools.is_some() {
            provider
                .chat_with_tools(&llm_messages, provider.tools())
                .await
        } else {
            provider.chat(&llm_messages).await
        }
        .map_err(RatatoskrError::from)?;

        let tool_calls = response
            .tool_calls()
            .map(|tc| from_llm_tool_calls(&tc))
            .unwrap_or_default();

        let usage = response.usage().map(|u| from_llm_usage(&u));

        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };

        Ok(ChatResponse {
            content: response.text().unwrap_or_default(),
            reasoning: response.thinking(),
            tool_calls,
            usage,
            model: Some(options.model.clone()),
            finish_reason,
        })
    }

    #[instrument(name = "llm.chat_stream", skip(self, messages, tools, options), fields(model = %options.model, provider = %self.name))]
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let (system_prompt, llm_messages) = to_llm_messages(messages)?;
        let provider = self.build_provider(options, system_prompt.as_deref(), tools)?;

        let stream = provider
            .chat_stream_with_tools(&llm_messages, provider.tools())
            .await
            .map_err(RatatoskrError::from)?;

        // Convert llm StreamChunk to our ChatEvent type
        let converted = stream.map(|result| {
            result
                .map(|chunk| match chunk {
                    llm::chat::StreamChunk::Text(text) => ChatEvent::Content(text),
                    llm::chat::StreamChunk::ToolUseStart { index, id, name } => {
                        ChatEvent::ToolCallStart { index, id, name }
                    }
                    llm::chat::StreamChunk::ToolUseInputDelta {
                        index,
                        partial_json,
                    } => ChatEvent::ToolCallDelta {
                        index,
                        arguments: partial_json,
                    },
                    llm::chat::StreamChunk::ToolUseComplete { index, .. } => {
                        ChatEvent::ToolCallEnd { index }
                    }
                    llm::chat::StreamChunk::Thinking(text) => ChatEvent::Reasoning(text),
                    llm::chat::StreamChunk::Done { .. } => ChatEvent::Done,
                })
                .map_err(RatatoskrError::from)
        });

        Ok(Box::pin(converted))
    }

    fn supported_chat_parameters(&self) -> Vec<ParameterName> {
        vec![
            ParameterName::Temperature,
            ParameterName::MaxTokens,
            ParameterName::TopP,
            ParameterName::Reasoning,
            ParameterName::Stop,
            ParameterName::ToolChoice,
            ParameterName::ParallelToolCalls,
        ]
    }

    #[instrument(name = "llm.fetch_metadata", skip(self), fields(provider = %self.name))]
    async fn fetch_metadata(&self, model: &str) -> Result<crate::types::ModelMetadata> {
        use super::openrouter_models::{ModelsResponse, into_model_metadata};

        // Only OpenRouter supports metadata fetch for now
        if self.backend != LLMBackend::OpenRouter {
            return Err(RatatoskrError::ModelNotAvailable);
        }

        let base = self
            .models_base_url
            .as_deref()
            .unwrap_or("https://openrouter.ai");
        let url = format!("{base}/api/v1/models");
        let mut request = self.http_client.get(url);
        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(RatatoskrError::Api {
                status: response.status().as_u16(),
                message: response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".into()),
            });
        }

        let models: ModelsResponse = response
            .json()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        let entry = models
            .data
            .into_iter()
            .find(|m| m.id == model)
            .ok_or(RatatoskrError::ModelNotAvailable)?;

        Ok(into_model_metadata(entry))
    }
}

#[async_trait]
impl GenerateProvider for LlmChatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(name = "llm.generate", skip(self, prompt, options), fields(model = %options.model, provider = %self.name))]
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        // Build chat options for the provider
        let mut chat_options = ChatOptions::new(&options.model);
        if let Some(temp) = options.temperature {
            chat_options = chat_options.temperature(temp);
        }
        if let Some(max) = options.max_tokens {
            chat_options = chat_options.max_tokens(max);
        }
        if let Some(p) = options.top_p {
            chat_options = chat_options.top_p(p);
        }
        // top_k, frequency_penalty, presence_penalty, seed, reasoning
        // are not yet passed to llm crate — tracked for future provider updates

        // Build provider without tools
        let provider = self.build_provider(&chat_options, None, None)?;

        // Build completion request
        let mut req_builder = CompletionRequest::builder(prompt);
        if let Some(max_tokens) = options.max_tokens {
            req_builder = req_builder.max_tokens(max_tokens as u32);
        }
        if let Some(temp) = options.temperature {
            req_builder = req_builder.temperature(temp);
        }

        let response = provider
            .complete(&req_builder.build())
            .await
            .map_err(RatatoskrError::from)?;

        Ok(GenerateResponse {
            text: response.text,
            usage: None, // llm crate's CompletionResponse doesn't include usage
            model: Some(options.model.clone()),
            finish_reason: FinishReason::Stop,
        })
    }

    #[instrument(name = "llm.generate_stream", skip(self, prompt, options), fields(model = %options.model, provider = %self.name))]
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        // llm crate doesn't have streaming completion, so we use chat_stream
        // by wrapping the prompt in a user message
        let messages = vec![Message::user(prompt)];

        let mut chat_options = ChatOptions::new(&options.model);
        if let Some(temp) = options.temperature {
            chat_options = chat_options.temperature(temp);
        }
        if let Some(max) = options.max_tokens {
            chat_options = chat_options.max_tokens(max);
        }
        if let Some(p) = options.top_p {
            chat_options = chat_options.top_p(p);
        }

        // Get streaming chat response
        let chat_stream = self.chat_stream(&messages, None, &chat_options).await?;

        // Convert ChatEvent to GenerateEvent, filtering out non-content events.
        let generate_stream = chat_stream.filter_map(|result| async {
            match result {
                Ok(ChatEvent::Content(text)) => Some(Ok(GenerateEvent::Text(text))),
                Ok(ChatEvent::Done) => Some(Ok(GenerateEvent::Done)),
                Err(e) => Some(Err(e)),
                // Discard reasoning, tool calls, usage — not relevant for generate.
                _ => None,
            }
        });

        Ok(Box::pin(generate_stream))
    }

    fn supported_generate_parameters(&self) -> Vec<ParameterName> {
        vec![
            ParameterName::Temperature,
            ParameterName::MaxTokens,
            ParameterName::TopP,
            ParameterName::Stop,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider =
            LlmChatProvider::new(LLMBackend::OpenRouter, Some("test-key"), "test-provider");
        assert_eq!(ChatProvider::name(&provider), "test-provider");
    }

    #[test]
    fn test_ollama_url_builder() {
        let provider = LlmChatProvider::new(LLMBackend::Ollama, Some("ollama"), "ollama")
            .ollama_url("http://localhost:11434");
        assert_eq!(
            provider.ollama_url,
            Some("http://localhost:11434".to_string())
        );
    }

    #[test]
    fn test_timeout_builder() {
        let provider = LlmChatProvider::new(LLMBackend::OpenRouter, Some("key"), "openrouter")
            .timeout_secs(60);
        assert_eq!(provider.timeout_secs, 60);
    }
}
