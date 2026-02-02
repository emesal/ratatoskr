//! EmbeddedGateway - wraps the llm crate for embedded mode

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use llm::LLMProvider;
use llm::builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder};

use crate::convert::{backend_from_model, from_llm_tool_calls, from_llm_usage, to_llm_messages};
use crate::types::response::FinishReason;
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, Message, ModelGateway, RatatoskrError,
    Result, ToolDefinition,
};

/// Gateway that wraps the llm crate for embedded mode.
pub struct EmbeddedGateway {
    openrouter_key: Option<String>,
    anthropic_key: Option<String>,
    openai_key: Option<String>,
    google_key: Option<String>,
    ollama_url: Option<String>,
    timeout_secs: u64,
    #[cfg(feature = "huggingface")]
    huggingface: Option<crate::providers::HuggingFaceClient>,
    #[cfg(feature = "huggingface")]
    router: super::routing::CapabilityRouter,
}

impl EmbeddedGateway {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        openrouter_key: Option<String>,
        anthropic_key: Option<String>,
        openai_key: Option<String>,
        google_key: Option<String>,
        ollama_url: Option<String>,
        timeout_secs: u64,
        #[cfg(feature = "huggingface")] huggingface: Option<crate::providers::HuggingFaceClient>,
        #[cfg(feature = "huggingface")] router: super::routing::CapabilityRouter,
    ) -> Self {
        Self {
            openrouter_key,
            anthropic_key,
            openai_key,
            google_key,
            ollama_url,
            timeout_secs,
            #[cfg(feature = "huggingface")]
            huggingface,
            #[cfg(feature = "huggingface")]
            router,
        }
    }

    /// Check if at least one chat provider is configured.
    fn has_chat_provider(&self) -> bool {
        self.openrouter_key.is_some()
            || self.anthropic_key.is_some()
            || self.openai_key.is_some()
            || self.google_key.is_some()
            || self.ollama_url.is_some()
    }

    /// Build an llm provider for the given model and tools
    fn build_provider(
        &self,
        options: &ChatOptions,
        system_prompt: Option<&str>,
        tools: Option<&[ToolDefinition]>,
    ) -> Result<Box<dyn LLMProvider>> {
        let backend = backend_from_model(&options.model);

        let api_key = match backend {
            LLMBackend::OpenRouter => self
                .openrouter_key
                .clone()
                .ok_or(RatatoskrError::NoProvider)?,
            LLMBackend::Anthropic => self
                .anthropic_key
                .clone()
                .ok_or(RatatoskrError::NoProvider)?,
            LLMBackend::OpenAI => self.openai_key.clone().ok_or(RatatoskrError::NoProvider)?,
            LLMBackend::Google => self.google_key.clone().ok_or(RatatoskrError::NoProvider)?,
            LLMBackend::Ollama => "ollama".to_string(), // Ollama doesn't need a key
            _ => return Err(RatatoskrError::Unsupported),
        };

        // Clone backend for later comparison since it's consumed by builder
        let backend_for_check = backend.clone();

        let mut builder = LLMBuilder::new()
            .backend(backend)
            .api_key(api_key)
            .model(&options.model)
            .timeout_seconds(self.timeout_secs);

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
                let llm_effort = match effort {
                    crate::ReasoningEffort::Low => llm::chat::ReasoningEffort::Low,
                    crate::ReasoningEffort::Medium => llm::chat::ReasoningEffort::Medium,
                    crate::ReasoningEffort::High => llm::chat::ReasoningEffort::High,
                };
                builder = builder.reasoning_effort(llm_effort);
            }
            if let Some(max_tokens) = reasoning.max_tokens {
                builder = builder.reasoning_budget_tokens(max_tokens as u32);
            }
            builder = builder.reasoning(true);
        }

        // Handle Ollama URL
        if backend_for_check == LLMBackend::Ollama
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

                builder = builder.function(func_builder);
            }
        }

        builder
            .build()
            .map_err(|e| RatatoskrError::Llm(e.to_string()))
    }
}

#[async_trait]
impl ModelGateway for EmbeddedGateway {
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let (system_prompt, llm_messages) = to_llm_messages(messages);
        let provider = self.build_provider(options, system_prompt.as_deref(), tools)?;

        // Use chat_stream_with_tools to properly handle tool calls in streaming
        let stream = provider
            .chat_stream_with_tools(&llm_messages, provider.tools())
            .await
            .map_err(RatatoskrError::from)?;

        // Convert llm StreamChunk to our ChatEvent type
        let converted_stream = stream.map(
            |result: std::result::Result<llm::chat::StreamChunk, llm::error::LLMError>| {
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
                        llm::chat::StreamChunk::ToolUseComplete { .. } => {
                            // We don't have a direct equivalent; the caller accumulates from deltas
                            // Emit Done to signal completion (caller should have accumulated tool calls)
                            ChatEvent::Done
                        }
                        llm::chat::StreamChunk::Done { .. } => ChatEvent::Done,
                    })
                    .map_err(RatatoskrError::from)
            },
        );

        Ok(Box::pin(converted_stream))
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let (system_prompt, llm_messages) = to_llm_messages(messages);
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

    fn capabilities(&self) -> Capabilities {
        let mut caps = if self.has_chat_provider() {
            Capabilities::chat_only()
        } else {
            Capabilities::default()
        };

        #[cfg(feature = "huggingface")]
        if self.huggingface.is_some() {
            caps.embeddings = self.router.embed_provider().is_some();
            caps.nli = self.router.nli_provider().is_some();
            caps.classification = self.router.classify_provider().is_some();
        }

        caps
    }

    #[cfg(feature = "huggingface")]
    async fn embed(&self, text: &str, model: &str) -> Result<crate::Embedding> {
        let hf = self
            .huggingface
            .as_ref()
            .ok_or(RatatoskrError::Unsupported)?;
        hf.embed(text, model).await
    }

    #[cfg(feature = "huggingface")]
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<crate::Embedding>> {
        let hf = self
            .huggingface
            .as_ref()
            .ok_or(RatatoskrError::Unsupported)?;
        hf.embed_batch(texts, model).await
    }

    #[cfg(feature = "huggingface")]
    async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<crate::NliResult> {
        let hf = self
            .huggingface
            .as_ref()
            .ok_or(RatatoskrError::Unsupported)?;
        hf.infer_nli(premise, hypothesis, model).await
    }

    #[cfg(feature = "huggingface")]
    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<crate::ClassifyResult> {
        let hf = self
            .huggingface
            .as_ref()
            .ok_or(RatatoskrError::Unsupported)?;
        hf.classify(text, labels, model).await
    }
}
