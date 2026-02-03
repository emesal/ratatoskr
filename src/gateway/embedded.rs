//! EmbeddedGateway - wraps the ProviderRegistry for embedded mode
//!
//! This module provides [`EmbeddedGateway`], which implements [`ModelGateway`]
//! by delegating to a [`ProviderRegistry`] fallback chain.

use std::pin::Pin;
#[cfg(feature = "local-inference")]
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use crate::providers::ProviderRegistry;
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, ModelGateway, Result, ToolDefinition,
};

#[cfg(feature = "local-inference")]
use crate::model::ModelManager;
#[cfg(feature = "local-inference")]
use crate::tokenizer::TokenizerRegistry;

/// Gateway that wraps a ProviderRegistry for embedded mode.
///
/// All method calls are delegated to the underlying registry, which handles
/// the fallback chain (local providers → API providers) automatically.
pub struct EmbeddedGateway {
    registry: ProviderRegistry,
    #[cfg(feature = "local-inference")]
    #[allow(dead_code)] // used for model status queries
    model_manager: Arc<ModelManager>,
    #[cfg(feature = "local-inference")]
    tokenizer_registry: Arc<TokenizerRegistry>,
}

impl EmbeddedGateway {
    /// Create a new EmbeddedGateway with the given registry.
    pub(crate) fn new(
        registry: ProviderRegistry,
        #[cfg(feature = "local-inference")] model_manager: Arc<ModelManager>,
        #[cfg(feature = "local-inference")] tokenizer_registry: Arc<TokenizerRegistry>,
    ) -> Self {
        Self {
            registry,
            #[cfg(feature = "local-inference")]
            model_manager,
            #[cfg(feature = "local-inference")]
            tokenizer_registry,
        }
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
        self.registry.chat_stream(messages, tools, options).await
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.registry.chat(messages, tools, options).await
    }

    fn capabilities(&self) -> Capabilities {
        #[cfg(feature = "local-inference")]
        let (token_counting, local_inference) = {
            // Check if any local providers are registered
            let names = self.registry.provider_names();
            let has_local = names
                .embedding
                .iter()
                .any(|n| n.starts_with("local-") || n.contains("fastembed"))
                || names
                    .nli
                    .iter()
                    .any(|n| n.starts_with("local-") || n.contains("onnx"));
            (true, has_local)
        };

        Capabilities {
            chat: self.registry.has_chat(),
            chat_streaming: self.registry.has_chat(),
            generate: self.registry.has_generate(),
            tool_use: self.registry.has_chat(),
            embeddings: self.registry.has_embedding(),
            nli: self.registry.has_nli(),
            classification: self.registry.has_classify(),
            stance: self.registry.has_stance(),
            #[cfg(feature = "local-inference")]
            token_counting,
            #[cfg(not(feature = "local-inference"))]
            token_counting: false,
            #[cfg(feature = "local-inference")]
            local_inference,
            #[cfg(not(feature = "local-inference"))]
            local_inference: false,
        }
    }

    async fn embed(&self, text: &str, model: &str) -> Result<crate::Embedding> {
        self.registry.embed(text, model).await
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<crate::Embedding>> {
        self.registry.embed_batch(texts, model).await
    }

    async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<crate::NliResult> {
        self.registry.infer_nli(premise, hypothesis, model).await
    }

    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<crate::ClassifyResult> {
        self.registry.classify_zero_shot(text, labels, model).await
    }

    #[cfg(feature = "local-inference")]
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        self.tokenizer_registry.count_tokens(text, model)
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        self.registry.generate(prompt, options).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        self.registry.generate_stream(prompt, options).await
    }
}
