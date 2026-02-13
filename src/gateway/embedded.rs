//! EmbeddedGateway - wraps the ProviderRegistry for embedded mode
//!
//! This module provides [`EmbeddedGateway`], which implements [`ModelGateway`]
//! by delegating to a [`ProviderRegistry`] fallback chain.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use tracing::instrument;

#[cfg(feature = "local-inference")]
use crate::Token;
use crate::cache::response::merge_batch_results;
use crate::cache::{ModelCache, ResponseCache};
use crate::providers::ProviderRegistry;
use crate::registry::ModelRegistry;
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, CostTier, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, ModelCapability, ModelGateway, ModelInfo, ModelMetadata,
    ModelStatus, Result, StanceResult, ToolDefinition,
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
    model_registry: ModelRegistry,
    model_cache: Arc<ModelCache>,
    /// Opt-in response cache for deterministic operations (embed, NLI).
    ///
    /// When `Some`, embed/NLI calls check the cache before dispatching to
    /// the provider registry. Cache sits above retry, fallback, and metrics —
    /// a hit bypasses the entire provider stack.
    ///
    /// # Future: shared/distributed caching
    ///
    /// To support shared caching across multiple gateway instances (e.g.
    /// redis-backed for a ratd cluster), replace this field with
    /// `Option<Arc<dyn CacheBackend>>` where `CacheBackend` is the trait
    /// described in [`crate::cache::response`] module docs. All cache
    /// interactions flow through this single point in `EmbeddedGateway`,
    /// so no other modules need changes.
    response_cache: Option<ResponseCache>,
    #[cfg(feature = "local-inference")]
    model_manager: Arc<ModelManager>,
    #[cfg(feature = "local-inference")]
    tokenizer_registry: Arc<TokenizerRegistry>,
}

impl EmbeddedGateway {
    /// Create a new EmbeddedGateway with the given registries.
    pub(crate) fn new(
        registry: ProviderRegistry,
        model_registry: ModelRegistry,
        model_cache: Arc<ModelCache>,
        response_cache: Option<ResponseCache>,
        #[cfg(feature = "local-inference")] model_manager: Arc<ModelManager>,
        #[cfg(feature = "local-inference")] tokenizer_registry: Arc<TokenizerRegistry>,
    ) -> Self {
        Self {
            registry,
            model_registry,
            model_cache,
            response_cache,
            #[cfg(feature = "local-inference")]
            model_manager,
            #[cfg(feature = "local-inference")]
            tokenizer_registry,
        }
    }
}

#[async_trait]
impl ModelGateway for EmbeddedGateway {
    #[instrument(name = "gateway.chat_stream", skip(self, messages, tools, options), fields(model = %options.model))]
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        self.registry.chat_stream(messages, tools, options).await
    }

    #[instrument(name = "gateway.chat", skip(self, messages, tools, options), fields(model = %options.model))]
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
            embed: self.registry.has_embedding(),
            nli: self.registry.has_nli(),
            classify: self.registry.has_classify(),
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

    #[instrument(name = "gateway.embed", skip(self, text))]
    async fn embed(&self, text: &str, model: &str) -> Result<crate::Embedding> {
        if let Some(cache) = &self.response_cache {
            if let Some(cached) = cache.get_embedding(model, text).await {
                return Ok(cached);
            }
            let result = self.registry.embed(text, model).await?;
            cache.insert_embedding(model, text, result.clone()).await;
            return Ok(result);
        }
        self.registry.embed(text, model).await
    }

    #[instrument(name = "gateway.embed_batch", skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<crate::Embedding>> {
        if let Some(cache) = &self.response_cache {
            let cached = cache.get_embedding_batch(model, texts).await;

            // Collect miss indices and their texts for provider dispatch
            let miss_texts: Vec<&str> = cached
                .iter()
                .enumerate()
                .filter(|(_, opt)| opt.is_none())
                .map(|(i, _)| texts[i])
                .collect();

            if miss_texts.is_empty() {
                // All hits — no provider call needed
                return Ok(merge_batch_results(cached, vec![]));
            }

            let provider_results = self.registry.embed_batch(&miss_texts, model).await?;

            // Cache the newly fetched results
            cache
                .insert_embedding_batch(model, &miss_texts, &provider_results)
                .await;

            return Ok(merge_batch_results(cached, provider_results));
        }
        self.registry.embed_batch(texts, model).await
    }

    #[instrument(name = "gateway.infer_nli", skip(self, premise, hypothesis))]
    async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<crate::NliResult> {
        if let Some(cache) = &self.response_cache {
            if let Some(cached) = cache.get_nli(model, premise, hypothesis).await {
                return Ok(cached);
            }
            let result = self.registry.infer_nli(premise, hypothesis, model).await?;
            cache
                .insert_nli(model, premise, hypothesis, result.clone())
                .await;
            return Ok(result);
        }
        self.registry.infer_nli(premise, hypothesis, model).await
    }

    #[instrument(name = "gateway.classify_zero_shot", skip(self, text, labels))]
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

    #[instrument(name = "gateway.generate", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        self.registry.generate(prompt, options).await
    }

    #[instrument(name = "gateway.generate_stream", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        self.registry.generate_stream(prompt, options).await
    }

    #[instrument(name = "gateway.classify_stance", skip(self, text, target))]
    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        self.registry.classify_stance(text, target, model).await
    }

    #[cfg(feature = "local-inference")]
    fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>> {
        self.tokenizer_registry.tokenize_detailed(text, model)
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let names = self.registry.provider_names();

        let mut models = Vec::new();

        for name in names.embedding {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Embed],
                context_window: None,
                dimensions: None,
            });
        }

        for name in names.nli {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Nli],
                context_window: None,
                dimensions: None,
            });
        }

        for name in names.classify {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Classify],
                context_window: None,
                dimensions: None,
            });
        }

        for name in names.chat {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Chat],
                context_window: None,
                dimensions: None,
            });
        }

        for name in names.generate {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Generate],
                context_window: None,
                dimensions: None,
            });
        }

        models
    }

    #[allow(unused_variables)]
    fn model_status(&self, model: &str) -> ModelStatus {
        #[cfg(feature = "local-inference")]
        {
            let loaded = self.model_manager.loaded_models();
            let model_str = model.to_string();
            if loaded.embeddings.contains(&model_str) || loaded.nli.contains(&model_str) {
                return ModelStatus::Ready;
            }
            if self.model_manager.can_load(model) {
                return ModelStatus::Available;
            }
            return ModelStatus::Unavailable {
                reason: "RAM budget exceeded".into(),
            };
        }

        #[cfg(not(feature = "local-inference"))]
        {
            // For API-only mode, always "ready" if provider exists
            ModelStatus::Ready
        }
    }

    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        // Registry (curated) takes priority over cache (ephemeral)
        self.model_registry
            .get(model)
            .cloned()
            .or_else(|| self.model_cache.get(model))
    }

    #[instrument(name = "gateway.fetch_model_metadata", skip(self))]
    async fn fetch_model_metadata(&self, model: &str) -> Result<ModelMetadata> {
        let metadata = self.registry.fetch_chat_metadata(model).await?;
        self.model_cache.insert(metadata.clone());
        Ok(metadata)
    }

    fn resolve_preset(&self, tier: CostTier, capability: &str) -> Option<String> {
        self.model_registry
            .preset(tier, capability)
            .map(String::from)
    }
}
