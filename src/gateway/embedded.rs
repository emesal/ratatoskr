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
use crate::registry::{ModelRegistry, PresetParameters};
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, ModelCapability, ModelGateway, ModelInfo, ModelMetadata,
    ModelStatus, Result, StanceResult, ToolDefinition,
};

#[cfg(feature = "local-inference")]
use crate::model::ModelManager;
#[cfg(feature = "local-inference")]
use crate::tokenizer::TokenizerRegistry;

/// Result of parsing a model string — may include a provider routing hint.
///
/// When a model string contains a `provider:model` prefix matching a known
/// provider name, the prefix is stripped and routing bypasses the fallback
/// chain, dispatching directly to that provider.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedModel {
    /// If set, skip the fallback chain and route to this provider only.
    pub provider: Option<String>,
    /// The actual model identifier (prefix stripped if provider was matched).
    pub model: String,
    /// Default generation parameters from the preset, if any.
    pub preset_parameters: Option<PresetParameters>,
}

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
    /// Prefix for preset model URIs.
    const PRESET_PREFIX: &str = "ratatoskr:";

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

    /// Resolve a model string, handling two prefix schemes:
    ///
    /// 1. `ratatoskr:<tier>/<capability>` — preset URI, resolved to a concrete
    ///    model ID with no provider hint.
    /// 2. `<provider>:<model>` — provider prefix routing. If the prefix matches
    ///    a registered provider name, the request is routed directly to that
    ///    provider, bypassing the fallback chain.
    /// 3. No prefix — plain model string, uses the normal fallback chain.
    fn resolve_model(&self, model: &str) -> Result<ResolvedModel> {
        // 1. ratatoskr: preset URIs
        if let Some(rest) = model.strip_prefix(Self::PRESET_PREFIX) {
            let Some((tier, capability)) = rest.split_once('/') else {
                return Err(crate::RatatoskrError::InvalidInput(format!(
                    "preset URI must be `ratatoskr:<tier>/<capability>`, got `{model}`"
                )));
            };

            if tier.is_empty() || capability.is_empty() {
                return Err(crate::RatatoskrError::InvalidInput(format!(
                    "preset URI must be `ratatoskr:<tier>/<capability>`, got `{model}`"
                )));
            }

            let entry = self
                .model_registry
                .preset(tier, capability)
                .ok_or_else(|| crate::RatatoskrError::PresetNotFound {
                    tier: tier.to_string(),
                    capability: capability.to_string(),
                })?;

            return Ok(ResolvedModel {
                provider: None,
                model: entry.model().to_owned(),
                preset_parameters: entry.parameters().cloned(),
            });
        }

        // 2. provider:model prefix routing
        if let Some((prefix, rest)) = model.split_once(':') {
            let known = self.registry.provider_names();
            if known.all_unique().contains(prefix) {
                return Ok(ResolvedModel {
                    provider: Some(prefix.to_string()),
                    model: rest.to_string(),
                    preset_parameters: None,
                });
            }
        }

        // 3. plain model string
        Ok(ResolvedModel {
            provider: None,
            model: model.to_string(),
            preset_parameters: None,
        })
    }

    /// Build a patched `ChatOptions` when the resolved model or preset
    /// parameters differ from the caller's options.
    /// Returns `None` when the original options can be used as-is.
    fn apply_resolved_chat(
        &self,
        options: &ChatOptions,
        resolved: &ResolvedModel,
    ) -> Option<ChatOptions> {
        let needs_model_swap = resolved.model != options.model;
        let has_params = resolved.preset_parameters.is_some();

        if !needs_model_swap && !has_params {
            return None;
        }

        let mut opts = options.clone();
        opts.model = resolved.model.clone();
        if let Some(params) = &resolved.preset_parameters {
            params.apply_defaults_to_chat(&mut opts);
        }
        Some(opts)
    }

    /// Build a patched `GenerateOptions` when the resolved model or preset
    /// parameters differ from the caller's options.
    /// Returns `None` when the original options can be used as-is.
    fn apply_resolved_generate(
        &self,
        options: &GenerateOptions,
        resolved: &ResolvedModel,
    ) -> Option<GenerateOptions> {
        let needs_model_swap = resolved.model != options.model;
        let has_params = resolved.preset_parameters.is_some();

        if !needs_model_swap && !has_params {
            return None;
        }

        let mut opts = options.clone();
        opts.model = resolved.model.clone();
        if let Some(params) = &resolved.preset_parameters {
            params.apply_defaults_to_generate(&mut opts);
        }
        Some(opts)
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
        let resolved = self.resolve_model(&options.model)?;
        let opts = self.apply_resolved_chat(options, &resolved);
        self.registry
            .chat_stream(
                messages,
                tools,
                opts.as_ref().unwrap_or(options),
                resolved.provider.as_deref(),
            )
            .await
    }

    #[instrument(name = "gateway.chat", skip(self, messages, tools, options), fields(model = %options.model))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let resolved = self.resolve_model(&options.model)?;
        let opts = self.apply_resolved_chat(options, &resolved);
        self.registry
            .chat(
                messages,
                tools,
                opts.as_ref().unwrap_or(options),
                resolved.provider.as_deref(),
            )
            .await
    }

    fn capabilities(&self) -> Capabilities {
        use crate::ModelCapability::*;

        #[allow(unused_mut)]
        let mut caps = vec![
            (self.registry.has_chat(), Chat),
            (self.registry.has_chat(), ChatStreaming),
            (self.registry.has_generate(), Generate),
            (self.registry.has_chat(), ToolUse),
            (self.registry.has_embedding(), Embed),
            (self.registry.has_nli(), Nli),
            (self.registry.has_classify(), Classify),
            (self.registry.has_stance(), Stance),
        ];

        #[cfg(feature = "local-inference")]
        {
            let names = self.registry.provider_names();
            let has_local = names
                .embedding
                .iter()
                .any(|n| n.starts_with("local-") || n.contains("fastembed"))
                || names
                    .nli
                    .iter()
                    .any(|n| n.starts_with("local-") || n.contains("onnx"));
            caps.push((true, TokenCounting));
            caps.push((has_local, LocalInference));
        }

        caps.into_iter()
            .filter(|(enabled, _)| *enabled)
            .map(|(_, cap)| cap)
            .collect()
    }

    #[instrument(name = "gateway.embed", skip(self, text))]
    async fn embed(&self, text: &str, model: &str) -> Result<crate::Embedding> {
        let resolved = self.resolve_model(model)?;
        let hint = resolved.provider.as_deref();
        if let Some(cache) = &self.response_cache {
            if let Some(cached) = cache.get_embedding(&resolved.model, text).await {
                return Ok(cached);
            }
            let result = self.registry.embed(text, &resolved.model, hint).await?;
            cache
                .insert_embedding(&resolved.model, text, result.clone())
                .await;
            return Ok(result);
        }
        self.registry.embed(text, &resolved.model, hint).await
    }

    #[instrument(name = "gateway.embed_batch", skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<crate::Embedding>> {
        let resolved = self.resolve_model(model)?;
        let hint = resolved.provider.as_deref();
        if let Some(cache) = &self.response_cache {
            let cached = cache.get_embedding_batch(&resolved.model, texts).await;

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

            let provider_results = self
                .registry
                .embed_batch(&miss_texts, &resolved.model, hint)
                .await?;

            // Cache the newly fetched results
            cache
                .insert_embedding_batch(&resolved.model, &miss_texts, &provider_results)
                .await;

            return Ok(merge_batch_results(cached, provider_results));
        }
        self.registry
            .embed_batch(texts, &resolved.model, hint)
            .await
    }

    #[instrument(name = "gateway.infer_nli", skip(self, premise, hypothesis))]
    async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<crate::NliResult> {
        let resolved = self.resolve_model(model)?;
        let hint = resolved.provider.as_deref();
        if let Some(cache) = &self.response_cache {
            if let Some(cached) = cache.get_nli(&resolved.model, premise, hypothesis).await {
                return Ok(cached);
            }
            let result = self
                .registry
                .infer_nli(premise, hypothesis, &resolved.model, hint)
                .await?;
            cache
                .insert_nli(&resolved.model, premise, hypothesis, result.clone())
                .await;
            return Ok(result);
        }
        self.registry
            .infer_nli(premise, hypothesis, &resolved.model, hint)
            .await
    }

    #[instrument(name = "gateway.classify_zero_shot", skip(self, text, labels))]
    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<crate::ClassifyResult> {
        let resolved = self.resolve_model(model)?;
        self.registry
            .classify_zero_shot(text, labels, &resolved.model, resolved.provider.as_deref())
            .await
    }

    #[cfg(feature = "local-inference")]
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        self.tokenizer_registry.count_tokens(text, model)
    }

    #[instrument(name = "gateway.generate", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        let resolved = self.resolve_model(&options.model)?;
        let opts = self.apply_resolved_generate(options, &resolved);
        self.registry
            .generate(
                prompt,
                opts.as_ref().unwrap_or(options),
                resolved.provider.as_deref(),
            )
            .await
    }

    #[instrument(name = "gateway.generate_stream", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        let resolved = self.resolve_model(&options.model)?;
        let opts = self.apply_resolved_generate(options, &resolved);
        self.registry
            .generate_stream(
                prompt,
                opts.as_ref().unwrap_or(options),
                resolved.provider.as_deref(),
            )
            .await
    }

    #[instrument(name = "gateway.classify_stance", skip(self, text, target))]
    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        let resolved = self.resolve_model(model)?;
        self.registry
            .classify_stance(text, target, &resolved.model, resolved.provider.as_deref())
            .await
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
        let resolved = self.resolve_model(model).ok()?;
        // Registry (curated) takes priority over cache (ephemeral)
        self.model_registry
            .get(&resolved.model)
            .cloned()
            .or_else(|| self.model_cache.get(&resolved.model))
    }

    #[instrument(name = "gateway.fetch_model_metadata", skip(self))]
    async fn fetch_model_metadata(&self, model: &str) -> Result<ModelMetadata> {
        let resolved = self.resolve_model(model)?;
        let metadata = self
            .registry
            .fetch_chat_metadata(&resolved.model, resolved.provider.as_deref())
            .await?;
        self.model_cache.insert(metadata.clone());
        Ok(metadata)
    }

    fn resolve_preset(&self, tier: &str, capability: &str) -> Option<crate::PresetResolution> {
        self.model_registry.preset(tier, capability).map(|entry| {
            crate::PresetResolution {
                model: entry.model().to_owned(),
                parameters: entry.parameters().cloned(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RatatoskrError;

    /// Build a minimal gateway with the embedded seed registry (has presets).
    fn test_gateway() -> EmbeddedGateway {
        let registry = crate::providers::ProviderRegistry::new();
        let model_registry = ModelRegistry::with_embedded_seed();
        let model_cache = Arc::new(ModelCache::new());
        EmbeddedGateway::new(
            registry,
            model_registry,
            model_cache,
            None,
            #[cfg(feature = "local-inference")]
            Arc::new(crate::model::ModelManager::new(
                std::path::PathBuf::from("/tmp"),
                None,
            )),
            #[cfg(feature = "local-inference")]
            Arc::new(crate::tokenizer::TokenizerRegistry::new()),
        )
    }

    #[test]
    fn test_resolve_preset_uri() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:free/text-generation").unwrap();
        assert!(
            !resolved.model.is_empty(),
            "free/text-generation should resolve"
        );
        assert!(
            !resolved.model.starts_with("ratatoskr:"),
            "should resolve to concrete model"
        );
        assert!(
            resolved.provider.is_none(),
            "presets should not set provider hint"
        );
    }

    #[test]
    fn test_resolve_preset_uri_agentic() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:free/agentic").unwrap();
        assert!(!resolved.model.is_empty(), "free/agentic should resolve");
        assert!(
            !resolved.model.starts_with("ratatoskr:"),
            "should resolve to concrete model"
        );
        assert!(resolved.provider.is_none());
    }

    #[test]
    fn test_resolve_preset_uri_premium() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:premium/agentic").unwrap();
        assert!(!resolved.model.is_empty(), "premium/agentic should resolve");
        assert!(
            !resolved.model.starts_with("ratatoskr:"),
            "should resolve to concrete model"
        );
        assert!(resolved.provider.is_none());
    }

    #[test]
    fn test_resolve_plain_model_passthrough() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("anthropic/claude-sonnet-4").unwrap();
        assert_eq!(resolved.model, "anthropic/claude-sonnet-4");
        assert!(resolved.provider.is_none());
    }

    #[test]
    fn test_resolve_preset_unknown_tier() {
        let gw = test_gateway();
        let err = gw
            .resolve_model("ratatoskr:nonexistent/agentic")
            .unwrap_err();
        assert!(matches!(err, RatatoskrError::PresetNotFound { .. }));
    }

    #[test]
    fn test_resolve_preset_unknown_capability() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:free/nonexistent").unwrap_err();
        assert!(matches!(err, RatatoskrError::PresetNotFound { .. }));
    }

    #[test]
    fn test_resolve_preset_missing_capability() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:free").unwrap_err();
        assert!(matches!(err, RatatoskrError::InvalidInput(_)));
    }

    #[test]
    fn test_resolve_preset_empty_after_prefix() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:").unwrap_err();
        assert!(matches!(err, RatatoskrError::InvalidInput(_)));
    }

    // ========================================================================
    // Provider prefix routing tests
    // ========================================================================

    /// Build a gateway with named providers for prefix routing tests.
    fn gateway_with_providers() -> EmbeddedGateway {
        let mut registry = crate::providers::ProviderRegistry::new();

        // Register mock embedding providers with known names
        struct StubEmbed(&'static str);
        #[async_trait]
        impl crate::providers::traits::EmbeddingProvider for StubEmbed {
            fn name(&self) -> &str {
                self.0
            }
            async fn embed(&self, _text: &str, model: &str) -> Result<crate::Embedding> {
                Ok(crate::Embedding {
                    values: vec![1.0],
                    model: model.to_string(),
                    dimensions: 1,
                })
            }
        }

        registry.add_embedding(std::sync::Arc::new(StubEmbed("openrouter")));
        registry.add_embedding(std::sync::Arc::new(StubEmbed("anthropic")));

        let model_registry = ModelRegistry::with_embedded_seed();
        let model_cache = std::sync::Arc::new(ModelCache::new());
        EmbeddedGateway::new(
            registry,
            model_registry,
            model_cache,
            None,
            #[cfg(feature = "local-inference")]
            std::sync::Arc::new(crate::model::ModelManager::new(
                std::path::PathBuf::from("/tmp"),
                None,
            )),
            #[cfg(feature = "local-inference")]
            std::sync::Arc::new(crate::tokenizer::TokenizerRegistry::new()),
        )
    }

    #[test]
    fn test_resolve_provider_prefix() {
        let gw = gateway_with_providers();
        let resolved = gw
            .resolve_model("anthropic:claude-sonnet-4-20250514")
            .unwrap();
        assert_eq!(
            resolved,
            ResolvedModel {
                provider: Some("anthropic".to_string()),
                model: "claude-sonnet-4-20250514".to_string(),
                preset_parameters: None,
            }
        );
    }

    #[test]
    fn test_resolve_provider_prefix_with_slash_model() {
        let gw = gateway_with_providers();
        let resolved = gw
            .resolve_model("openrouter:anthropic/claude-sonnet-4")
            .unwrap();
        assert_eq!(
            resolved,
            ResolvedModel {
                provider: Some("openrouter".to_string()),
                model: "anthropic/claude-sonnet-4".to_string(),
                preset_parameters: None,
            }
        );
    }

    #[test]
    fn test_resolve_unknown_prefix_passthrough() {
        let gw = gateway_with_providers();
        // "unknown" is not a registered provider, so the whole string passes through
        let resolved = gw.resolve_model("unknown:something").unwrap();
        assert_eq!(
            resolved,
            ResolvedModel {
                provider: None,
                model: "unknown:something".to_string(),
                preset_parameters: None,
            }
        );
    }

    #[test]
    fn test_resolve_no_colon_passthrough() {
        let gw = gateway_with_providers();
        let resolved = gw.resolve_model("claude-sonnet-4-20250514").unwrap();
        assert_eq!(
            resolved,
            ResolvedModel {
                provider: None,
                model: "claude-sonnet-4-20250514".to_string(),
                preset_parameters: None,
            }
        );
    }

    #[test]
    fn test_resolve_preset_no_provider_hint() {
        let gw = gateway_with_providers();
        let resolved = gw.resolve_model("ratatoskr:free/agentic").unwrap();
        assert!(resolved.provider.is_none());
        assert!(!resolved.model.starts_with("ratatoskr:"));
    }

    #[test]
    fn test_resolve_plain_model_no_parameters() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("anthropic/claude-sonnet-4").unwrap();
        assert!(resolved.preset_parameters.is_none());
    }

    #[test]
    fn test_resolve_preset_carries_parameters() {
        use crate::registry::{ModelRegistry, PresetEntry, PresetParameters};

        // Build a gateway with a parameterised preset in its registry.
        let mut model_registry = ModelRegistry::new();
        model_registry.set_preset(
            "test",
            "agentic",
            PresetEntry::WithParams {
                model: "some/model".to_owned(),
                parameters: PresetParameters {
                    temperature: Some(0.3),
                    top_p: Some(0.9),
                    ..Default::default()
                },
            },
        );

        let registry = crate::providers::ProviderRegistry::new();
        let model_cache = Arc::new(ModelCache::new());
        let gw = EmbeddedGateway::new(
            registry,
            model_registry,
            model_cache,
            None,
            #[cfg(feature = "local-inference")]
            Arc::new(crate::model::ModelManager::new(
                std::path::PathBuf::from("/tmp"),
                None,
            )),
            #[cfg(feature = "local-inference")]
            Arc::new(crate::tokenizer::TokenizerRegistry::new()),
        );

        let resolved = gw.resolve_model("ratatoskr:test/agentic").unwrap();
        assert_eq!(resolved.model, "some/model");
        let params = resolved.preset_parameters.unwrap();
        assert_eq!(params.temperature, Some(0.3));
        assert_eq!(params.top_p, Some(0.9));
    }
}
