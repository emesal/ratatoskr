//! Builder for configuring gateway instances

use super::EmbeddedGateway;
use crate::ParameterValidationPolicy;
use crate::{RatatoskrError, Result};

#[cfg(feature = "local-inference")]
use std::path::PathBuf;

#[cfg(feature = "local-inference")]
use crate::model::Device;
#[cfg(feature = "local-inference")]
use crate::providers::{LocalEmbeddingModel, LocalNliModel};
#[cfg(feature = "local-inference")]
use crate::tokenizer::TokenizerSource;

/// Main entry point for creating gateway instances.
pub struct Ratatoskr;

impl Ratatoskr {
    /// Create a new builder for configuring the gateway.
    pub fn builder() -> RatatoskrBuilder {
        RatatoskrBuilder::new()
    }
}

/// Builder for configuring gateway instances.
pub struct RatatoskrBuilder {
    openrouter_key: Option<String>,
    anthropic_key: Option<String>,
    openai_key: Option<String>,
    google_key: Option<String>,
    ollama_url: Option<String>,
    default_timeout_secs: Option<u64>,
    validation_policy: ParameterValidationPolicy,
    #[cfg(feature = "huggingface")]
    huggingface_key: Option<String>,
    #[cfg(feature = "local-inference")]
    local_embedding_model: Option<LocalEmbeddingModel>,
    #[cfg(feature = "local-inference")]
    local_nli_model: Option<LocalNliModel>,
    #[cfg(feature = "local-inference")]
    device: Device,
    #[cfg(feature = "local-inference")]
    cache_dir: Option<PathBuf>,
    #[cfg(feature = "local-inference")]
    tokenizer_mappings: Vec<(String, TokenizerSource)>,
    #[cfg(feature = "local-inference")]
    ram_budget: Option<usize>,
}

impl RatatoskrBuilder {
    pub fn new() -> Self {
        Self {
            openrouter_key: None,
            anthropic_key: None,
            openai_key: None,
            google_key: None,
            ollama_url: None,
            default_timeout_secs: None,
            validation_policy: ParameterValidationPolicy::default(),
            #[cfg(feature = "huggingface")]
            huggingface_key: None,
            #[cfg(feature = "local-inference")]
            local_embedding_model: None,
            #[cfg(feature = "local-inference")]
            local_nli_model: None,
            #[cfg(feature = "local-inference")]
            device: Device::default(),
            #[cfg(feature = "local-inference")]
            cache_dir: None,
            #[cfg(feature = "local-inference")]
            tokenizer_mappings: Vec::new(),
            #[cfg(feature = "local-inference")]
            ram_budget: None,
        }
    }

    /// Configure OpenRouter provider (routes to many models).
    pub fn openrouter(mut self, api_key: impl Into<String>) -> Self {
        self.openrouter_key = Some(api_key.into());
        self
    }

    /// Configure direct Anthropic provider.
    pub fn anthropic(mut self, api_key: impl Into<String>) -> Self {
        self.anthropic_key = Some(api_key.into());
        self
    }

    /// Configure direct OpenAI provider.
    pub fn openai(mut self, api_key: impl Into<String>) -> Self {
        self.openai_key = Some(api_key.into());
        self
    }

    /// Configure Google (Gemini) provider.
    pub fn google(mut self, api_key: impl Into<String>) -> Self {
        self.google_key = Some(api_key.into());
        self
    }

    /// Configure Ollama provider with custom URL.
    pub fn ollama(mut self, url: impl Into<String>) -> Self {
        self.ollama_url = Some(url.into());
        self
    }

    /// Configure HuggingFace provider for embeddings, NLI, and classification.
    #[cfg(feature = "huggingface")]
    pub fn huggingface(mut self, api_key: impl Into<String>) -> Self {
        self.huggingface_key = Some(api_key.into());
        self
    }

    /// Enable local embeddings via FastEmbed.
    #[cfg(feature = "local-inference")]
    pub fn local_embeddings(mut self, model: LocalEmbeddingModel) -> Self {
        self.local_embedding_model = Some(model);
        self
    }

    /// Enable local NLI via ONNX Runtime.
    #[cfg(feature = "local-inference")]
    pub fn local_nli(mut self, model: LocalNliModel) -> Self {
        self.local_nli_model = Some(model);
        self
    }

    /// Set the device for local inference (default: CPU).
    #[cfg(feature = "local-inference")]
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the cache directory for model downloads.
    #[cfg(feature = "local-inference")]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set the RAM budget for local model loading.
    ///
    /// When set, the model manager will refuse to load models that would
    /// exceed this budget, returning `ModelNotAvailable` so the registry
    /// can fall back to API providers.
    #[cfg(feature = "local-inference")]
    pub fn ram_budget(mut self, bytes: usize) -> Self {
        self.ram_budget = Some(bytes);
        self
    }

    /// Add a custom tokenizer mapping.
    #[cfg(feature = "local-inference")]
    pub fn tokenizer_mapping(
        mut self,
        model_pattern: impl Into<String>,
        source: TokenizerSource,
    ) -> Self {
        self.tokenizer_mappings.push((model_pattern.into(), source));
        self
    }

    /// Set default timeout for all requests (seconds).
    pub fn timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = Some(secs);
        self
    }

    /// Set the parameter validation policy.
    ///
    /// Controls how the registry handles requests containing parameters
    /// not declared as supported by the provider:
    ///
    /// - [`ParameterValidationPolicy::Warn`] — log a warning, proceed with request (default)
    /// - [`ParameterValidationPolicy::Error`] — return `UnsupportedParameter` error
    /// - [`ParameterValidationPolicy::Ignore`] — silently proceed
    pub fn validation_policy(mut self, policy: ParameterValidationPolicy) -> Self {
        self.validation_policy = policy;
        self
    }

    /// Check if at least one chat provider is configured.
    fn has_chat_provider(&self) -> bool {
        self.openrouter_key.is_some()
            || self.anthropic_key.is_some()
            || self.openai_key.is_some()
            || self.google_key.is_some()
            || self.ollama_url.is_some()
    }

    /// Check if at least one capability provider is configured.
    fn has_capability_provider(&self) -> bool {
        #[cfg(feature = "huggingface")]
        if self.huggingface_key.is_some() {
            return true;
        }
        #[cfg(feature = "local-inference")]
        if self.local_embedding_model.is_some() || self.local_nli_model.is_some() {
            return true;
        }
        false
    }

    /// Build the gateway.
    pub fn build(self) -> Result<EmbeddedGateway> {
        use crate::cache::ModelCache;
        use crate::providers::{LlmChatProvider, ProviderRegistry};
        use llm::builder::LLMBackend;
        use std::sync::Arc;

        // Must have at least one provider (chat or capability)
        if !self.has_chat_provider() && !self.has_capability_provider() {
            return Err(RatatoskrError::NoProvider);
        }

        let timeout_secs = self.default_timeout_secs.unwrap_or(120);

        // Build model manager for local inference
        #[cfg(feature = "local-inference")]
        let model_manager = {
            use crate::model::ModelManagerConfig;
            let config = ModelManagerConfig {
                cache_dir: self.cache_dir.unwrap_or_else(|| {
                    std::env::var("RATATOSKR_CACHE_DIR")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| {
                            dirs::cache_dir()
                                .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
                                .join("ratatoskr")
                                .join("models")
                        })
                }),
                default_device: self.device.clone(),
                ram_budget: self.ram_budget,
            };
            Arc::new(crate::model::ModelManager::new(config))
        };

        // Build provider registry with fallback chain
        let mut registry = ProviderRegistry::new();
        registry.set_validation_policy(self.validation_policy);

        // =====================================================================
        // Register LOCAL providers FIRST (higher priority)
        // =====================================================================

        #[cfg(feature = "local-inference")]
        if let Some(model) = &self.local_embedding_model {
            let provider = Arc::new(crate::providers::LocalEmbeddingProvider::new(
                model.clone(),
                model_manager.clone(),
            ));
            registry.add_embedding(provider);
        }

        #[cfg(feature = "local-inference")]
        if let Some(model) = &self.local_nli_model {
            let provider = Arc::new(crate::providers::LocalNliProvider::new(
                model.clone(),
                model_manager.clone(),
            ));
            registry.add_nli(provider);
        }

        // =====================================================================
        // Register API providers as FALLBACKS (lower priority)
        // =====================================================================

        #[cfg(feature = "huggingface")]
        if let Some(ref key) = self.huggingface_key {
            let client = Arc::new(crate::providers::HuggingFaceClient::new(key));

            // HuggingFace accepts any model, registered as fallback
            registry.add_embedding(client.clone());
            registry.add_nli(client.clone());
            registry.add_classify(client.clone());

            // Also register ZeroShotStanceProvider as stance fallback
            let stance_fallback = Arc::new(crate::providers::ZeroShotStanceProvider::new(
                client.clone(),
                "facebook/bart-large-mnli",
            ));
            registry.add_stance(stance_fallback);
        }

        // =====================================================================
        // Register CHAT providers (shared HTTP client for metadata fetches)
        // =====================================================================

        let http_client = reqwest::Client::new();

        // Helper: build an LlmChatProvider with the shared http client
        let make_provider = |backend, key: String, name: &str| -> Arc<LlmChatProvider> {
            Arc::new(
                LlmChatProvider::with_http_client(backend, key, name, http_client.clone())
                    .timeout_secs(timeout_secs),
            )
        };

        // OpenRouter (routes to many models, good default)
        if let Some(ref key) = self.openrouter_key {
            let provider = make_provider(LLMBackend::OpenRouter, key.clone(), "openrouter");
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }

        // Direct Anthropic
        if let Some(ref key) = self.anthropic_key {
            let provider = make_provider(LLMBackend::Anthropic, key.clone(), "anthropic");
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }

        // Direct OpenAI
        if let Some(ref key) = self.openai_key {
            let provider = make_provider(LLMBackend::OpenAI, key.clone(), "openai");
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }

        // Google (Gemini)
        if let Some(ref key) = self.google_key {
            let provider = make_provider(LLMBackend::Google, key.clone(), "google");
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }

        // Ollama
        if let Some(ref url) = self.ollama_url {
            let provider = Arc::new(
                LlmChatProvider::with_http_client(
                    LLMBackend::Ollama,
                    "ollama",
                    "ollama",
                    http_client.clone(),
                )
                .timeout_secs(timeout_secs)
                .ollama_url(url.clone()),
            );
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }

        // Build tokenizer registry for local inference
        #[cfg(feature = "local-inference")]
        let tokenizer_registry = {
            let mut registry = crate::tokenizer::TokenizerRegistry::new();
            for (pattern, source) in self.tokenizer_mappings {
                registry.register(&pattern, source);
            }
            Arc::new(registry)
        };

        let model_registry = crate::registry::ModelRegistry::with_embedded_seed();
        let model_cache = Arc::new(ModelCache::new());

        Ok(EmbeddedGateway::new(
            registry,
            model_registry,
            model_cache,
            #[cfg(feature = "local-inference")]
            model_manager,
            #[cfg(feature = "local-inference")]
            tokenizer_registry,
        ))
    }
}

impl Default for RatatoskrBuilder {
    fn default() -> Self {
        Self::new()
    }
}
