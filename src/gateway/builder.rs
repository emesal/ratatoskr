//! Builder for configuring gateway instances

use super::EmbeddedGateway;
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
        // Must have at least one provider (chat or capability)
        if !self.has_chat_provider() && !self.has_capability_provider() {
            return Err(RatatoskrError::NoProvider);
        }

        #[cfg(feature = "huggingface")]
        let huggingface = self
            .huggingface_key
            .map(crate::providers::HuggingFaceClient::new);

        // Build the capability router
        let mut router = super::routing::CapabilityRouter::new();

        #[cfg(feature = "huggingface")]
        if huggingface.is_some() {
            router = router.with_huggingface();
        }

        #[cfg(feature = "local-inference")]
        {
            if self.local_embedding_model.is_some() {
                router = router.with_local_embeddings();
            }
            if self.local_nli_model.is_some() {
                router = router.with_local_nli();
            }
        }

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
                default_device: self.device,
                ram_budget: self.ram_budget,
            };
            std::sync::Arc::new(crate::model::ModelManager::new(config))
        };

        // Build tokenizer registry for local inference
        #[cfg(feature = "local-inference")]
        let tokenizer_registry = {
            let mut registry = crate::tokenizer::TokenizerRegistry::new();
            for (pattern, source) in self.tokenizer_mappings {
                registry.register(&pattern, source);
            }
            std::sync::Arc::new(registry)
        };

        Ok(EmbeddedGateway::new(
            self.openrouter_key,
            self.anthropic_key,
            self.openai_key,
            self.google_key,
            self.ollama_url,
            self.default_timeout_secs.unwrap_or(120),
            #[cfg(feature = "huggingface")]
            huggingface,
            router,
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
