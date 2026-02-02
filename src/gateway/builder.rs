//! Builder for configuring gateway instances

use super::EmbeddedGateway;
use crate::{RatatoskrError, Result};

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
    #[cfg(feature = "huggingface")]
    fn has_capability_provider(&self) -> bool {
        self.huggingface_key.is_some()
    }

    #[cfg(not(feature = "huggingface"))]
    fn has_capability_provider(&self) -> bool {
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

        #[cfg(feature = "huggingface")]
        let router = if huggingface.is_some() {
            super::routing::CapabilityRouter::new().with_huggingface()
        } else {
            super::routing::CapabilityRouter::new()
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
            #[cfg(feature = "huggingface")]
            router,
        ))
    }
}

impl Default for RatatoskrBuilder {
    fn default() -> Self {
        Self::new()
    }
}
