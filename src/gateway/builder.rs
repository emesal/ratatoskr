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

    /// Set default timeout for all requests (seconds).
    pub fn timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = Some(secs);
        self
    }

    /// Build the gateway.
    pub fn build(self) -> Result<EmbeddedGateway> {
        // Must have at least one provider
        if self.openrouter_key.is_none()
            && self.anthropic_key.is_none()
            && self.openai_key.is_none()
            && self.google_key.is_none()
            && self.ollama_url.is_none()
        {
            return Err(RatatoskrError::NoProvider);
        }

        Ok(EmbeddedGateway::new(
            self.openrouter_key,
            self.anthropic_key,
            self.openai_key,
            self.google_key,
            self.ollama_url,
            self.default_timeout_secs.unwrap_or(120),
        ))
    }
}

impl Default for RatatoskrBuilder {
    fn default() -> Self {
        Self::new()
    }
}
