//! Ratatoskr error types

use std::time::Duration;

/// Ratatoskr error types
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum RatatoskrError {
    // Provider/network errors
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// Rate limited by the provider.
    ///
    /// `retry_after` is populated when the provider or gRPC status encodes a
    /// duration. Currently `None` for errors originating from the `llm` crate
    /// (which doesn't expose `Retry-After` headers yet); populated for
    /// HuggingFace 429 responses and gRPC round-trips.
    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    #[error("authentication failed")]
    AuthenticationFailed,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("preset not found: tier '{tier}', capability '{capability}'")]
    PresetNotFound { tier: String, capability: String },

    // Streaming errors
    #[error("stream error: {0}")]
    Stream(String),

    // Data errors
    #[error("JSON error: {0}")]
    Json(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    // Configuration errors
    #[error("no provider configured")]
    NoProvider,

    /// Provider cannot handle this model (wrong model, RAM constrained, etc.)
    /// The registry should try the next provider in the fallback chain.
    #[error("model not available from this provider")]
    ModelNotAvailable,

    #[error("configuration error: {0}")]
    Configuration(String),

    #[error("operation not implemented: {0}")]
    NotImplemented(String),

    #[error("provider does not support this operation")]
    Unsupported,

    /// Parameter is not supported by the target model/provider.
    ///
    /// Returned when validation policy is `Error` and a request contains
    /// parameters the provider doesn't support.
    #[error("unsupported parameter '{param}' for model '{model}' (provider: {provider})")]
    UnsupportedParameter {
        param: String,
        model: String,
        provider: String,
    },

    // Data processing errors
    #[error("data error: {0}")]
    DataError(String),

    // Soft errors
    #[error("empty response from model")]
    EmptyResponse,

    #[error("content filtered: {reason}")]
    ContentFiltered { reason: String },

    #[error("context length exceeded: {limit} tokens")]
    ContextLengthExceeded { limit: usize },

    // Wrapped llm crate error
    #[error("LLM error: {0}")]
    Llm(String),
}

impl RatatoskrError {
    /// Whether this error is transient and the request may succeed on retry.
    ///
    /// Used by `RetryingProvider` to decide whether to retry a failed request.
    /// Permanent errors (auth, validation, model not found) return `false`.
    pub fn is_transient(&self) -> bool {
        match self {
            // Always transient
            Self::RateLimited { .. } => true,
            Self::Http(_) => true,
            Self::Stream(_) => true,
            Self::EmptyResponse => true,

            // Server errors are transient; client errors are not
            Self::Api { status, .. } => *status >= 500,

            // Heuristic: network-sounding messages are transient
            Self::Llm(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("timeout")
                    || lower.contains("connection")
                    || lower.contains("reset")
                    || lower.contains("refused")
            }

            // Everything else is permanent
            Self::Json(_)
            | Self::AuthenticationFailed
            | Self::ModelNotFound(_)
            | Self::PresetNotFound { .. }
            | Self::InvalidInput(_)
            | Self::NoProvider
            | Self::ModelNotAvailable
            | Self::Configuration(_)
            | Self::NotImplemented(_)
            | Self::Unsupported
            | Self::UnsupportedParameter { .. }
            | Self::DataError(_)
            | Self::ContentFiltered { .. }
            | Self::ContextLengthExceeded { .. } => false,
        }
    }

    /// For `RateLimited` errors, the duration the provider suggests waiting.
    ///
    /// Returns `None` for non-rate-limit errors or when the provider didn't
    /// specify a retry-after duration.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }
}

impl From<serde_json::Error> for RatatoskrError {
    fn from(err: serde_json::Error) -> Self {
        RatatoskrError::Json(err.to_string())
    }
}
impl From<llm::error::LLMError> for RatatoskrError {
    fn from(err: llm::error::LLMError) -> Self {
        // Map llm errors to our error types
        let msg = err.to_string();
        if msg.contains("rate limit") || msg.contains("429") {
            RatatoskrError::RateLimited { retry_after: None }
        } else if msg.contains("authentication")
            || msg.contains("401")
            || msg.contains("invalid api key")
        {
            RatatoskrError::AuthenticationFailed
        } else if msg.contains("not found") || msg.contains("404") {
            RatatoskrError::ModelNotFound(msg)
        } else {
            RatatoskrError::Llm(msg)
        }
    }
}

/// Result type alias for Ratatoskr operations
pub type Result<T> = std::result::Result<T, RatatoskrError>;
