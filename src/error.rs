//! Ratatoskr error types

use std::time::Duration;

/// Ratatoskr error types
#[derive(Debug, thiserror::Error)]
pub enum RatatoskrError {
    // Provider/network errors
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    #[error("authentication failed")]
    AuthenticationFailed,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    // Streaming errors
    #[error("stream error: {0}")]
    Stream(String),

    // Data errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

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
    NotImplemented(&'static str),

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
