//! Parameter validation policy types.

use serde::{Deserialize, Serialize};

/// How to handle unsupported parameters in requests.
///
/// When a provider declares its supported parameters via `supported_chat_parameters()`,
/// the registry can validate incoming requests against that list. This policy controls
/// what happens when a request contains unsupported parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterValidationPolicy {
    /// Log a warning and proceed without the unsupported parameter.
    ///
    /// This is the default — graceful degradation with visibility.
    #[default]
    Warn,

    /// Return an `UnsupportedParameter` error.
    ///
    /// Strict mode for consumers that want explicit failures.
    Error,

    /// Silently ignore unsupported parameters (legacy behaviour).
    ///
    /// Use this for backwards compatibility with providers that don't
    /// declare their parameters yet.
    Ignore,
}
