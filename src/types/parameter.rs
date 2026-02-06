//! Parameter metadata types for model introspection.
//!
//! These types describe what parameters a model supports and their constraints.
//! Used by the model registry and `model_metadata()` for consumer introspection.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Well-known parameter names with a `Custom` escape hatch.
///
/// The well-known variants cover the standard parameter surface shared across
/// providers (OpenAI conventions). `Custom(String)` handles provider-specific
/// parameters without requiring a ratatoskr release.
///
/// Serializes as a flat string (e.g. `"temperature"`, `"logit_bias"`) so it
/// works both as JSON values and as JSON object keys in `HashMap<ParameterName, _>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParameterName {
    Temperature,
    TopP,
    TopK,
    MaxTokens,
    FrequencyPenalty,
    PresencePenalty,
    Seed,
    Stop,
    Reasoning,
    CachePrompt,
    ResponseFormat,
    ToolChoice,
    ParallelToolCalls,
    /// Provider-specific parameter not in the well-known set.
    Custom(String),
}

impl ParameterName {
    /// Canonical string representation matching `ChatOptions`/`GenerateOptions` field names.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Temperature => "temperature",
            Self::TopP => "top_p",
            Self::TopK => "top_k",
            Self::MaxTokens => "max_tokens",
            Self::FrequencyPenalty => "frequency_penalty",
            Self::PresencePenalty => "presence_penalty",
            Self::Seed => "seed",
            Self::Stop => "stop",
            Self::Reasoning => "reasoning",
            Self::CachePrompt => "cache_prompt",
            Self::ResponseFormat => "response_format",
            Self::ToolChoice => "tool_choice",
            Self::ParallelToolCalls => "parallel_tool_calls",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for ParameterName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ParameterName {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "temperature" => Self::Temperature,
            "top_p" => Self::TopP,
            "top_k" => Self::TopK,
            "max_tokens" => Self::MaxTokens,
            "frequency_penalty" => Self::FrequencyPenalty,
            "presence_penalty" => Self::PresencePenalty,
            "seed" => Self::Seed,
            "stop" => Self::Stop,
            "reasoning" => Self::Reasoning,
            "cache_prompt" => Self::CachePrompt,
            "response_format" => Self::ResponseFormat,
            "tool_choice" => Self::ToolChoice,
            "parallel_tool_calls" => Self::ParallelToolCalls,
            other => Self::Custom(other.to_string()),
        })
    }
}

impl Serialize for ParameterName {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ParameterName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        // FromStr is infallible for ParameterName
        Ok(s.parse().unwrap())
    }
}

/// Numeric range constraints for a parameter.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParameterRange {
    /// Minimum allowed value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum allowed value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Default value if not specified by the consumer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
}

impl ParameterRange {
    /// Create an empty range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum value.
    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value.
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set default value.
    pub fn default_value(mut self, default: f64) -> Self {
        self.default = Some(default);
        self
    }
}

/// How a parameter is exposed for a given model.
///
/// This tells consumers whether they can set a parameter, what range it accepts,
/// or whether it's fixed/unsupported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
pub enum ParameterAvailability {
    /// Consumer can set this freely within range.
    Mutable {
        #[serde(default)]
        range: ParameterRange,
    },
    /// Value is fixed by the provider/model.
    ReadOnly { value: serde_json::Value },
    /// Parameter exists but constraints are unknown.
    Opaque,
    /// Parameter is not supported by this model.
    Unsupported,
}

impl ParameterAvailability {
    /// Whether this parameter is supported (anything other than `Unsupported`).
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}
