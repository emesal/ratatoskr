//! Model information and status types.
//!
//! Types for describing available models, their capabilities, and runtime status.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::parameter::{ParameterAvailability, ParameterName};

/// A capability that a model or gateway may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelCapability {
    /// Multi-turn chat conversations.
    Chat,
    /// Streaming chat responses.
    ChatStreaming,
    /// Single-turn text generation.
    Generate,
    /// Tool/function calling support.
    ToolUse,
    /// Text embeddings.
    Embed,
    /// Natural language inference.
    Nli,
    /// Zero-shot classification.
    Classify,
    /// Stance detection.
    Stance,
    /// Token counting for models.
    TokenCounting,
    /// Local inference (no API calls needed).
    LocalInference,
}

/// Information about an available model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "all-MiniLM-L6-v2", "anthropic/claude-sonnet-4").
    pub id: String,
    /// Provider name (e.g., "fastembed", "huggingface", "openrouter").
    pub provider: String,
    /// Capabilities this model supports.
    pub capabilities: Vec<ModelCapability>,
    /// Maximum context window in tokens (if known).
    pub context_window: Option<usize>,
    /// Embedding dimensions (for embedding models).
    pub dimensions: Option<usize>,
}

impl ModelInfo {
    /// Create new model info with required fields.
    pub fn new(id: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            provider: provider.into(),
            capabilities: Vec::new(),
            context_window: None,
            dimensions: None,
        }
    }

    /// Add a capability to this model.
    pub fn with_capability(mut self, cap: ModelCapability) -> Self {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
        }
        self
    }

    /// Set the context window size.
    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Set the embedding dimensions.
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

/// Runtime status of a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelStatus {
    /// Model is available and can be loaded.
    Available,
    /// Model is currently being loaded.
    Loading,
    /// Model is loaded and ready to use.
    Ready,
    /// Model is not available.
    Unavailable {
        /// Reason for unavailability.
        reason: String,
    },
}

/// Cost tier for autoconfig presets.
///
/// Maps to a price-quality tradeoff: consumers choose the tier, and the
/// registry resolves a concrete model ID per capability.
///
/// Ordered from cheapest (`Free`) to most expensive (`Premium`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CostTier {
    /// Free-tier models (community/experimental, e.g. `:free` suffixed).
    Free,
    /// Low-cost models optimised for throughput over quality.
    Budget,
    /// High-quality models for demanding tasks.
    Premium,
}

impl std::fmt::Display for CostTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Free => write!(f, "free"),
            Self::Budget => write!(f, "budget"),
            Self::Premium => write!(f, "premium"),
        }
    }
}

impl std::str::FromStr for CostTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "free" => Ok(Self::Free),
            "budget" => Ok(Self::Budget),
            "premium" => Ok(Self::Premium),
            _ => Err(format!(
                "unknown cost tier: '{s}' (expected: free, budget, premium)"
            )),
        }
    }
}

impl ModelStatus {
    /// Create an unavailable status with a reason.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    /// Check if the model is usable (ready or available).
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Available | Self::Ready)
    }
}

/// Pricing information for a model (cost per million tokens).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingInfo {
    /// Cost per million prompt/input tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cost_per_mtok: Option<f64>,
    /// Cost per million completion/output tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_cost_per_mtok: Option<f64>,
}

/// Extended model metadata including parameter availability and pricing.
///
/// This is the primary type returned by the model registry. It extends
/// [`ModelInfo`] with parameter constraints and cost information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Basic model information.
    pub info: ModelInfo,
    /// Per-parameter availability and constraints.
    #[serde(default)]
    pub parameters: HashMap<ParameterName, ParameterAvailability>,
    /// Pricing information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingInfo>,
    /// Maximum output tokens (distinct from context window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
}

impl ModelMetadata {
    /// Create metadata from a [`ModelInfo`] with empty parameter map.
    pub fn from_info(info: ModelInfo) -> Self {
        Self {
            info,
            parameters: HashMap::new(),
            pricing: None,
            max_output_tokens: None,
        }
    }

    /// Add a parameter declaration.
    pub fn with_parameter(
        mut self,
        name: ParameterName,
        availability: ParameterAvailability,
    ) -> Self {
        self.parameters.insert(name, availability);
        self
    }

    /// Set pricing information.
    pub fn with_pricing(mut self, pricing: PricingInfo) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Set maximum output tokens.
    pub fn with_max_output_tokens(mut self, max: usize) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// Merge parameter overrides into this metadata.
    ///
    /// Override values replace existing entries; base entries not present
    /// in the override are preserved. Used for the registry merge strategy
    /// (live data overrides embedded defaults).
    pub fn merge_parameters(
        mut self,
        overrides: HashMap<ParameterName, ParameterAvailability>,
    ) -> Self {
        for (name, avail) in overrides {
            self.parameters.insert(name, avail);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_builder() {
        let info = ModelInfo::new("test-model", "test-provider")
            .with_capability(ModelCapability::Embed)
            .with_capability(ModelCapability::Nli)
            .with_dimensions(384);

        assert_eq!(info.id, "test-model");
        assert_eq!(info.provider, "test-provider");
        assert_eq!(info.capabilities.len(), 2);
        assert!(info.capabilities.contains(&ModelCapability::Embed));
        assert_eq!(info.dimensions, Some(384));
    }

    #[test]
    fn model_info_no_duplicate_capabilities() {
        let info = ModelInfo::new("test", "test")
            .with_capability(ModelCapability::Chat)
            .with_capability(ModelCapability::Chat);

        assert_eq!(info.capabilities.len(), 1);
    }

    #[test]
    fn model_status_is_usable() {
        assert!(ModelStatus::Available.is_usable());
        assert!(ModelStatus::Ready.is_usable());
        assert!(!ModelStatus::Loading.is_usable());
        assert!(!ModelStatus::unavailable("no RAM").is_usable());
    }
}
