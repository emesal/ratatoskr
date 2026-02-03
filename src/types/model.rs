//! Model information and status types.
//!
//! Types for describing available models, their capabilities, and runtime status.

use serde::{Deserialize, Serialize};

/// A capability that a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    /// Multi-turn chat conversations.
    Chat,
    /// Single-turn text generation.
    Generate,
    /// Text embeddings.
    Embed,
    /// Natural language inference.
    Nli,
    /// Zero-shot classification.
    Classify,
    /// Stance detection.
    Stance,
}

/// Information about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
