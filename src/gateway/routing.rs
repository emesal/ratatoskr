//! Capability-based routing for non-chat operations.
//!
//! This module determines which provider handles each capability.
//! The router is intentionally simple - just a lookup table.
//! This makes it easy to replace with programmable routing later.

/// Which provider handles embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedProvider {
    HuggingFace,
    // Future: OpenAI, Ollama, Cohere, etc.
}

/// Which provider handles NLI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NliProvider {
    HuggingFace,
    // Future: Local ONNX, etc.
}

/// Which provider handles zero-shot classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifyProvider {
    HuggingFace,
    // Future: Local ONNX, etc.
}

/// Routes capabilities to providers.
///
/// This is a simple lookup table. When a provider is configured via the builder,
/// it becomes the provider for its supported capabilities.
#[derive(Debug, Clone, Default)]
pub struct CapabilityRouter {
    embed: Option<EmbedProvider>,
    nli: Option<NliProvider>,
    classify: Option<ClassifyProvider>,
}

impl CapabilityRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure HuggingFace for all non-chat capabilities.
    pub fn with_huggingface(mut self) -> Self {
        self.embed = Some(EmbedProvider::HuggingFace);
        self.nli = Some(NliProvider::HuggingFace);
        self.classify = Some(ClassifyProvider::HuggingFace);
        self
    }

    /// Get the provider for embeddings.
    pub fn embed_provider(&self) -> Option<EmbedProvider> {
        self.embed
    }

    /// Get the provider for NLI.
    pub fn nli_provider(&self) -> Option<NliProvider> {
        self.nli
    }

    /// Get the provider for classification.
    pub fn classify_provider(&self) -> Option<ClassifyProvider> {
        self.classify
    }
}
