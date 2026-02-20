//! Gateway capability reporting

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::model::ModelCapability;

/// What capabilities a gateway supports.
///
/// A typed set of [`ModelCapability`] values. Use [`Capabilities::has`] to query
/// individual capabilities and the factory methods for common configurations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities(HashSet<ModelCapability>);

impl Capabilities {
    /// Returns `true` if this set contains the given capability.
    pub fn has(&self, cap: ModelCapability) -> bool {
        self.0.contains(&cap)
    }

    /// Inserts a capability into the set.
    pub fn insert(&mut self, cap: ModelCapability) {
        self.0.insert(cap);
    }

    /// Returns an iterator over the capabilities in this set.
    pub fn iter(&self) -> impl Iterator<Item = &ModelCapability> {
        self.0.iter()
    }

    /// Merges two capability sets using union (OR) logic.
    pub fn merge(&self, other: &Self) -> Self {
        Self(self.0.union(&other.0).copied().collect())
    }

    /// Chat-capable gateway (chat, streaming, generate, tool use).
    pub fn chat_only() -> Self {
        Self::from_iter([
            ModelCapability::Chat,
            ModelCapability::ChatStreaming,
            ModelCapability::Generate,
            ModelCapability::ToolUse,
        ])
    }

    /// All capabilities enabled.
    pub fn full() -> Self {
        Self::from_iter([
            ModelCapability::Chat,
            ModelCapability::ChatStreaming,
            ModelCapability::Generate,
            ModelCapability::ToolUse,
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::Classify,
            ModelCapability::Stance,
            ModelCapability::TokenCounting,
            ModelCapability::LocalInference,
        ])
    }

    /// Local inference only (no API calls needed).
    pub fn local_only() -> Self {
        Self::from_iter([
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::TokenCounting,
            ModelCapability::LocalInference,
        ])
    }

    /// HuggingFace API capabilities (embeddings, NLI, classification).
    pub fn huggingface_only() -> Self {
        Self::from_iter([
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::Classify,
        ])
    }
}

impl FromIterator<ModelCapability> for Capabilities {
    fn from_iter<I: IntoIterator<Item = ModelCapability>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<HashSet<ModelCapability>> for Capabilities {
    fn from(set: HashSet<ModelCapability>) -> Self {
        Self(set)
    }
}

impl From<Capabilities> for HashSet<ModelCapability> {
    fn from(caps: Capabilities) -> Self {
        caps.0
    }
}
