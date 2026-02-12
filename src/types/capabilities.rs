//! Gateway capability reporting

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::model::ModelCapability;

/// What capabilities a gateway supports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Multi-turn chat conversations.
    pub chat: bool,
    /// Streaming chat responses.
    pub chat_streaming: bool,
    /// Single-turn text generation.
    pub generate: bool,
    /// Tool/function calling support.
    pub tool_use: bool,
    /// Text embeddings.
    #[serde(alias = "embeddings")]
    pub embed: bool,
    /// Natural language inference.
    pub nli: bool,
    /// Zero-shot classification.
    #[serde(alias = "classification")]
    pub classify: bool,
    /// Stance detection toward target topics.
    pub stance: bool,
    /// Token counting for models.
    pub token_counting: bool,
    /// Local inference (no API calls needed).
    pub local_inference: bool,
}

impl Capabilities {
    /// Phase 1 embedded gateway capabilities (chat only)
    pub fn chat_only() -> Self {
        Self {
            chat: true,
            chat_streaming: true,
            generate: true,
            tool_use: true,
            ..Default::default()
        }
    }

    /// Full capabilities (all features).
    pub fn full() -> Self {
        Self {
            chat: true,
            chat_streaming: true,
            generate: true,
            tool_use: true,
            embed: true,
            nli: true,
            classify: true,
            stance: true,
            token_counting: true,
            local_inference: true,
        }
    }

    /// Local inference only capabilities (no API needed)
    pub fn local_only() -> Self {
        Self {
            embed: true,
            nli: true,
            token_counting: true,
            local_inference: true,
            ..Default::default()
        }
    }

    /// HuggingFace API capabilities (embeddings, NLI, classification)
    pub fn huggingface_only() -> Self {
        Self {
            embed: true,
            nli: true,
            classify: true,
            ..Default::default()
        }
    }

    /// Merge capabilities using OR logic (combines two capability sets).
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            chat: self.chat || other.chat,
            chat_streaming: self.chat_streaming || other.chat_streaming,
            generate: self.generate || other.generate,
            tool_use: self.tool_use || other.tool_use,
            embed: self.embed || other.embed,
            nli: self.nli || other.nli,
            classify: self.classify || other.classify,
            stance: self.stance || other.stance,
            token_counting: self.token_counting || other.token_counting,
            local_inference: self.local_inference || other.local_inference,
        }
    }
}

impl From<&Capabilities> for HashSet<ModelCapability> {
    fn from(caps: &Capabilities) -> Self {
        let pairs = [
            (caps.chat, ModelCapability::Chat),
            (caps.chat_streaming, ModelCapability::ChatStreaming),
            (caps.generate, ModelCapability::Generate),
            (caps.tool_use, ModelCapability::ToolUse),
            (caps.embed, ModelCapability::Embed),
            (caps.nli, ModelCapability::Nli),
            (caps.classify, ModelCapability::Classify),
            (caps.stance, ModelCapability::Stance),
            (caps.token_counting, ModelCapability::TokenCounting),
            (caps.local_inference, ModelCapability::LocalInference),
        ];
        pairs
            .into_iter()
            .filter(|(enabled, _)| *enabled)
            .map(|(_, cap)| cap)
            .collect()
    }
}

impl From<&HashSet<ModelCapability>> for Capabilities {
    fn from(set: &HashSet<ModelCapability>) -> Self {
        Self {
            chat: set.contains(&ModelCapability::Chat),
            chat_streaming: set.contains(&ModelCapability::ChatStreaming),
            generate: set.contains(&ModelCapability::Generate),
            tool_use: set.contains(&ModelCapability::ToolUse),
            embed: set.contains(&ModelCapability::Embed),
            nli: set.contains(&ModelCapability::Nli),
            classify: set.contains(&ModelCapability::Classify),
            stance: set.contains(&ModelCapability::Stance),
            token_counting: set.contains(&ModelCapability::TokenCounting),
            local_inference: set.contains(&ModelCapability::LocalInference),
        }
    }
}
