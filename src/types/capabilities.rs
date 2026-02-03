//! Gateway capability reporting

use serde::{Deserialize, Serialize};

/// What capabilities a gateway supports.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    pub embeddings: bool,
    /// Natural language inference.
    pub nli: bool,
    /// Zero-shot classification.
    pub classification: bool,
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
            embeddings: true,
            nli: true,
            classification: true,
            stance: true,
            token_counting: true,
            local_inference: true,
        }
    }

    /// Local inference only capabilities (no API needed)
    pub fn local_only() -> Self {
        Self {
            embeddings: true,
            nli: true,
            token_counting: true,
            local_inference: true,
            ..Default::default()
        }
    }

    /// HuggingFace API capabilities (embeddings, NLI, classification)
    pub fn huggingface_only() -> Self {
        Self {
            embeddings: true,
            nli: true,
            classification: true,
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
            embeddings: self.embeddings || other.embeddings,
            nli: self.nli || other.nli,
            classification: self.classification || other.classification,
            stance: self.stance || other.stance,
            token_counting: self.token_counting || other.token_counting,
            local_inference: self.local_inference || other.local_inference,
        }
    }
}
