//! Gateway capability reporting

use serde::{Deserialize, Serialize};

/// What capabilities a gateway supports
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capabilities {
    pub chat: bool,
    pub chat_streaming: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub classification: bool,
    pub token_counting: bool,
}

impl Capabilities {
    /// Phase 1 embedded gateway capabilities (chat only)
    pub fn chat_only() -> Self {
        Self {
            chat: true,
            chat_streaming: true,
            ..Default::default()
        }
    }

    /// Full capabilities (future)
    pub fn full() -> Self {
        Self {
            chat: true,
            chat_streaming: true,
            embeddings: true,
            nli: true,
            classification: true,
            token_counting: true,
        }
    }

    /// HuggingFace-only capabilities (embeddings, NLI, classification)
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
            embeddings: self.embeddings || other.embeddings,
            nli: self.nli || other.nli,
            classification: self.classification || other.classification,
            token_counting: self.token_counting || other.token_counting,
        }
    }
}
