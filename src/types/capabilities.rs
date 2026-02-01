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
}
