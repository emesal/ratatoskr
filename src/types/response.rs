//! Response and streaming event types

use super::tool::ToolCall;
use serde::{Deserialize, Serialize};

/// Non-streaming chat response
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub finish_reason: FinishReason,
}

/// Events emitted during streaming chat
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Text content chunk
    Content(String),

    /// Reasoning/thinking content (extended thinking models)
    Reasoning(String),

    /// Start of a tool call
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },

    /// Incremental tool call arguments
    ToolCallDelta { index: usize, arguments: String },

    /// A tool call's arguments are complete (does NOT end the stream)
    ToolCallEnd { index: usize },

    /// Usage statistics (typically at end of stream)
    Usage(Usage),

    /// Stream complete
    Done,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

/// Reason the model stopped generating
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}
