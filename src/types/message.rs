//! Message types for chat conversations

use super::tool::ToolCall;
use serde::{Deserialize, Serialize};

/// Role of a message participant
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool { tool_call_id: String },
}

/// Message content (extensible for future multimodal)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    // Future: Parts(Vec<ContentPart>) for images, etc.
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl MessageContent {
    /// Get the text content, if any
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
        }
    }
}

/// A chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Create an assistant message with tool calls
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.map(MessageContent::Text).unwrap_or_default(),
            tool_calls: Some(tool_calls),
            name: None,
        }
    }

    /// Create a tool result message
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool {
                tool_call_id: tool_call_id.into(),
            },
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Set the name field (for multi-agent scenarios)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}
