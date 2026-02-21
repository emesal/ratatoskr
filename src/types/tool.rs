//! Tool types for function calling

use crate::RatatoskrError;
use serde::{Deserialize, Serialize};

/// Tool definition for function calling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Optional cache control directive (e.g. `{"type": "ephemeral"}` for Anthropic prompt caching).
    /// When set, the tool definition is eligible for prompt-cache hits, reducing cost on repeated calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            cache_control: None,
        }
    }

    /// Set a cache control directive for this tool (e.g. `json!({"type": "ephemeral"})`).
    #[must_use]
    pub fn with_cache_control(mut self, cache_control: serde_json::Value) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Convert from OpenAI-format JSON
impl TryFrom<&serde_json::Value> for ToolDefinition {
    type Error = RatatoskrError;

    fn try_from(value: &serde_json::Value) -> std::result::Result<Self, Self::Error> {
        let function = value
            .get("function")
            .ok_or_else(|| RatatoskrError::InvalidInput("missing 'function' field".into()))?;

        Ok(Self {
            name: function["name"]
                .as_str()
                .ok_or_else(|| RatatoskrError::InvalidInput("missing function name".into()))?
                .to_string(),
            description: function["description"].as_str().unwrap_or("").to_string(),
            parameters: function
                .get("parameters")
                .cloned()
                .unwrap_or(serde_json::json!({})),
            cache_control: value.get("cache_control").cloned(),
        })
    }
}

/// A tool call made by the model
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Parse the arguments as JSON
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(
        &self,
    ) -> std::result::Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Tool choice configuration
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        name: String,
    },
}
