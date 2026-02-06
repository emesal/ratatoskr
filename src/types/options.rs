//! Chat options and configuration types

use super::parameter::ParameterName;
use super::tool::ToolChoice;
use serde::{Deserialize, Serialize};

/// Options for chat requests (provider-agnostic)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    // Normalized cross-provider options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    // Escape hatch for truly provider-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_provider_options: Option<serde_json::Value>,
}

impl ChatOptions {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning = Some(config);
        self
    }

    pub fn cache_prompt(mut self, cache: bool) -> Self {
        self.cache_prompt = Some(cache);
        self
    }

    /// Returns the list of parameters that are set (not None) in these options.
    ///
    /// Used by the registry for validation against provider-declared parameters.
    pub fn set_parameters(&self) -> Vec<ParameterName> {
        let mut params = Vec::new();
        if self.temperature.is_some() {
            params.push(ParameterName::Temperature);
        }
        if self.max_tokens.is_some() {
            params.push(ParameterName::MaxTokens);
        }
        if self.top_p.is_some() {
            params.push(ParameterName::TopP);
        }
        if self.top_k.is_some() {
            params.push(ParameterName::TopK);
        }
        if self.stop.is_some() {
            params.push(ParameterName::Stop);
        }
        if self.frequency_penalty.is_some() {
            params.push(ParameterName::FrequencyPenalty);
        }
        if self.presence_penalty.is_some() {
            params.push(ParameterName::PresencePenalty);
        }
        if self.seed.is_some() {
            params.push(ParameterName::Seed);
        }
        if self.tool_choice.is_some() {
            params.push(ParameterName::ToolChoice);
        }
        if self.response_format.is_some() {
            params.push(ParameterName::ResponseFormat);
        }
        if self.cache_prompt.is_some() {
            params.push(ParameterName::CachePrompt);
        }
        if self.reasoning.is_some() {
            params.push(ParameterName::Reasoning);
        }
        params
    }
}

/// Reasoning configuration for extended thinking models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_from_output: Option<bool>,
}

/// Reasoning effort level
///
/// Matches OpenRouter's reasoning effort levels. Providers that support fewer
/// levels (e.g. only low/medium/high) should map to the nearest equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

/// Response format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}
