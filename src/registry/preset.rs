//! Preset entry types: model ID + optional default generation parameters.

use serde::{Deserialize, Serialize};

use crate::types::{ReasoningConfig, ResponseFormat, ToolChoice};

/// A single preset: model ID with optional default generation parameters.
///
/// Deserialises from either a bare string (`"model-id"`) or an object
/// (`{ "model": "...", "parameters": { ... } }`).  The bare string form
/// provides backwards compatibility with the pre-parameters format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PresetEntry {
    /// Full preset with model and optional parameters.
    WithParams {
        model: String,
        #[serde(default, skip_serializing_if = "PresetParameters::is_empty")]
        parameters: PresetParameters,
    },
    /// Legacy bare string (just a model ID).
    Bare(String),
}

impl PresetEntry {
    /// The model identifier this preset resolves to.
    pub fn model(&self) -> &str {
        match self {
            Self::WithParams { model, .. } => model,
            Self::Bare(id) => id,
        }
    }

    /// Optional preset parameters, `None` for bare entries or empty params.
    pub fn parameters(&self) -> Option<&PresetParameters> {
        match self {
            Self::WithParams { parameters, .. } if !parameters.is_empty() => Some(parameters),
            _ => None,
        }
    }
}

/// Default generation parameters attached to a preset.
///
/// All fields are optional — only set fields act as defaults.  When applied,
/// they fill `None` slots in the caller's `ChatOptions` / `GenerateOptions`
/// but never overwrite explicitly-set values.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PresetParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_provider_options: Option<serde_json::Value>,
}

impl PresetParameters {
    /// True if no parameters are set.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.max_tokens.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
            && self.seed.is_none()
            && self.stop.is_none()
            && self.reasoning.is_none()
            && self.tool_choice.is_none()
            && self.parallel_tool_calls.is_none()
            && self.response_format.is_none()
            && self.cache_prompt.is_none()
            && self.raw_provider_options.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_string_deserializes() {
        let json = r#""some/model""#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        assert!(entry.parameters().is_none());
    }

    #[test]
    fn with_params_deserializes() {
        let json = r#"{"model": "some/model", "parameters": {"temperature": 0.3, "top_p": 0.95}}"#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        let params = entry.parameters().unwrap();
        assert_eq!(params.temperature, Some(0.3));
        assert_eq!(params.top_p, Some(0.95));
        assert!(params.max_tokens.is_none());
    }

    #[test]
    fn with_params_no_parameters_key() {
        let json = r#"{"model": "some/model"}"#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        // WithParams variant but empty parameters
        assert!(entry.parameters().map_or(true, |p| p.is_empty()));
    }

    #[test]
    fn bare_string_round_trips() {
        let entry = PresetEntry::Bare("x/model".to_owned());
        let json = serde_json::to_string(&entry).unwrap();
        let back: PresetEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model(), "x/model");
    }

    #[test]
    fn with_params_round_trips() {
        let entry = PresetEntry::WithParams {
            model: "x/model".to_owned(),
            parameters: PresetParameters {
                temperature: Some(0.5),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: PresetEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model(), "x/model");
        assert_eq!(back.parameters().unwrap().temperature, Some(0.5));
    }

    #[test]
    fn preset_parameters_is_empty() {
        assert!(PresetParameters::default().is_empty());
        assert!(!PresetParameters {
            temperature: Some(0.5),
            ..Default::default()
        }
        .is_empty());
    }
}
