//! Preset entry types: model ID + optional default generation parameters.

use serde::{Deserialize, Serialize};

use crate::types::{ChatOptions, GenerateOptions, ReasoningConfig, ResponseFormat, ToolChoice};

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
        parameters: Box<PresetParameters>,
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

    /// Apply these parameters as defaults to `ChatOptions`.
    ///
    /// Fills `None` fields from preset values; never overwrites `Some`.
    pub fn apply_defaults_to_chat(&self, opts: &mut ChatOptions) {
        macro_rules! fill {
            ($field:ident) => {
                if opts.$field.is_none() {
                    opts.$field = self.$field.clone();
                }
            };
        }
        fill!(temperature);
        fill!(top_p);
        fill!(top_k);
        fill!(max_tokens);
        fill!(frequency_penalty);
        fill!(presence_penalty);
        fill!(seed);
        fill!(stop);
        fill!(reasoning);
        fill!(tool_choice);
        fill!(parallel_tool_calls);
        fill!(response_format);
        fill!(cache_prompt);
        fill!(raw_provider_options);
    }

    /// Apply these parameters as defaults to `GenerateOptions`.
    ///
    /// Fills `None` fields from preset values; never overwrites `Some`.
    /// Fields absent from `GenerateOptions` (e.g. `tool_choice`) are
    /// silently ignored.
    pub fn apply_defaults_to_generate(&self, opts: &mut GenerateOptions) {
        macro_rules! fill {
            ($field:ident) => {
                if opts.$field.is_none() {
                    opts.$field = self.$field.clone();
                }
            };
        }
        fill!(temperature);
        fill!(top_p);
        fill!(top_k);
        fill!(max_tokens);
        fill!(frequency_penalty);
        fill!(presence_penalty);
        fill!(seed);
        fill!(reasoning);
        // `stop` in PresetParameters maps to `stop_sequences` in GenerateOptions.
        // Only fill if the caller left the vec empty.
        if opts.stop_sequences.is_empty()
            && let Some(stop) = &self.stop
        {
            opts.stop_sequences = stop.clone();
        }
    }
}

/// Result of resolving a preset: the concrete model ID and any default parameters.
///
/// Returned by [`crate::ModelGateway::resolve_preset`].
#[derive(Debug, Clone, PartialEq)]
pub struct PresetResolution {
    /// The resolved model identifier.
    pub model: String,
    /// Optional default parameters for this preset.
    pub parameters: Option<PresetParameters>,
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
        assert!(entry.parameters().is_none_or(|p| p.is_empty()));
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
            parameters: Box::new(PresetParameters {
                temperature: Some(0.5),
                ..Default::default()
            }),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: PresetEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model(), "x/model");
        assert_eq!(back.parameters().unwrap().temperature, Some(0.5));
    }

    #[test]
    fn preset_parameters_is_empty() {
        assert!(PresetParameters::default().is_empty());
        assert!(
            !PresetParameters {
                temperature: Some(0.5),
                ..Default::default()
            }
            .is_empty()
        );
    }

    // ===== Task 2: apply_defaults =====

    use crate::types::{ChatOptions, GenerateOptions};

    #[test]
    fn apply_defaults_fills_none_chat() {
        let params = PresetParameters {
            temperature: Some(0.3),
            top_p: Some(0.95),
            max_tokens: Some(4096),
            ..Default::default()
        };
        let mut opts = ChatOptions::new("x");
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts.temperature, Some(0.3));
        assert_eq!(opts.top_p, Some(0.95));
        assert_eq!(opts.max_tokens, Some(4096));
    }

    #[test]
    fn apply_defaults_preserves_caller_chat() {
        let params = PresetParameters {
            temperature: Some(0.3),
            top_p: Some(0.95),
            ..Default::default()
        };
        let mut opts = ChatOptions::new("x");
        opts.temperature = Some(0.7); // caller set this
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts.temperature, Some(0.7)); // caller wins
        assert_eq!(opts.top_p, Some(0.95)); // preset fills
    }

    #[test]
    fn apply_defaults_noop_when_empty() {
        let params = PresetParameters::default();
        let mut opts = ChatOptions::new("x");
        opts.temperature = Some(0.5);
        let before = opts.clone();
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts, before);
    }

    #[test]
    fn apply_defaults_fills_none_generate() {
        let params = PresetParameters {
            temperature: Some(0.8),
            top_k: Some(50),
            ..Default::default()
        };
        let mut opts = GenerateOptions::new("x");
        params.apply_defaults_to_generate(&mut opts);
        assert_eq!(opts.temperature, Some(0.8));
        assert_eq!(opts.top_k, Some(50));
    }

    #[test]
    fn apply_defaults_preserves_caller_generate() {
        let params = PresetParameters {
            temperature: Some(0.8),
            ..Default::default()
        };
        let mut opts = GenerateOptions::new("x");
        opts.temperature = Some(0.1);
        params.apply_defaults_to_generate(&mut opts);
        assert_eq!(opts.temperature, Some(0.1));
    }
}
