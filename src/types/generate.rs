//! Types for text generation (non-chat) operations.

use crate::types::options::ReasoningConfig;
use crate::types::parameter::ParameterName;
use crate::types::{FinishReason, Usage};
use serde::{Deserialize, Serialize};

/// Options for text generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Model to use for generation.
    pub model: String,

    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Sampling temperature (0.0 to 2.0).
    /// Higher values make output more random.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Sequences where generation should stop.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// Top-k sampling: only consider the k most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Penalise tokens based on frequency in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Penalise tokens based on whether they appear in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Seed for deterministic generation (where supported).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Reasoning configuration for extended thinking models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

impl GenerateOptions {
    /// Create options with the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            reasoning: None,
        }
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top_p.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Add a single stop sequence.
    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Set top-k sampling.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set seed for deterministic generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set reasoning configuration.
    pub fn reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning = Some(config);
        self
    }

    /// Returns the list of parameters that are set (not None) in these options.
    ///
    /// Used by the registry for validation against provider-declared parameters.
    pub fn set_parameters(&self) -> Vec<ParameterName> {
        let mut params = Vec::new();
        if self.max_tokens.is_some() {
            params.push(ParameterName::MaxTokens);
        }
        if self.temperature.is_some() {
            params.push(ParameterName::Temperature);
        }
        if self.top_p.is_some() {
            params.push(ParameterName::TopP);
        }
        if !self.stop_sequences.is_empty() {
            params.push(ParameterName::Stop);
        }
        if self.top_k.is_some() {
            params.push(ParameterName::TopK);
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
        if self.reasoning.is_some() {
            params.push(ParameterName::Reasoning);
        }
        params
    }
}

/// Response from text generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated text.
    pub text: String,

    /// Token usage information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Model used for generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Reason generation stopped.
    pub finish_reason: FinishReason,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[non_exhaustive]
pub enum GenerateEvent {
    /// Text chunk generated.
    #[serde(rename = "text")]
    Text(String),

    /// Generation complete.
    #[serde(rename = "done")]
    Done,
}
