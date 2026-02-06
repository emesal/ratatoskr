//! OpenRouter model metadata types and conversion.
//!
//! Handles the `/api/v1/models` response from the OpenRouter API and
//! converts it into [`ModelMetadata`]. This module is separate from
//! [`llm_chat`](super::llm_chat) to keep chat/generate logic focused.

use std::collections::HashMap;

use serde::Deserialize;

use crate::types::{
    ModelCapability, ModelInfo, ModelMetadata, ParameterAvailability, ParameterName, ParameterRange,
    PricingInfo,
};

/// OpenRouter `/api/v1/models` list response.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    pub data: Vec<ModelEntry>,
}

/// A single model entry from the OpenRouter API.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub context_length: Option<usize>,
    #[serde(default)]
    pub pricing: Option<PricingEntry>,
    #[serde(default)]
    pub top_provider: Option<TopProvider>,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub architecture: Option<Architecture>,
}

/// Pricing from the OpenRouter API (string-encoded decimals, cost per token).
#[derive(Debug, Deserialize)]
pub(crate) struct PricingEntry {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
}

/// Top provider info from the OpenRouter API.
#[derive(Debug, Deserialize)]
pub(crate) struct TopProvider {
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
}

/// Architecture metadata from the OpenRouter API.
///
/// Deserialized for forward compatibility; not yet surfaced in [`ModelMetadata`].
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct Architecture {
    #[serde(default)]
    pub modality: Option<String>,
}

/// Convert an OpenRouter model entry into our [`ModelMetadata`].
pub(crate) fn into_model_metadata(entry: ModelEntry) -> ModelMetadata {
    let mut info = ModelInfo::new(&entry.id, "openrouter")
        .with_capability(ModelCapability::Chat);

    if let Some(ctx) = entry.context_length {
        info = info.with_context_window(ctx);
    }

    let pricing = entry.pricing.and_then(|p| {
        let prompt = parse_per_token_to_per_mtok(p.prompt.as_deref())?;
        let completion = parse_per_token_to_per_mtok(p.completion.as_deref())?;
        Some(PricingInfo {
            prompt_cost_per_mtok: Some(prompt),
            completion_cost_per_mtok: Some(completion),
        })
    });

    let max_output_tokens = entry
        .top_provider
        .and_then(|tp| tp.max_completion_tokens);

    let parameters = build_parameter_map(&entry.supported_parameters);

    let mut metadata = ModelMetadata::from_info(info);
    metadata.pricing = pricing;
    metadata.max_output_tokens = max_output_tokens;
    metadata.parameters = parameters;
    metadata
}

/// Parse OpenRouter per-token price string to per-million-token cost.
///
/// OpenRouter prices are decimal strings representing cost per single token
/// (e.g. `"0.000005"` = $5 per million tokens).
fn parse_per_token_to_per_mtok(s: Option<&str>) -> Option<f64> {
    let per_token: f64 = s?.parse().ok()?;
    Some(per_token * 1_000_000.0)
}

/// Map OpenRouter `supported_parameters` strings to our parameter availability.
fn build_parameter_map(params: &[String]) -> HashMap<ParameterName, ParameterAvailability> {
    // Standard ranges for well-known parameters (OpenAI conventions)
    let standard_ranges: &[(&str, ParameterRange)] = &[
        ("temperature", ParameterRange::new().min(0.0).max(2.0)),
        ("top_p", ParameterRange::new().min(0.0).max(1.0)),
        ("top_k", ParameterRange::new().min(0.0)),
        ("frequency_penalty", ParameterRange::new().min(-2.0).max(2.0)),
        ("presence_penalty", ParameterRange::new().min(-2.0).max(2.0)),
    ];
    let range_map: HashMap<&str, &ParameterRange> =
        standard_ranges.iter().map(|(k, v)| (*k, v)).collect();

    params
        .iter()
        .map(|name| {
            let param_name: ParameterName = name.parse().unwrap();
            let availability = match range_map.get(name.as_str()) {
                Some(range) => ParameterAvailability::Mutable {
                    range: (*range).clone(),
                },
                None => ParameterAvailability::Opaque,
            };
            (param_name, availability)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> ModelEntry {
        ModelEntry {
            id: "anthropic/claude-sonnet-4".to_string(),
            context_length: Some(200_000),
            pricing: Some(PricingEntry {
                prompt: Some("0.000003".to_string()),
                completion: Some("0.000015".to_string()),
            }),
            top_provider: Some(TopProvider {
                max_completion_tokens: Some(8192),
            }),
            supported_parameters: vec![
                "temperature".to_string(),
                "max_tokens".to_string(),
                "top_p".to_string(),
                "reasoning".to_string(),
            ],
            architecture: Some(Architecture {
                modality: Some("text+image->text".to_string()),
            }),
        }
    }

    #[test]
    fn converts_basic_fields() {
        let metadata = into_model_metadata(sample_entry());

        assert_eq!(metadata.info.id, "anthropic/claude-sonnet-4");
        assert_eq!(metadata.info.provider, "openrouter");
        assert_eq!(metadata.info.context_window, Some(200_000));
        assert_eq!(metadata.max_output_tokens, Some(8192));
        assert!(metadata.info.capabilities.contains(&ModelCapability::Chat));
    }

    #[test]
    fn converts_pricing_per_token_to_per_mtok() {
        let metadata = into_model_metadata(sample_entry());
        let pricing = metadata.pricing.unwrap();

        assert!((pricing.prompt_cost_per_mtok.unwrap() - 3.0).abs() < 0.001);
        assert!((pricing.completion_cost_per_mtok.unwrap() - 15.0).abs() < 0.001);
    }

    #[test]
    fn converts_supported_parameters() {
        let metadata = into_model_metadata(sample_entry());

        // temperature should have a range
        let temp = &metadata.parameters[&ParameterName::Temperature];
        assert!(matches!(temp, ParameterAvailability::Mutable { range } if range.max == Some(2.0)));

        // reasoning should be opaque (no standard range)
        let reasoning = &metadata.parameters[&ParameterName::Reasoning];
        assert!(matches!(reasoning, ParameterAvailability::Opaque));

        assert_eq!(metadata.parameters.len(), 4);
    }

    #[test]
    fn handles_missing_pricing() {
        let entry = ModelEntry {
            id: "test/model".to_string(),
            context_length: None,
            pricing: None,
            top_provider: None,
            supported_parameters: vec![],
            architecture: None,
        };
        let metadata = into_model_metadata(entry);
        assert!(metadata.pricing.is_none());
        assert!(metadata.max_output_tokens.is_none());
        assert!(metadata.info.context_window.is_none());
    }

    #[test]
    fn parse_per_token_to_per_mtok_works() {
        assert!((parse_per_token_to_per_mtok(Some("0.000005")).unwrap() - 5.0).abs() < 0.001);
        assert!(parse_per_token_to_per_mtok(None).is_none());
        assert!(parse_per_token_to_per_mtok(Some("invalid")).is_none());
    }
}
