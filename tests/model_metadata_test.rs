use std::collections::HashMap;

use ratatoskr::{
    ModelCapability, ModelInfo, ModelMetadata, ParameterAvailability, ParameterName,
    ParameterRange, PricingInfo,
};

#[test]
fn model_metadata_from_info() {
    let info = ModelInfo::new("claude-sonnet-4", "openrouter")
        .with_capability(ModelCapability::Chat)
        .with_context_window(200_000);

    let metadata = ModelMetadata::from_info(info);
    assert_eq!(metadata.info.id, "claude-sonnet-4");
    assert!(metadata.parameters.is_empty());
    assert!(metadata.pricing.is_none());
    assert!(metadata.max_output_tokens.is_none());
}

#[test]
fn model_metadata_builder() {
    let metadata = ModelMetadata::from_info(ModelInfo::new("gpt-4o", "openrouter"))
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(2.0).default_value(1.0),
            },
        )
        .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported)
        .with_max_output_tokens(16384)
        .with_pricing(PricingInfo {
            prompt_cost_per_mtok: Some(2.50),
            completion_cost_per_mtok: Some(10.0),
        });

    assert_eq!(metadata.parameters.len(), 2);
    assert!(metadata.parameters[&ParameterName::Temperature].is_supported());
    assert!(!metadata.parameters[&ParameterName::TopK].is_supported());
    assert_eq!(metadata.max_output_tokens, Some(16384));
    assert_eq!(
        metadata.pricing.as_ref().unwrap().prompt_cost_per_mtok,
        Some(2.50)
    );
}

#[test]
fn model_metadata_serde_roundtrip() {
    let mut params = HashMap::new();
    params.insert(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(2.0),
        },
    );
    params.insert(
        ParameterName::Custom("logit_bias".into()),
        ParameterAvailability::Opaque,
    );

    let metadata = ModelMetadata {
        info: ModelInfo::new("test", "test").with_capability(ModelCapability::Chat),
        parameters: params,
        pricing: Some(PricingInfo {
            prompt_cost_per_mtok: Some(1.0),
            completion_cost_per_mtok: Some(2.0),
        }),
        max_output_tokens: Some(4096),
    };

    let json = serde_json::to_string(&metadata).unwrap();
    let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.info.id, "test");
    assert_eq!(parsed.parameters.len(), 2);
    assert!(parsed.parameters.contains_key(&ParameterName::Temperature));
    assert!(
        parsed
            .parameters
            .contains_key(&ParameterName::Custom("logit_bias".into()))
    );
    assert_eq!(parsed.max_output_tokens, Some(4096));
}

#[test]
fn pricing_info_default() {
    let pricing = PricingInfo::default();
    assert!(pricing.prompt_cost_per_mtok.is_none());
    assert!(pricing.completion_cost_per_mtok.is_none());
}

#[test]
fn model_metadata_merge_parameters() {
    let base = ModelMetadata::from_info(ModelInfo::new("test", "test"))
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(1.0),
            },
        )
        .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported);

    // override merges on top — more specific data wins
    let override_params = {
        let mut m = HashMap::new();
        m.insert(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(2.0).default_value(1.0),
            },
        );
        m.insert(ParameterName::Seed, ParameterAvailability::Opaque);
        m
    };

    let merged = base.merge_parameters(override_params);
    // temperature updated to new range
    assert!(matches!(
        merged.parameters[&ParameterName::Temperature],
        ParameterAvailability::Mutable { ref range } if range.max == Some(2.0)
    ));
    // top_k preserved from base
    assert!(matches!(
        merged.parameters[&ParameterName::TopK],
        ParameterAvailability::Unsupported
    ));
    // seed added from override
    assert!(matches!(
        merged.parameters[&ParameterName::Seed],
        ParameterAvailability::Opaque
    ));
}
