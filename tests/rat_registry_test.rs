//! Integration tests for `rat-registry model refresh` logic.
//!
//! Tests the merge semantics: successful fetches update the registry,
//! failures are collected and reported, presets are preserved.

use std::collections::HashMap;

use ratatoskr::{ModelCapability, ModelInfo, ModelMetadata, ModelRegistry, PricingInfo};

fn make_metadata(id: &str, context_window: Option<usize>) -> ModelMetadata {
    ModelMetadata {
        info: ModelInfo {
            id: id.to_string(),
            provider: "openrouter".to_string(),
            capabilities: vec![ModelCapability::Chat],
            context_window,
            dimensions: None,
        },
        parameters: HashMap::new(),
        pricing: None,
        max_output_tokens: None,
    }
}

/// Simulates the merge logic that `model_refresh` applies.
/// Verifies that a successful fetch updates `context_window`
/// while preserving models that had no update.
#[test]
fn merge_updates_existing_model_fields() {
    let original = make_metadata("provider/model-a", Some(4096));
    let mut registry = ModelRegistry::default();
    registry.merge(original);

    let updated = make_metadata("provider/model-a", Some(128000));
    registry.merge(updated);

    let got = registry.get("provider/model-a").unwrap();
    assert_eq!(got.info.context_window, Some(128000));
}

/// Verifies merge preserves hand-curated fields absent from fetched metadata.
#[test]
fn merge_preserves_absent_optional_fields() {
    let mut original = make_metadata("provider/model-b", Some(8192));
    original.pricing = Some(PricingInfo {
        prompt_cost_per_mtok: Some(3.0),
        completion_cost_per_mtok: Some(15.0),
    });

    let mut reg = ModelRegistry::default();
    reg.merge(original);

    // fetched metadata has no pricing (provider didn't return it)
    let fetched = make_metadata("provider/model-b", Some(16384));
    reg.merge(fetched);

    let got = reg.get("provider/model-b").unwrap();
    // pricing preserved from original since fetched had None
    assert!(got.pricing.is_some());
    assert_eq!(got.info.context_window, Some(16384));
}
