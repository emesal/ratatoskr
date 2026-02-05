use ratatoskr::{
    ModelCapability, ModelInfo, ModelMetadata, ModelRegistry, ParameterAvailability, ParameterName,
    ParameterRange,
};

#[test]
fn empty_registry() {
    let registry = ModelRegistry::new();
    assert!(registry.get("nonexistent").is_none());
    assert!(registry.list().is_empty());
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn insert_and_get() {
    let mut registry = ModelRegistry::new();
    let metadata = ModelMetadata::from_info(
        ModelInfo::new("claude-sonnet-4", "openrouter")
            .with_capability(ModelCapability::Chat)
            .with_context_window(200_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(1.0),
        },
    );

    registry.insert(metadata);

    let found = registry.get("claude-sonnet-4").unwrap();
    assert_eq!(found.info.id, "claude-sonnet-4");
    assert_eq!(found.info.context_window, Some(200_000));
    assert!(!registry.is_empty());
    assert_eq!(registry.len(), 1);
}

#[test]
fn list_returns_all() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(ModelInfo::new("a", "p")));
    registry.insert(ModelMetadata::from_info(ModelInfo::new("b", "p")));
    registry.insert(ModelMetadata::from_info(ModelInfo::new("c", "p")));

    let list = registry.list();
    assert_eq!(list.len(), 3);
}

#[test]
fn merge_updates_existing() {
    let mut registry = ModelRegistry::new();

    // Base entry (from embedded seed).
    let base = ModelMetadata::from_info(
        ModelInfo::new("gpt-4o", "openrouter").with_context_window(128_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(1.0),
        },
    )
    .with_max_output_tokens(4096);
    registry.insert(base);

    // Live data (overrides).
    let live = ModelMetadata::from_info(
        ModelInfo::new("gpt-4o", "openrouter").with_context_window(128_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(2.0).default_value(1.0),
        },
    )
    .with_max_output_tokens(16384);

    registry.merge(live);

    let result = registry.get("gpt-4o").unwrap();
    // max_output_tokens updated
    assert_eq!(result.max_output_tokens, Some(16384));
    // temperature range updated
    assert!(matches!(
        &result.parameters[&ParameterName::Temperature],
        ParameterAvailability::Mutable { range } if range.max == Some(2.0)
    ));
}

#[test]
fn merge_preserves_non_overridden_params() {
    let mut registry = ModelRegistry::new();

    let base = ModelMetadata::from_info(ModelInfo::new("m", "p"))
        .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported)
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::default(),
            },
        );
    registry.insert(base);

    // Live data only knows about temperature.
    let live = ModelMetadata::from_info(ModelInfo::new("m", "p")).with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().max(2.0),
        },
    );
    registry.merge(live);

    let result = registry.get("m").unwrap();
    // top_k preserved from base
    assert!(result.parameters.contains_key(&ParameterName::TopK));
    // temperature updated
    assert_eq!(result.parameters.len(), 2);
}

#[test]
fn merge_batch() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(ModelInfo::new("a", "p")));

    let batch = vec![
        ModelMetadata::from_info(ModelInfo::new("a", "p")).with_max_output_tokens(100),
        ModelMetadata::from_info(ModelInfo::new("b", "p")).with_max_output_tokens(200),
    ];

    registry.merge_batch(batch);

    assert_eq!(registry.get("a").unwrap().max_output_tokens, Some(100));
    assert_eq!(registry.get("b").unwrap().max_output_tokens, Some(200));
}

#[test]
fn merge_adds_new_entry_when_absent() {
    let mut registry = ModelRegistry::new();
    assert!(registry.is_empty());

    let entry =
        ModelMetadata::from_info(ModelInfo::new("new-model", "p")).with_max_output_tokens(8192);
    registry.merge(entry);

    assert_eq!(registry.len(), 1);
    let found = registry.get("new-model").unwrap();
    assert_eq!(found.max_output_tokens, Some(8192));
}

#[test]
fn merge_unions_capabilities() {
    let mut registry = ModelRegistry::new();

    let base = ModelMetadata::from_info(
        ModelInfo::new("multi", "p").with_capability(ModelCapability::Chat),
    );
    registry.insert(base);

    let incoming = ModelMetadata::from_info(
        ModelInfo::new("multi", "p").with_capability(ModelCapability::Embed),
    );
    registry.merge(incoming);

    let result = registry.get("multi").unwrap();
    assert!(result.info.capabilities.contains(&ModelCapability::Chat));
    assert!(result.info.capabilities.contains(&ModelCapability::Embed));
}

#[test]
fn filter_by_capability() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("chat-model", "p").with_capability(ModelCapability::Chat),
    ));
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("embed-model", "p").with_capability(ModelCapability::Embed),
    ));
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("multi-model", "p")
            .with_capability(ModelCapability::Chat)
            .with_capability(ModelCapability::Embed),
    ));

    let chat_models = registry.filter_by_capability(ModelCapability::Chat);
    assert_eq!(chat_models.len(), 2);
    assert!(chat_models.iter().any(|m| m.info.id == "chat-model"));
    assert!(chat_models.iter().any(|m| m.info.id == "multi-model"));
}

#[test]
fn insert_replaces_existing() {
    let mut registry = ModelRegistry::new();

    registry.insert(ModelMetadata::from_info(ModelInfo::new("x", "p")).with_max_output_tokens(100));
    registry.insert(ModelMetadata::from_info(ModelInfo::new("x", "p")).with_max_output_tokens(200));

    assert_eq!(registry.len(), 1);
    assert_eq!(registry.get("x").unwrap().max_output_tokens, Some(200));
}

// ============================================================================
// Embedded seed tests
// ============================================================================

#[test]
fn load_embedded_seed() {
    let registry = ModelRegistry::with_embedded_seed();
    // Seed should contain at least a few well-known models.
    assert!(!registry.is_empty());

    // Check a known model exists.
    let claude = registry.get("anthropic/claude-sonnet-4");
    assert!(
        claude.is_some(),
        "embedded seed should contain claude-sonnet-4"
    );
    let claude = claude.unwrap();
    assert!(claude.info.capabilities.contains(&ModelCapability::Chat));
    assert!(claude.info.context_window.is_some());
}

#[test]
fn embedded_seed_has_parameter_metadata() {
    let registry = ModelRegistry::with_embedded_seed();
    let claude = registry.get("anthropic/claude-sonnet-4").unwrap();

    // Should declare at least temperature.
    assert!(claude.parameters.contains_key(&ParameterName::Temperature));
    assert!(claude.parameters[&ParameterName::Temperature].is_supported());
}

#[test]
fn embedded_seed_has_pricing() {
    let registry = ModelRegistry::with_embedded_seed();
    let claude = registry.get("anthropic/claude-sonnet-4").unwrap();

    let pricing = claude
        .pricing
        .as_ref()
        .expect("seed should include pricing");
    assert!(pricing.prompt_cost_per_mtok.is_some());
    assert!(pricing.completion_cost_per_mtok.is_some());
}

#[test]
fn embedded_seed_has_embedding_model() {
    let registry = ModelRegistry::with_embedded_seed();
    let embed = registry.get("sentence-transformers/all-MiniLM-L6-v2");
    assert!(embed.is_some(), "seed should contain an embedding model");

    let embed = embed.unwrap();
    assert!(embed.info.capabilities.contains(&ModelCapability::Embed));
    assert_eq!(embed.info.dimensions, Some(384));
}

#[test]
fn embedded_seed_multiple_providers() {
    let registry = ModelRegistry::with_embedded_seed();

    // Should have models from multiple providers.
    let providers: std::collections::HashSet<&str> = registry
        .list()
        .iter()
        .map(|m| m.info.provider.as_str())
        .collect();
    assert!(
        providers.len() >= 2,
        "seed should contain models from at least 2 providers"
    );
}

#[test]
fn embedded_seed_merge_over_seed() {
    let mut registry = ModelRegistry::with_embedded_seed();
    let original_count = registry.len();

    // Merge live data on top of seed.
    let live = ModelMetadata::from_info(
        ModelInfo::new("anthropic/claude-sonnet-4", "openrouter")
            .with_capability(ModelCapability::Chat),
    )
    .with_max_output_tokens(32768);

    registry.merge(live);

    // Count shouldn't change (merge, not insert).
    assert_eq!(registry.len(), original_count);
    // max_output_tokens should be updated.
    let claude = registry.get("anthropic/claude-sonnet-4").unwrap();
    assert_eq!(claude.max_output_tokens, Some(32768));
    // Original parameters should be preserved.
    assert!(claude.parameters.contains_key(&ParameterName::Temperature));
}
