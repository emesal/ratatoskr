use ratatoskr::{ModelGateway, ParameterName, Ratatoskr};

#[test]
fn test_builder_no_provider_error() {
    let result = Ratatoskr::builder().build();
    assert!(result.is_err());
}

#[test]
fn test_builder_with_openrouter() {
    // Test that builder accepts the key and can build (no network call)
    let gateway = Ratatoskr::builder().openrouter("sk-or-test-key").build();

    // Should succeed since we have a provider configured
    assert!(gateway.is_ok());
}

#[test]
#[cfg(feature = "huggingface")]
fn test_builder_with_huggingface() {
    let gateway = Ratatoskr::builder()
        .huggingface("hf_test_key")
        .build()
        .expect("should build with huggingface only");

    let caps = gateway.capabilities();
    assert!(caps.embed);
    assert!(caps.nli);
    assert!(caps.classify);
}

#[test]
#[cfg(feature = "huggingface")]
fn test_builder_openrouter_and_huggingface() {
    let gateway = Ratatoskr::builder()
        .openrouter("sk-or-test")
        .huggingface("hf_test")
        .build()
        .expect("should build with both providers");

    let caps = gateway.capabilities();
    assert!(caps.chat);
    assert!(caps.embed);
}

// ===== Phase 6: model_metadata tests =====

#[test]
fn test_embedded_gateway_model_metadata() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    // the embedded seed should provide metadata
    let metadata = gateway.model_metadata("anthropic/claude-sonnet-4");
    assert!(
        metadata.is_some(),
        "should find claude-sonnet-4 in registry"
    );

    let metadata = metadata.unwrap();
    assert!(
        metadata
            .parameters
            .contains_key(&ParameterName::Temperature)
    );
    assert!(metadata.info.context_window.is_some());
}

#[test]
fn test_embedded_gateway_model_metadata_unknown() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    assert!(gateway.model_metadata("totally-fake-model-xyz").is_none());
}

// Local inference builder tests
#[test]
#[cfg(feature = "local-inference")]
fn test_builder_methods_compile() {
    use ratatoskr::tokenizer::TokenizerSource;
    use ratatoskr::{Device, LocalEmbeddingModel, LocalNliModel};
    use std::path::PathBuf;

    // This test just verifies the builder methods compile and chain correctly.
    // We don't actually build because that requires chat OR local providers,
    // and we don't want to load models in unit tests.
    let _builder = Ratatoskr::builder()
        .openrouter("test-key")
        .local_embeddings(LocalEmbeddingModel::AllMiniLmL6V2)
        .local_nli(LocalNliModel::NliDebertaV3Small)
        .device(Device::Cpu)
        .cache_dir("/tmp/cache")
        .tokenizer_mapping(
            "custom-model",
            TokenizerSource::Local {
                path: PathBuf::from("/path/to/tokenizer.json"),
            },
        )
        .timeout(60);
}

// ===== Preset URI resolution =====

#[test]
fn test_preset_uri_resolves_in_model_metadata() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    // preset URI should resolve to the same metadata as the concrete model
    let via_preset = gateway.model_metadata("ratatoskr:free/text-generation");
    let via_direct = gateway.model_metadata("google/gemini-2.0-flash-001");

    assert!(via_preset.is_some(), "preset should resolve to metadata");
    assert_eq!(
        via_preset.unwrap().info.id,
        via_direct.unwrap().info.id,
        "preset and direct should yield same model"
    );
}

#[test]
fn test_preset_uri_resolve_preset_still_works() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    // the resolve_preset trait method should still work independently
    let model = gateway.resolve_preset("free", "agentic");
    assert!(model.is_some());
    assert_eq!(model.unwrap(), "google/gemini-2.0-flash-001");
}

#[test]
fn test_preset_uri_unknown_tier_returns_error() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();
    let result = gateway.model_metadata("ratatoskr:nonexistent/agentic");
    assert!(result.is_none(), "unknown preset tier should return None from model_metadata");
}
