use ratatoskr::{Capabilities, ModelGateway, Ratatoskr};

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
fn test_capabilities_chat_only() {
    let caps = Capabilities::chat_only();
    assert!(caps.chat);
    assert!(caps.chat_streaming);
    assert!(!caps.embeddings);
}

#[test]
#[cfg(feature = "huggingface")]
fn test_builder_with_huggingface() {
    let gateway = Ratatoskr::builder()
        .huggingface("hf_test_key")
        .build()
        .expect("should build with huggingface only");

    let caps = gateway.capabilities();
    assert!(caps.embeddings);
    assert!(caps.nli);
    assert!(caps.classification);
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
    assert!(caps.embeddings);
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
