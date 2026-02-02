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
