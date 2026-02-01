use ratatoskr::{Capabilities, Ratatoskr};

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
