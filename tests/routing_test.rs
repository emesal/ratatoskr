// tests/routing_test.rs
use ratatoskr::gateway::routing::{CapabilityRouter, ClassifyProvider, EmbedProvider, NliProvider};

#[test]
fn test_router_default_no_providers() {
    let router = CapabilityRouter::default();
    assert!(router.embed_provider().is_none());
    assert!(router.nli_provider().is_none());
    assert!(router.classify_provider().is_none());
}

#[test]
fn test_router_with_huggingface() {
    let router = CapabilityRouter::default().with_huggingface();

    assert!(matches!(
        router.embed_provider(),
        Some(EmbedProvider::HuggingFace)
    ));
    assert!(matches!(
        router.nli_provider(),
        Some(NliProvider::HuggingFace)
    ));
    assert!(matches!(
        router.classify_provider(),
        Some(ClassifyProvider::HuggingFace)
    ));
}
