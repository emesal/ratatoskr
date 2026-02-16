//! Integration tests for provider fallback behaviour.
//!
//! These tests verify the fallback chain works correctly across provider types:
//! - Local provider → API fallback on `ModelNotAvailable`
//! - RAM budget constraints trigger fallback
//! - Stance classification via `ZeroShotStanceProvider`

#![cfg(all(feature = "huggingface", feature = "local-inference"))]

use std::sync::Arc;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ratatoskr::model::{ModelManager, ModelManagerConfig};
use ratatoskr::providers::registry::ProviderRegistry;
use ratatoskr::providers::traits::{ClassifyProvider, StanceProvider, ZeroShotStanceProvider};
use ratatoskr::providers::{HuggingFaceClient, LocalEmbeddingModel, LocalEmbeddingProvider};
use ratatoskr::{Device, RatatoskrError, StanceLabel};

// ============================================================================
// Local → API Fallback Tests
// ============================================================================

/// Test that when local provider returns `ModelNotAvailable` (wrong model),
/// the registry falls back to the HuggingFace API provider.
#[tokio::test]
async fn fallback_local_to_api_wrong_model() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-mpnet-base-v2"; // Model local provider doesn't have

    // Setup mock for HuggingFace API
    let embedding_response = serde_json::json!([[0.1, 0.2, 0.3, 0.4, 0.5]]);

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response))
        .expect(1) // Expect exactly one call to API
        .mount(&mock_server)
        .await;

    // Create registry with local provider first (handles AllMiniLmL6V2 only), then HuggingFace
    let mut registry = ProviderRegistry::new();

    // Local provider only handles "all-MiniLM-L6-v2"
    let manager = Arc::new(ModelManager::with_defaults());
    let local_provider = Arc::new(LocalEmbeddingProvider::new(
        LocalEmbeddingModel::AllMiniLmL6V2,
        manager,
    ));
    registry.add_embedding(local_provider);

    // HuggingFace as fallback
    let hf_client = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    registry.add_embedding(hf_client);

    // Request a model the local provider doesn't support
    let result = registry.embed("hello world", model, None).await;

    // Should succeed via HuggingFace fallback
    let embedding = result.expect("should fallback to HuggingFace");
    assert_eq!(embedding.dimensions, 5);
    assert_eq!(embedding.model, model);
}

// ============================================================================
// RAM Budget Fallback Tests
// ============================================================================

/// Test that when RAM budget would be exceeded, local provider returns
/// `ModelNotAvailable` and registry falls back to API.
#[tokio::test]
async fn fallback_ram_budget_exceeded() {
    let mock_server = MockServer::start().await;
    let model = "all-MiniLM-L6-v2"; // Model local provider handles

    // Setup mock for HuggingFace API
    let embedding_response = serde_json::json!([[0.5, 0.5, 0.5]]);

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create registry with RAM-constrained local provider
    let mut registry = ProviderRegistry::new();

    // ModelManager with tiny RAM budget (1 byte - impossible to load any model)
    let config = ModelManagerConfig {
        ram_budget: Some(1),
        default_device: Device::Cpu,
        ..Default::default()
    };
    let manager = Arc::new(ModelManager::new(config));
    let local_provider = Arc::new(LocalEmbeddingProvider::new(
        LocalEmbeddingModel::AllMiniLmL6V2,
        manager,
    ));
    registry.add_embedding(local_provider);

    // HuggingFace as fallback
    let hf_client = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    registry.add_embedding(hf_client);

    // Request the model local provider normally handles
    let result = registry.embed("hello world", model, None).await;

    // Should succeed via HuggingFace fallback due to RAM budget
    let embedding = result.expect("should fallback to HuggingFace due to RAM budget");
    assert_eq!(embedding.dimensions, 3);
}

// ============================================================================
// Stance Classification Tests
// ============================================================================

/// Test that `ZeroShotStanceProvider` correctly wraps ClassifyProvider
/// and returns proper stance results.
#[tokio::test]
async fn stance_via_zero_shot_provider() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    // Mock response for zero-shot classification with favor/against/neutral
    let classify_response = serde_json::json!({
        "labels": ["favor", "neutral", "against"],
        "scores": [0.75, 0.15, 0.10]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(classify_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Create ZeroShotStanceProvider wrapping HuggingFace
    let hf_client: Arc<dyn ClassifyProvider> = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    let stance_provider = ZeroShotStanceProvider::new(hf_client, model);

    // Test stance classification
    let result = stance_provider
        .classify_stance(
            "I strongly support renewable energy",
            "renewable energy",
            model,
        )
        .await;

    let stance = result.expect("stance classification should succeed");
    assert_eq!(stance.label, StanceLabel::Favor);
    assert_eq!(stance.target, "renewable energy");
    assert!((stance.favor - 0.75).abs() < 0.001);
    assert!((stance.neutral - 0.15).abs() < 0.001);
    assert!((stance.against - 0.10).abs() < 0.001);
}

/// Test stance classification with "against" result.
#[tokio::test]
async fn stance_against_via_zero_shot() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    let classify_response = serde_json::json!({
        "labels": ["against", "neutral", "favor"],
        "scores": [0.80, 0.12, 0.08]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(classify_response))
        .mount(&mock_server)
        .await;

    let hf_client: Arc<dyn ClassifyProvider> = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    let stance_provider = ZeroShotStanceProvider::new(hf_client, model);

    let result = stance_provider
        .classify_stance("This policy is terrible and harmful", "the policy", model)
        .await;

    let stance = result.expect("stance classification should succeed");
    assert_eq!(stance.label, StanceLabel::Against);
    assert!((stance.against - 0.80).abs() < 0.001);
}

/// Test stance classification with "neutral" result.
#[tokio::test]
async fn stance_neutral_via_zero_shot() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    let classify_response = serde_json::json!({
        "labels": ["neutral", "favor", "against"],
        "scores": [0.60, 0.25, 0.15]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(classify_response))
        .mount(&mock_server)
        .await;

    let hf_client: Arc<dyn ClassifyProvider> = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    let stance_provider = ZeroShotStanceProvider::new(hf_client, model);

    let result = stance_provider
        .classify_stance(
            "The committee discussed several options",
            "the proposal",
            model,
        )
        .await;

    let stance = result.expect("stance classification should succeed");
    assert_eq!(stance.label, StanceLabel::Neutral);
}

// ============================================================================
// Stance in Registry Tests
// ============================================================================

/// Test stance classification through the registry fallback chain.
#[tokio::test]
async fn stance_through_registry() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    let classify_response = serde_json::json!({
        "labels": ["favor", "against", "neutral"],
        "scores": [0.70, 0.20, 0.10]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(classify_response))
        .mount(&mock_server)
        .await;

    let mut registry = ProviderRegistry::new();

    // Add ZeroShotStanceProvider to registry
    let hf_client: Arc<dyn ClassifyProvider> = Arc::new(HuggingFaceClient::with_base_url(
        "test_key",
        mock_server.uri(),
    ));
    let stance_provider = Arc::new(ZeroShotStanceProvider::new(hf_client, model));
    registry.add_stance(stance_provider);

    // Use registry's stance method
    let result = registry
        .classify_stance("I'm in favor of this initiative", "the initiative", model)
        .await;

    let stance = result.expect("stance via registry should succeed");
    assert_eq!(stance.label, StanceLabel::Favor);
}

/// Test that registry returns NoProvider when no stance provider is registered.
#[tokio::test]
async fn stance_no_provider_error() {
    let registry = ProviderRegistry::new();

    let result = registry
        .classify_stance("test text", "test target", "any-model")
        .await;

    assert!(
        matches!(result, Err(RatatoskrError::NoProvider)),
        "expected NoProvider, got {:?}",
        result
    );
}
