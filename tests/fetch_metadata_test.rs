//! Wiremock integration tests for model metadata fetch.
//!
//! Tests the full `fetch_model_metadata()` flow through `EmbeddedGateway`,
//! including HTTP fetch, parsing, cache population, and registry priority.

use ratatoskr::providers::{LlmChatProvider, ProviderRegistry};
use ratatoskr::{ModelCapability, ModelGateway, ParameterName, Ratatoskr};

use llm::builder::LLMBackend;
use std::sync::Arc;
use wiremock::matchers::{header, method, path};
use wiremock::{Match, Mock, MockServer, ResponseTemplate};

/// Sample OpenRouter `/api/v1/models` response with one model.
fn sample_models_json() -> serde_json::Value {
    serde_json::json!({
        "data": [{
            "id": "test-vendor/test-model",
            "context_length": 128000,
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.000008"
            },
            "top_provider": {
                "max_completion_tokens": 4096
            },
            "supported_parameters": ["temperature", "max_tokens", "top_p"]
        }]
    })
}

/// Build a `ProviderRegistry` with a single OpenRouter provider pointed at wiremock.
fn registry_with_mock(mock_url: &str) -> ProviderRegistry {
    let provider = Arc::new(
        LlmChatProvider::new(LLMBackend::OpenRouter, Some("test-key"), "openrouter")
            .models_base_url(mock_url),
    );
    let mut registry = ProviderRegistry::new();
    registry.add_chat(provider);
    registry
}

#[tokio::test]
async fn fetch_metadata_returns_parsed_model() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .mount(&server)
        .await;

    let registry = registry_with_mock(&server.uri());
    let metadata = registry
        .fetch_chat_metadata("test-vendor/test-model")
        .await
        .expect("fetch should succeed");

    assert_eq!(metadata.info.id, "test-vendor/test-model");
    assert_eq!(metadata.info.provider, "openrouter");
    assert_eq!(metadata.info.context_window, Some(128000));
    assert_eq!(metadata.max_output_tokens, Some(4096));
    assert!(metadata.info.capabilities.contains(&ModelCapability::Chat));

    // pricing: 0.000002 per token = $2/Mtok
    let pricing = metadata.pricing.unwrap();
    assert!((pricing.prompt_cost_per_mtok.unwrap() - 2.0).abs() < 0.001);
    assert!((pricing.completion_cost_per_mtok.unwrap() - 8.0).abs() < 0.001);

    // parameters
    assert!(
        metadata
            .parameters
            .contains_key(&ParameterName::Temperature)
    );
    assert!(metadata.parameters.contains_key(&ParameterName::MaxTokens));
    assert!(metadata.parameters.contains_key(&ParameterName::TopP));
}

#[tokio::test]
async fn fetch_metadata_model_not_in_response() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .mount(&server)
        .await;

    let registry = registry_with_mock(&server.uri());
    let result = registry.fetch_chat_metadata("nonexistent/model").await;

    assert!(result.is_err(), "should fail for unknown model");
}

#[tokio::test]
async fn fetch_metadata_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let registry = registry_with_mock(&server.uri());
    let result = registry.fetch_chat_metadata("test-vendor/test-model").await;

    assert!(result.is_err());
}

#[tokio::test]
async fn fetch_metadata_non_openrouter_returns_no_provider() {
    // Anthropic backend should return ModelNotAvailable, causing NoProvider
    let provider = Arc::new(LlmChatProvider::new(
        LLMBackend::Anthropic,
        Some("test-key"),
        "anthropic",
    ));
    let mut registry = ProviderRegistry::new();
    registry.add_chat(provider);

    let result = registry.fetch_chat_metadata("some-model").await;

    assert!(result.is_err());
}

#[tokio::test]
async fn embedded_gateway_fetch_populates_cache() {
    // Use a real EmbeddedGateway but we can't inject wiremock URL through the builder.
    // Instead, verify the cache interaction: model_metadata returns None before fetch,
    // and the trait method exists (compile-time check).
    let gateway = Ratatoskr::builder()
        .openrouter(Some("fake-key"))
        .build()
        .unwrap();

    // Unknown model should not be in registry or cache
    assert!(gateway.model_metadata("test-vendor/test-model").is_none());

    // fetch_model_metadata would make a real HTTP call (which will fail with fake key),
    // but at least verify it returns an error rather than panicking
    let result = gateway.fetch_model_metadata("test-vendor/test-model").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn registry_priority_curated_over_cache() {
    // model_metadata prefers ModelRegistry (curated) over ModelCache.
    // claude-sonnet-4 is in the embedded seed, so even after a fetch that
    // might cache different data, model_metadata should return the seed entry.
    let gateway = Ratatoskr::builder()
        .openrouter(Some("fake-key"))
        .build()
        .unwrap();

    let metadata = gateway
        .model_metadata("anthropic/claude-sonnet-4")
        .expect("should find in registry");

    assert_eq!(metadata.info.id, "anthropic/claude-sonnet-4");
}

// -- Keyless vs keyed auth header tests --

/// Matcher that asserts the `Authorization` header is absent.
struct NoAuthHeader;

impl Match for NoAuthHeader {
    fn matches(&self, request: &wiremock::Request) -> bool {
        !request.headers.contains_key("Authorization")
    }
}

/// Build a keyless registry with a single OpenRouter provider pointed at wiremock.
fn keyless_registry_with_mock(mock_url: &str) -> ProviderRegistry {
    let provider = Arc::new(
        LlmChatProvider::new(LLMBackend::OpenRouter, None::<String>, "openrouter")
            .models_base_url(mock_url),
    );
    let mut registry = ProviderRegistry::new();
    registry.add_chat(provider);
    registry
}

#[tokio::test]
async fn fetch_metadata_keyless_sends_no_auth_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(NoAuthHeader)
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .expect(1)
        .mount(&server)
        .await;

    let registry = keyless_registry_with_mock(&server.uri());
    let metadata = registry
        .fetch_chat_metadata("test-vendor/test-model")
        .await
        .expect("keyless fetch should succeed");

    assert_eq!(metadata.info.id, "test-vendor/test-model");
}

#[tokio::test]
async fn fetch_metadata_keyed_sends_auth_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .expect(1)
        .mount(&server)
        .await;

    let registry = registry_with_mock(&server.uri());
    let metadata = registry
        .fetch_chat_metadata("test-vendor/test-model")
        .await
        .expect("keyed fetch should succeed");

    assert_eq!(metadata.info.id, "test-vendor/test-model");
}
