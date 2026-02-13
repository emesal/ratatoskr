//! Integration tests for [`RemoteRegistry`] — remote fetch, cache roundtrip,
//! format parsing, and `update_registry` flow.

use std::collections::{BTreeMap, HashMap};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use ratatoskr::registry::remote::{self, RegistryPayload, RemoteRegistry, RemoteRegistryConfig};
use ratatoskr::{CostTier, ModelInfo, ModelMetadata};

fn sample_metadata(id: &str) -> ModelMetadata {
    ModelMetadata {
        info: ModelInfo {
            id: id.to_string(),
            provider: "test".to_string(),
            capabilities: vec![],
            context_window: Some(4096),
            dimensions: None,
        },
        parameters: HashMap::new(),
        pricing: None,
        max_output_tokens: None,
    }
}

fn sample_payload(models: Vec<ModelMetadata>) -> RegistryPayload {
    RegistryPayload {
        models,
        presets: BTreeMap::new(),
    }
}

// =============================================================================
// Cache roundtrip (integration-level, uses tempdir)
// =============================================================================

#[test]
fn save_and_load_preserves_versioned_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("registry.json");

    let payload = sample_payload(vec![sample_metadata("gpt-4"), sample_metadata("claude-3")]);
    remote::save_cache(&path, &payload).unwrap();

    // Verify the on-disk format is versioned
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(raw["version"], 1);
    assert!(raw["models"].is_array());
    assert_eq!(raw["models"].as_array().unwrap().len(), 2);

    // Load back via the public API
    let loaded = remote::load_cached(&path).unwrap();
    assert_eq!(loaded.models.len(), 2);
    assert_eq!(loaded.models[0].info.id, "gpt-4");
    assert_eq!(loaded.models[1].info.id, "claude-3");
}

#[test]
fn load_cached_accepts_legacy_bare_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.json");

    // Write legacy bare-array format directly
    let models = vec![sample_metadata("legacy-model")];
    let json = serde_json::to_string_pretty(&models).unwrap();
    std::fs::write(&path, json).unwrap();

    let loaded = remote::load_cached(&path).unwrap();
    assert_eq!(loaded.models.len(), 1);
    assert_eq!(loaded.models[0].info.id, "legacy-model");
    assert!(loaded.presets.is_empty());
}

#[test]
fn load_cached_rejects_unsupported_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("future.json");

    let json = r#"{"version": 999, "models": []}"#;
    std::fs::write(&path, json).unwrap();

    // Unsupported version → treated as corrupt → returns None
    assert!(remote::load_cached(&path).is_none());
}

#[test]
fn load_cached_returns_none_for_missing_file() {
    assert!(remote::load_cached(std::path::Path::new("/tmp/nonexistent_registry.json")).is_none());
}

#[test]
fn load_cached_returns_none_for_corrupt_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.json");
    std::fs::write(&path, "not valid json {{{").unwrap();

    assert!(remote::load_cached(&path).is_none());
}

// =============================================================================
// Preset roundtrip through save/load
// =============================================================================

#[test]
fn presets_survive_save_load_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("registry.json");

    let mut presets = BTreeMap::new();
    let mut free_map = BTreeMap::new();
    free_map.insert("text-generation".to_owned(), "some/free-model".to_owned());
    free_map.insert("embedding".to_owned(), "embed/model".to_owned());
    presets.insert(CostTier::Free, free_map);

    let mut premium_map = BTreeMap::new();
    premium_map.insert("agentic".to_owned(), "big/model".to_owned());
    presets.insert(CostTier::Premium, premium_map);

    let payload = RegistryPayload {
        models: vec![sample_metadata("some/free-model")],
        presets,
    };
    remote::save_cache(&path, &payload).unwrap();

    let loaded = remote::load_cached(&path).unwrap();
    assert_eq!(loaded.presets.len(), 2);
    assert_eq!(
        loaded.presets[&CostTier::Free]["text-generation"],
        "some/free-model"
    );
    assert_eq!(loaded.presets[&CostTier::Free]["embedding"], "embed/model");
    assert_eq!(loaded.presets[&CostTier::Premium]["agentic"], "big/model");
}

#[test]
fn versioned_format_without_presets_parses_with_empty_presets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no-presets.json");

    // Write versioned format without presets key
    let json = r#"{"version": 1, "models": []}"#;
    std::fs::write(&path, json).unwrap();

    let loaded = remote::load_cached(&path).unwrap();
    assert!(loaded.models.is_empty());
    assert!(loaded.presets.is_empty());
}

// =============================================================================
// Remote fetch via wiremock
// =============================================================================

#[tokio::test]
async fn fetch_remote_versioned_format() {
    let server = MockServer::start().await;

    let registry = RemoteRegistry {
        version: 1,
        models: vec![sample_metadata("remote-model")],
        presets: BTreeMap::new(),
    };

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&registry))
        .mount(&server)
        .await;

    let payload = remote::fetch_remote(&format!("{}/registry.json", server.uri()))
        .await
        .unwrap();
    assert_eq!(payload.models.len(), 1);
    assert_eq!(payload.models[0].info.id, "remote-model");
}

#[tokio::test]
async fn fetch_remote_legacy_format() {
    let server = MockServer::start().await;

    let models = vec![sample_metadata("legacy-remote")];

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&models))
        .mount(&server)
        .await;

    let payload = remote::fetch_remote(&format!("{}/registry.json", server.uri()))
        .await
        .unwrap();
    assert_eq!(payload.models.len(), 1);
    assert_eq!(payload.models[0].info.id, "legacy-remote");
    assert!(payload.presets.is_empty());
}

#[tokio::test]
async fn fetch_remote_unsupported_version() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(r#"{"version": 999, "models": []}"#),
        )
        .mount(&server)
        .await;

    let result = remote::fetch_remote(&format!("{}/registry.json", server.uri())).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unsupported registry version")
    );
}

#[tokio::test]
async fn fetch_remote_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let result = remote::fetch_remote(&format!("{}/registry.json", server.uri())).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("HTTP 500"));
}

#[tokio::test]
async fn fetch_remote_invalid_json() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let result = remote::fetch_remote(&format!("{}/registry.json", server.uri())).await;
    assert!(result.is_err());
}

// =============================================================================
// Full update_registry flow (fetch + save)
// =============================================================================

#[tokio::test]
async fn update_registry_fetches_and_caches() {
    let server = MockServer::start().await;
    let dir = tempfile::tempdir().unwrap();
    let cache_path = dir.path().join("cached_registry.json");

    let registry = RemoteRegistry {
        version: 1,
        models: vec![sample_metadata("model-x"), sample_metadata("model-y")],
        presets: BTreeMap::new(),
    };

    Mock::given(method("GET"))
        .and(path("/registry.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&registry))
        .mount(&server)
        .await;

    let config = RemoteRegistryConfig {
        url: format!("{}/registry.json", server.uri()),
        cache_path: cache_path.clone(),
    };

    let payload = remote::update_registry(&config).await.unwrap();
    assert_eq!(payload.models.len(), 2);

    // Verify it was also saved to disk
    assert!(cache_path.exists());
    let cached = remote::load_cached(&cache_path).unwrap();
    assert_eq!(cached.models.len(), 2);
    assert_eq!(cached.models[0].info.id, "model-x");
}
