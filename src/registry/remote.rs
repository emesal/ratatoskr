//! Remote model registry — fetch, cache, and load curated model metadata.
//!
//! The remote registry uses the same JSON format as `seed.json`. It's fetched
//! from a configurable URL (default: `emesal/ratatoskr-registry` on GitHub)
//! and cached locally at `~/.cache/ratatoskr/registry.json`.
//!
//! # Startup behaviour
//!
//! Startup loads from local cache only (fast, no network). Use
//! `rat update-registry` or [`update_registry()`] to trigger a network fetch.
//!
//! # Merge priority
//!
//! Embedded seed → cached remote → live provider data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::types::ModelMetadata;
use crate::{RatatoskrError, Result};

/// Default URL for the curated remote registry.
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/emesal/ratatoskr-registry/main/registry.json";

/// Maximum supported registry format version.
const MAX_SUPPORTED_VERSION: u32 = 1;

/// Configuration for the remote registry.
///
/// ```rust
/// # use ratatoskr::RemoteRegistryConfig;
/// let config = RemoteRegistryConfig::default();
/// assert!(config.url.contains("ratatoskr-registry"));
/// ```
#[derive(Debug, Clone)]
pub struct RemoteRegistryConfig {
    /// URL to fetch the registry from.
    pub url: String,
    /// Local path to cache the registry JSON.
    pub cache_path: PathBuf,
}

impl Default for RemoteRegistryConfig {
    fn default() -> Self {
        Self {
            url: DEFAULT_REGISTRY_URL.to_string(),
            cache_path: default_cache_path(),
        }
    }
}

impl RemoteRegistryConfig {
    /// Create a config with a custom URL and default cache path.
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }
}

/// Default cache path: `~/.cache/ratatoskr/registry.json`.
fn default_cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ratatoskr")
        .join("registry.json")
}

/// Versioned payload wrapper for the remote registry format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRegistry {
    /// Format version (currently 1).
    pub version: u32,
    /// Model metadata entries.
    pub models: Vec<ModelMetadata>,
    /// Autoconfig presets: `cost_tier → { capability → model_id }`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub presets: BTreeMap<String, BTreeMap<String, String>>,
}

/// Accept both versioned and bare-array formats.
///
/// The versioned format (`{ "version": 1, "models": [...] }`) is preferred.
/// The bare-array format (`[...]`) is accepted for backwards compatibility
/// with `seed.json`-style files.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawPayload {
    Versioned(RemoteRegistry),
    Legacy(Vec<ModelMetadata>),
}

/// Parsed registry data: models + optional presets.
#[derive(Debug, Clone)]
pub struct RegistryPayload {
    /// Model metadata entries.
    pub models: Vec<ModelMetadata>,
    /// Autoconfig presets (empty if not present or legacy format).
    pub presets: BTreeMap<String, BTreeMap<String, String>>,
}

/// Parse a registry payload, accepting both versioned and legacy formats.
///
/// Returns an error if the version is unsupported.
pub(crate) fn parse_payload(json: &str) -> Result<RegistryPayload> {
    let payload: RawPayload = serde_json::from_str(json).map_err(|e| {
        RatatoskrError::Configuration(format!("failed to parse registry JSON: {e}"))
    })?;
    match payload {
        RawPayload::Versioned(registry) => {
            if registry.version > MAX_SUPPORTED_VERSION {
                return Err(RatatoskrError::Configuration(format!(
                    "unsupported registry version {} (max supported: {MAX_SUPPORTED_VERSION})",
                    registry.version
                )));
            }
            Ok(RegistryPayload {
                models: registry.models,
                presets: registry.presets,
            })
        }
        RawPayload::Legacy(models) => Ok(RegistryPayload {
            models,
            presets: BTreeMap::new(),
        }),
    }
}

// ============================================================================
// Local file cache
// ============================================================================

/// Load cached registry data from disk.
///
/// Returns `None` on missing or corrupt file (logs a warning on corrupt).
pub fn load_cached(path: &Path) -> Option<RegistryPayload> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "failed to read cached registry");
            return None;
        }
    };
    match parse_payload(&content) {
        Ok(payload) => Some(payload),
        Err(e) => {
            warn!(path = %path.display(), error = %e, "corrupt cached registry");
            None
        }
    }
}

/// Save registry data to the local cache (atomic write via tmp + rename).
pub fn save_cache(path: &Path, payload: &RegistryPayload) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            RatatoskrError::Configuration(format!(
                "failed to create cache dir {}: {e}",
                parent.display()
            ))
        })?;
    }

    // Write to tmp file with PID suffix to avoid collisions between
    // concurrent processes, then rename for atomicity.
    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let registry = RemoteRegistry {
        version: 1,
        models: payload.models.to_vec(),
        presets: payload.presets.clone(),
    };
    let json = serde_json::to_string_pretty(&registry)
        .map_err(|e| RatatoskrError::Configuration(format!("failed to serialize registry: {e}")))?;
    std::fs::write(&tmp_path, &json).map_err(|e| {
        RatatoskrError::Configuration(format!(
            "failed to write cache file {}: {e}",
            tmp_path.display()
        ))
    })?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        RatatoskrError::Configuration(format!(
            "failed to rename cache file {} → {}: {e}",
            tmp_path.display(),
            path.display()
        ))
    })?;

    Ok(())
}

// ============================================================================
// Remote fetch
// ============================================================================

/// Fetch registry data from a remote URL.
///
/// Uses a 30-second timeout to prevent indefinite blocking.
pub async fn fetch_remote(url: &str) -> Result<RegistryPayload> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| RatatoskrError::Configuration(format!("failed to build HTTP client: {e}")))?;

    let response = client.get(url).send().await.map_err(|e| {
        RatatoskrError::Configuration(format!("failed to fetch registry from {url}: {e}"))
    })?;

    if !response.status().is_success() {
        return Err(RatatoskrError::Configuration(format!(
            "registry fetch returned HTTP {}",
            response.status()
        )));
    }

    let body = response.text().await.map_err(|e| {
        RatatoskrError::Configuration(format!("failed to read registry response body: {e}"))
    })?;

    parse_payload(&body)
}

/// Fetch remote registry and save to local cache.
///
/// Returns the fetched payload. This is the high-level entrypoint
/// used by `rat update-registry`.
pub async fn update_registry(config: &RemoteRegistryConfig) -> Result<RegistryPayload> {
    info!(url = %config.url, "fetching remote registry");
    let payload = fetch_remote(&config.url).await?;
    save_cache(&config.cache_path, &payload)?;
    info!(
        count = payload.models.len(),
        path = %config.cache_path.display(),
        "saved remote registry to cache"
    );
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelInfo;
    use std::collections::HashMap;

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

    #[test]
    fn parse_versioned_format() {
        let json = serde_json::to_string(&RemoteRegistry {
            version: 1,
            models: vec![sample_metadata("model-a")],
            presets: BTreeMap::new(),
        })
        .unwrap();
        let payload = parse_payload(&json).unwrap();
        assert_eq!(payload.models.len(), 1);
        assert_eq!(payload.models[0].info.id, "model-a");
    }

    #[test]
    fn parse_legacy_bare_array() {
        let json = serde_json::to_string(&vec![sample_metadata("model-b")]).unwrap();
        let payload = parse_payload(&json).unwrap();
        assert_eq!(payload.models.len(), 1);
        assert_eq!(payload.models[0].info.id, "model-b");
        assert!(payload.presets.is_empty());
    }

    #[test]
    fn parse_unsupported_version_rejected() {
        let json = r#"{"version": 999, "models": []}"#;
        let result = parse_payload(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported registry version"));
    }

    #[test]
    fn parse_invalid_json_rejected() {
        let result = parse_payload("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        let payload = sample_payload(vec![sample_metadata("model-a"), sample_metadata("model-b")]);
        save_cache(&path, &payload).unwrap();

        let loaded = load_cached(&path).unwrap();
        assert_eq!(loaded.models.len(), 2);
        assert_eq!(loaded.models[0].info.id, "model-a");
        assert_eq!(loaded.models[1].info.id, "model-b");
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let result = load_cached(Path::new("/nonexistent/path/registry.json"));
        assert!(result.is_none());
    }

    #[test]
    fn load_corrupt_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        std::fs::write(&path, "this is not valid json").unwrap();

        let result = load_cached(&path);
        assert!(result.is_none());
    }

    #[test]
    fn save_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep").join("nested").join("registry.json");

        save_cache(&path, &sample_payload(vec![sample_metadata("model-a")])).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn default_config_uses_github_url() {
        let config = RemoteRegistryConfig::default();
        assert!(config.url.contains("ratatoskr-registry"));
        assert!(config.cache_path.ends_with("registry.json"));
    }

    #[test]
    fn config_with_url() {
        let config = RemoteRegistryConfig::with_url("https://example.com/reg.json");
        assert_eq!(config.url, "https://example.com/reg.json");
    }

    #[test]
    fn presets_round_trip_preserves_all_slots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        let json = r#"{
            "version": 1,
            "models": [],
            "presets": {
                "budget": {
                    "agentic": "x/model-a",
                    "embedding": "x/model-b",
                    "summariser": "x/model-c",
                    "text-generation": "x/model-d"
                },
                "premium": {
                    "agentic": "x/model-e",
                    "summariser": "x/model-f"
                }
            }
        }"#;

        let payload = parse_payload(json).unwrap();
        assert_eq!(
            payload.presets["budget"].get("summariser"),
            Some(&"x/model-c".to_string()),
            "summariser slot must survive parse"
        );

        save_cache(&path, &payload).unwrap();
        let loaded = load_cached(&path).unwrap();

        assert_eq!(
            loaded.presets["budget"].get("summariser"),
            Some(&"x/model-c".to_string()),
            "summariser slot must survive save+load"
        );
        assert_eq!(
            loaded.presets["premium"].get("summariser"),
            Some(&"x/model-f".to_string()),
        );
        assert_eq!(
            loaded.presets["budget"].len(),
            4,
            "all 4 budget slots preserved"
        );
        assert_eq!(
            loaded.presets["premium"].len(),
            2,
            "all 2 premium slots preserved"
        );
    }
}
