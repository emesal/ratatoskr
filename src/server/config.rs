//! Configuration loading for ratd.
//!
//! Configuration is loaded from TOML files with the following resolution order:
//! 1. `--config <path>` (CLI flag)
//! 2. `~/.ratatoskr/config.toml` (user)
//! 3. `/etc/ratatoskr/config.toml` (system)
//!
//! Secrets are loaded separately with mandatory permission checks:
//! 1. `~/.ratatoskr/secrets.toml` (user, must be 0600)
//! 2. `/etc/ratatoskr/secrets.toml` (system, must be 0600)

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::providers::RoutingConfig;
use crate::{RatatoskrError, Result};

/// Server configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub discovery: DiscoveryTomlConfig,
    #[serde(default)]
    pub registry: Option<RegistryTomlConfig>,
}

/// Remote registry configuration (TOML section).
///
/// ```toml
/// [registry]
/// remote_url = "https://raw.githubusercontent.com/..."
/// cache_path = "~/.cache/ratatoskr/registry.json"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RegistryTomlConfig {
    /// URL to fetch the registry from. Default: emesal/ratatoskr-registry on GitHub.
    #[serde(default = "default_registry_url")]
    pub remote_url: String,
    /// Local cache path. Default: ~/.cache/ratatoskr/registry.json.
    #[serde(default)]
    pub cache_path: Option<PathBuf>,
}

fn default_registry_url() -> String {
    crate::registry::remote::DEFAULT_REGISTRY_URL.to_string()
}

impl From<RegistryTomlConfig> for crate::registry::remote::RemoteRegistryConfig {
    fn from(toml: RegistryTomlConfig) -> Self {
        crate::registry::remote::RemoteRegistryConfig {
            url: toml.remote_url,
            cache_path: toml.cache_path.unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from(".cache"))
                    .join("ratatoskr")
                    .join("registry.json")
            }),
        }
    }
}

/// Runtime parameter discovery configuration (TOML section).
///
/// ```toml
/// [discovery]
/// enabled = true
/// ttl_hours = 24
/// max_entries = 1000
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DiscoveryTomlConfig {
    /// Whether parameter discovery is enabled. Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Time-to-live for cached entries, in hours. Default: 24.
    #[serde(default = "default_ttl_hours")]
    pub ttl_hours: u64,
    /// Maximum number of cached entries. Default: 1,000.
    #[serde(default = "default_discovery_entries")]
    pub max_entries: u64,
}

impl Default for DiscoveryTomlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_hours: default_ttl_hours(),
            max_entries: default_discovery_entries(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_ttl_hours() -> u64 {
    24
}

fn default_discovery_entries() -> u64 {
    1_000
}

impl From<DiscoveryTomlConfig> for crate::DiscoveryConfig {
    fn from(toml: DiscoveryTomlConfig) -> Self {
        crate::DiscoveryConfig::new()
            .max_entries(toml.max_entries)
            .ttl(std::time::Duration::from_secs(toml.ttl_hours * 3600))
    }
}

/// Server network configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Address to bind to (default: 127.0.0.1:9741).
    #[serde(default = "default_address")]
    pub address: String,
    #[serde(default)]
    pub limits: LimitsConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            address: default_address(),
            limits: LimitsConfig::default(),
        }
    }
}

fn default_address() -> String {
    "127.0.0.1:9741".to_string()
}

/// Resource limits.
#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    /// Maximum concurrent requests (default: 100).
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds (default: 30).
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: default_max_concurrent(),
            request_timeout_secs: default_timeout(),
        }
    }
}

fn default_max_concurrent() -> usize {
    100
}

fn default_timeout() -> u64 {
    30
}

/// Provider configurations.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openrouter: Option<ApiProviderConfig>,
    #[serde(default)]
    pub anthropic: Option<ApiProviderConfig>,
    #[serde(default)]
    pub openai: Option<ApiProviderConfig>,
    #[serde(default)]
    pub google: Option<ApiProviderConfig>,
    #[serde(default)]
    pub ollama: Option<OllamaConfig>,
    #[serde(default)]
    pub huggingface: Option<ApiProviderConfig>,
    #[serde(default)]
    pub local: Option<LocalConfig>,
}

/// API provider configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiProviderConfig {
    /// Default model to use when none specified.
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Ollama-specific configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaConfig {
    /// Ollama base URL (default: http://localhost:11434).
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

/// Local inference configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LocalConfig {
    /// Device to use: "cpu" or "cuda" (default: "cpu").
    #[serde(default = "default_device")]
    pub device: String,
    /// Directory for model downloads.
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    /// RAM budget in megabytes for local model loading.
    #[serde(default)]
    pub ram_budget_mb: Option<usize>,
}

fn default_device() -> String {
    "cpu".to_string()
}

// RoutingConfig is re-used from crate::providers::routing (single source of truth).

/// Secrets configuration (API keys).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Secrets {
    #[serde(default)]
    pub openrouter: Option<ApiKeySecret>,
    #[serde(default)]
    pub anthropic: Option<ApiKeySecret>,
    #[serde(default)]
    pub openai: Option<ApiKeySecret>,
    #[serde(default)]
    pub google: Option<ApiKeySecret>,
    #[serde(default)]
    pub huggingface: Option<ApiKeySecret>,
}

/// A single API key secret.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeySecret {
    pub api_key: String,
}

/// Provider name → environment variable name mapping.
const PROVIDER_ENV_VARS: &[(&str, &str)] = &[
    ("openrouter", "OPENROUTER_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("google", "GOOGLE_API_KEY"),
    ("huggingface", "HF_API_KEY"),
];

impl Config {
    /// Load configuration from the standard locations.
    ///
    /// Resolution order:
    /// 1. Explicit path (if provided)
    /// 2. `~/.ratatoskr/config.toml`
    /// 3. `/etc/ratatoskr/config.toml`
    pub fn load(explicit_path: Option<&Path>) -> Result<Self> {
        let path = Self::resolve_config_path(explicit_path)?;
        let content = fs::read_to_string(&path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to read config file {path:?}: {e}"))
        })?;
        toml::from_str(&content).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to parse config file {path:?}: {e}"))
        })
    }

    /// Resolve the config file path.
    fn resolve_config_path(explicit: Option<&Path>) -> Result<PathBuf> {
        if let Some(path) = explicit {
            if path.exists() {
                return Ok(path.to_path_buf());
            }
            return Err(RatatoskrError::Configuration(format!(
                "Config file not found: {path:?}"
            )));
        }

        // User config
        if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".ratatoskr").join("config.toml");
            if user_config.exists() {
                return Ok(user_config);
            }
        }

        // System config
        let system_config = PathBuf::from("/etc/ratatoskr/config.toml");
        if system_config.exists() {
            return Ok(system_config);
        }

        Err(RatatoskrError::Configuration(
            "No config file found. Create ~/.ratatoskr/config.toml or /etc/ratatoskr/config.toml"
                .to_string(),
        ))
    }
}

impl Secrets {
    /// Load secrets from the standard locations with permission checks.
    ///
    /// Resolution order:
    /// 1. `~/.ratatoskr/secrets.toml` (if exists, must be 0600)
    /// 2. `/etc/ratatoskr/secrets.toml` (if exists, must be 0600)
    ///
    /// Returns empty secrets if no file exists (providers may use env vars).
    pub fn load() -> Result<Self> {
        // Try user secrets first
        if let Some(home) = dirs::home_dir() {
            let user_secrets = home.join(".ratatoskr").join("secrets.toml");
            if user_secrets.exists() {
                Self::check_permissions(&user_secrets)?;
                return Self::load_from_file(&user_secrets);
            }
        }

        // Try system secrets
        let system_secrets = PathBuf::from("/etc/ratatoskr/secrets.toml");
        if system_secrets.exists() {
            Self::check_permissions(&system_secrets)?;
            return Self::load_from_file(&system_secrets);
        }

        // No secrets file — return empty (providers can fall back to env vars)
        Ok(Secrets::default())
    }

    fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to read secrets file {path:?}: {e}"))
        })?;
        toml::from_str(&content).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to parse secrets file {path:?}: {e}"))
        })
    }

    /// Check that the secrets file has secure permissions (0600 or 0400).
    #[cfg(unix)]
    fn check_permissions(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::metadata(path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to stat secrets file {path:?}: {e}"))
        })?;

        let mode = metadata.permissions().mode();
        // Reject if group or other bits are set
        if mode & 0o077 != 0 {
            return Err(RatatoskrError::Configuration(format!(
                "Secrets file {path:?} has insecure permissions {:o}. Must be 0600 or 0400.",
                mode & 0o777
            )));
        }

        Ok(())
    }

    #[cfg(not(unix))]
    fn check_permissions(_path: &Path) -> Result<()> {
        // Permission check not available on non-Unix platforms
        Ok(())
    }

    /// Get API key for a provider, falling back to the corresponding environment variable.
    pub fn api_key(&self, provider: &str) -> Option<String> {
        // Try secrets file first
        let from_file = match provider {
            "openrouter" => self.openrouter.as_ref(),
            "anthropic" => self.anthropic.as_ref(),
            "openai" => self.openai.as_ref(),
            "google" => self.google.as_ref(),
            "huggingface" => self.huggingface.as_ref(),
            _ => None,
        }
        .map(|s| s.api_key.clone());

        // Fall back to env var
        from_file.or_else(|| {
            PROVIDER_ENV_VARS
                .iter()
                .find(|(name, _)| *name == provider)
                .and_then(|(_, env_var)| std::env::var(env_var).ok())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = Config {
            server: ServerConfig::default(),
            providers: ProvidersConfig::default(),
            routing: RoutingConfig::default(),
            discovery: DiscoveryTomlConfig::default(),
        };
        assert_eq!(config.server.address, "127.0.0.1:9741");
        assert_eq!(config.server.limits.max_concurrent_requests, 100);
        assert_eq!(config.server.limits.request_timeout_secs, 30);
    }

    #[test]
    fn parse_minimal_config() {
        let toml = r#"
            [server]
            address = "0.0.0.0:9741"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.address, "0.0.0.0:9741");
        // Defaults preserved
        assert_eq!(config.server.limits.max_concurrent_requests, 100);
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
            [server]
            address = "127.0.0.1:9741"

            [server.limits]
            max_concurrent_requests = 50
            request_timeout_secs = 60

            [providers.openrouter]
            default_model = "anthropic/claude-sonnet-4"

            [providers.ollama]
            base_url = "http://localhost:11434"

            [routing]
            chat = "openrouter"
            embed = "local"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.limits.max_concurrent_requests, 50);
        assert_eq!(config.server.limits.request_timeout_secs, 60);
        assert_eq!(
            config.providers.openrouter.as_ref().unwrap().default_model,
            Some("anthropic/claude-sonnet-4".to_string())
        );
        assert_eq!(
            config.providers.ollama.as_ref().unwrap().base_url,
            "http://localhost:11434"
        );
        assert_eq!(config.routing.chat, Some("openrouter".to_string()));
        assert_eq!(config.routing.embed, Some("local".to_string()));
    }

    #[test]
    fn parse_secrets() {
        let toml = r#"
            [openrouter]
            api_key = "sk-or-test-key"

            [anthropic]
            api_key = "sk-ant-test-key"
        "#;
        let secrets: Secrets = toml::from_str(toml).unwrap();
        assert_eq!(
            secrets.openrouter.as_ref().unwrap().api_key,
            "sk-or-test-key"
        );
        assert_eq!(
            secrets.anthropic.as_ref().unwrap().api_key,
            "sk-ant-test-key"
        );
        assert!(secrets.openai.is_none());
    }

    #[test]
    fn api_key_from_secrets() {
        let secrets = Secrets {
            openrouter: Some(ApiKeySecret {
                api_key: "from-file".to_string(),
            }),
            ..Default::default()
        };
        assert_eq!(secrets.api_key("openrouter"), Some("from-file".to_string()));
        // Unknown provider returns None
        assert_eq!(secrets.api_key("nonexistent"), None);
    }

    #[test]
    fn config_not_found_returns_error() {
        let result = Config::load(Some(Path::new("/nonexistent/config.toml")));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Config file not found"));
    }

    #[test]
    fn parse_local_config() {
        let toml = r#"
            [providers.local]
            device = "cuda"
            models_dir = "/opt/models"
            ram_budget_mb = 2048
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let local = config.providers.local.unwrap();
        assert_eq!(local.device, "cuda");
        assert_eq!(local.models_dir, Some(PathBuf::from("/opt/models")));
        assert_eq!(local.ram_budget_mb, Some(2048));
    }

    #[test]
    fn parse_discovery_config() {
        let toml = r#"
            [discovery]
            enabled = false
            ttl_hours = 12
            max_entries = 500
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(!config.discovery.enabled);
        assert_eq!(config.discovery.ttl_hours, 12);
        assert_eq!(config.discovery.max_entries, 500);
    }

    #[test]
    fn discovery_defaults_when_omitted() {
        let toml = r#"
            [server]
            address = "127.0.0.1:9741"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(config.discovery.enabled);
        assert_eq!(config.discovery.ttl_hours, 24);
        assert_eq!(config.discovery.max_entries, 1_000);
    }

    #[test]
    fn parse_registry_config() {
        let toml = r#"
            [registry]
            remote_url = "https://example.com/registry.json"
            cache_path = "/tmp/registry.json"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let registry = config.registry.unwrap();
        assert_eq!(registry.remote_url, "https://example.com/registry.json");
        assert_eq!(
            registry.cache_path,
            Some(PathBuf::from("/tmp/registry.json"))
        );
    }

    #[test]
    fn registry_defaults_when_omitted() {
        let toml = r#"
            [server]
            address = "127.0.0.1:9741"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(config.registry.is_none());
    }

    #[test]
    fn registry_toml_to_config_conversion() {
        let toml_config = RegistryTomlConfig {
            remote_url: "https://example.com/reg.json".to_string(),
            cache_path: Some(PathBuf::from("/tmp/reg.json")),
        };
        let config: crate::registry::remote::RemoteRegistryConfig = toml_config.into();
        assert_eq!(config.url, "https://example.com/reg.json");
        assert_eq!(config.cache_path, PathBuf::from("/tmp/reg.json"));
    }

    #[test]
    fn discovery_toml_to_config_conversion() {
        let toml_config = DiscoveryTomlConfig {
            enabled: true,
            ttl_hours: 12,
            max_entries: 500,
        };
        let config: crate::DiscoveryConfig = toml_config.into();
        assert_eq!(config.max_entries, 500);
        assert_eq!(config.ttl, std::time::Duration::from_secs(12 * 3600));
    }
}
