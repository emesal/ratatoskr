//! Runtime parameter discovery cache.
//!
//! Records parameters that providers reject at runtime (e.g. HTTP 400 for
//! `parallel_tool_calls`). The validation path in [`ProviderRegistry`](crate::providers::ProviderRegistry)
//! consults this cache alongside static declarations, preventing repeated
//! failures for the same parameter.
//!
//! # Architecture
//!
//! - Moka-backed LRU + TTL cache, keyed on `hash(provider, model, parameter)`.
//! - On by default with sensible defaults; opt-out via
//!   [`RatatoskrBuilder::disable_parameter_discovery()`](crate::RatatoskrBuilder::disable_parameter_discovery).
//! - Forward-compatible with future aggregation (#14): [`DiscoveryRecord`]
//!   stores all fields needed for cross-session analysis.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use moka::sync::Cache;

use crate::types::ParameterName;

/// A single parameter rejection observed at runtime.
///
/// Stored in [`ParameterDiscoveryCache`] and designed for future aggregation
/// (issue #14) — all contextual fields are preserved.
#[derive(Debug, Clone)]
pub struct DiscoveryRecord {
    /// The rejected parameter.
    pub parameter: ParameterName,
    /// Provider that rejected it (e.g. `"openrouter"`, `"anthropic"`).
    pub provider: String,
    /// Model ID the rejection was observed for.
    pub model: String,
    /// When the rejection was recorded.
    pub discovered_at: Instant,
    /// Human-readable reason (typically the error message).
    pub reason: String,
}

/// Configuration for the parameter discovery cache.
///
/// ```rust
/// # use ratatoskr::DiscoveryConfig;
/// # use std::time::Duration;
/// let config = DiscoveryConfig::new()
///     .max_entries(2_000)
///     .ttl(Duration::from_secs(12 * 3600));
/// ```
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Maximum cached entries. Default: 1,000.
    pub max_entries: u64,
    /// Time-to-live for cached entries. Default: 24 hours.
    pub ttl: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000,
            ttl: Duration::from_secs(24 * 3600),
        }
    }
}

impl DiscoveryConfig {
    /// Create a config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of cached entries.
    pub fn max_entries(mut self, n: u64) -> Self {
        self.max_entries = n;
        self
    }

    /// Set the time-to-live for cached entries.
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }
}

/// In-memory cache of runtime parameter rejections.
///
/// Thread-safe (moka handles concurrent access internally). Keyed on
/// `hash(provider, model, parameter)` using `DefaultHasher` — the same
/// pattern as [`ResponseCache`](crate::ResponseCache).
pub struct ParameterDiscoveryCache {
    cache: Cache<u64, DiscoveryRecord>,
}

impl ParameterDiscoveryCache {
    /// Create a new cache from the given configuration.
    pub fn new(config: &DiscoveryConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.max_entries)
            .time_to_live(config.ttl)
            .build();
        Self { cache }
    }

    /// Record a parameter rejection.
    pub fn record(&self, record: DiscoveryRecord) {
        let key = discovery_key(&record.provider, &record.model, &record.parameter);
        self.cache.insert(key, record);
    }

    /// Check whether a specific `(provider, model, parameter)` triple has a
    /// known rejection in the cache.
    pub fn is_known_unsupported(&self, provider: &str, model: &str, param: &ParameterName) -> bool {
        let key = discovery_key(provider, model, param);
        self.cache.contains_key(&key)
    }

    /// Batch query: return the subset of `params` that are known-unsupported
    /// for the given provider and model.
    pub fn known_unsupported_params(
        &self,
        provider: &str,
        model: &str,
        params: &[ParameterName],
    ) -> Vec<ParameterName> {
        params
            .iter()
            .filter(|p| self.is_known_unsupported(provider, model, p))
            .cloned()
            .collect()
    }

    /// Return all active (non-expired) discovery records.
    ///
    /// Intended for future aggregation (#14). Order is not guaranteed.
    pub fn list_discoveries(&self) -> Vec<DiscoveryRecord> {
        self.cache.iter().map(|(_, v)| v).collect()
    }
}

/// Compute a cache key from `(provider, model, parameter)`.
///
/// Uses `DefaultHasher` (SipHash) — deterministic within a process lifetime,
/// which is sufficient for an in-memory cache.
fn discovery_key(provider: &str, model: &str, param: &ParameterName) -> u64 {
    let mut hasher = DefaultHasher::new();
    provider.hash(&mut hasher);
    model.hash(&mut hasher);
    param.as_str().hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DiscoveryConfig {
        DiscoveryConfig::new().max_entries(100)
    }

    fn make_record(provider: &str, model: &str, param: ParameterName) -> DiscoveryRecord {
        DiscoveryRecord {
            parameter: param,
            provider: provider.to_string(),
            model: model.to_string(),
            discovered_at: Instant::now(),
            reason: "unsupported by provider".to_string(),
        }
    }

    #[test]
    fn insert_and_query_round_trip() {
        let cache = ParameterDiscoveryCache::new(&test_config());
        let record = make_record("openrouter", "gpt-4", ParameterName::ParallelToolCalls);
        cache.record(record);

        assert!(cache.is_known_unsupported(
            "openrouter",
            "gpt-4",
            &ParameterName::ParallelToolCalls
        ));
    }

    #[test]
    fn unknown_triple_returns_false() {
        let cache = ParameterDiscoveryCache::new(&test_config());

        assert!(!cache.is_known_unsupported(
            "openrouter",
            "gpt-4",
            &ParameterName::ParallelToolCalls
        ));
    }

    #[test]
    fn same_param_different_models_are_independent() {
        let cache = ParameterDiscoveryCache::new(&test_config());
        let record = make_record("openrouter", "gpt-4", ParameterName::ParallelToolCalls);
        cache.record(record);

        // Same provider + param, different model → not cached
        assert!(!cache.is_known_unsupported(
            "openrouter",
            "claude-3",
            &ParameterName::ParallelToolCalls
        ));
        // Original still present
        assert!(cache.is_known_unsupported(
            "openrouter",
            "gpt-4",
            &ParameterName::ParallelToolCalls
        ));
    }

    #[test]
    fn same_param_different_providers_are_independent() {
        let cache = ParameterDiscoveryCache::new(&test_config());
        let record = make_record("openrouter", "gpt-4", ParameterName::ParallelToolCalls);
        cache.record(record);

        assert!(!cache.is_known_unsupported(
            "anthropic",
            "gpt-4",
            &ParameterName::ParallelToolCalls
        ));
    }

    #[test]
    fn batch_query_returns_matching_subset() {
        let cache = ParameterDiscoveryCache::new(&test_config());
        cache.record(make_record(
            "openrouter",
            "gpt-4",
            ParameterName::ParallelToolCalls,
        ));
        cache.record(make_record("openrouter", "gpt-4", ParameterName::Seed));

        let params = vec![
            ParameterName::ParallelToolCalls,
            ParameterName::Temperature,
            ParameterName::Seed,
        ];
        let unsupported = cache.known_unsupported_params("openrouter", "gpt-4", &params);

        assert_eq!(unsupported.len(), 2);
        assert!(unsupported.contains(&ParameterName::ParallelToolCalls));
        assert!(unsupported.contains(&ParameterName::Seed));
    }

    #[test]
    fn list_discoveries_returns_active_entries() {
        let cache = ParameterDiscoveryCache::new(&test_config());
        cache.record(make_record(
            "openrouter",
            "gpt-4",
            ParameterName::ParallelToolCalls,
        ));
        cache.record(make_record("anthropic", "claude-3", ParameterName::Seed));

        let entries = cache.list_discoveries();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn ttl_expiry() {
        // Use a very short TTL
        let config = DiscoveryConfig::new()
            .max_entries(100)
            .ttl(Duration::from_millis(1));
        let cache = ParameterDiscoveryCache::new(&config);

        cache.record(make_record(
            "openrouter",
            "gpt-4",
            ParameterName::ParallelToolCalls,
        ));

        // Sleep past TTL
        std::thread::sleep(Duration::from_millis(50));

        // Moka's sync cache may need a get to trigger lazy expiry
        assert!(!cache.is_known_unsupported(
            "openrouter",
            "gpt-4",
            &ParameterName::ParallelToolCalls
        ));
    }

    #[test]
    fn config_builder_pattern() {
        let config = DiscoveryConfig::new()
            .max_entries(500)
            .ttl(Duration::from_secs(3600));
        assert_eq!(config.max_entries, 500);
        assert_eq!(config.ttl, Duration::from_secs(3600));
    }
}
