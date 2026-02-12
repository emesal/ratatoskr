//! Caching subsystem.
//!
//! Three independent caches:
//!
//! - [`ModelCache`] — ephemeral model metadata fetched from providers at
//!   runtime. Populated by [`ModelGateway::fetch_model_metadata()`](crate::ModelGateway::fetch_model_metadata),
//!   consulted as a fallback when the registry has no entry.
//!
//! - [`response::ResponseCache`] — opt-in LRU + TTL cache for deterministic
//!   operations (embeddings, NLI). Activated via the builder's
//!   `.response_cache()` method. See [`response`] module docs for
//!   architecture and future extensibility notes.
//!
//! - [`discovery::ParameterDiscoveryCache`] — records parameter rejections
//!   from providers at runtime, preventing repeated failures. On by default;
//!   opt-out via [`RatatoskrBuilder::disable_parameter_discovery()`](crate::RatatoskrBuilder::disable_parameter_discovery).

pub mod discovery;
pub mod response;

pub use discovery::{DiscoveryConfig, DiscoveryRecord, ParameterDiscoveryCache};
pub use response::{CacheConfig, ResponseCache};

use crate::types::ModelMetadata;

/// Default maximum number of entries in the model metadata cache.
const DEFAULT_MODEL_CACHE_MAX: u64 = 1_000;

/// Thread-safe ephemeral store for provider-fetched model metadata.
///
/// Keyed on model ID. Uses a bounded LRU cache (moka) to prevent unbounded
/// growth in long-running processes. Default capacity: 1,000 entries.
pub struct ModelCache {
    entries: moka::sync::Cache<String, ModelMetadata>,
}

impl ModelCache {
    /// Create an empty cache with the default max capacity (1,000).
    pub fn new() -> Self {
        Self::with_max_entries(DEFAULT_MODEL_CACHE_MAX)
    }

    /// Create a cache with a custom max capacity.
    pub fn with_max_entries(max: u64) -> Self {
        Self {
            entries: moka::sync::Cache::new(max),
        }
    }

    /// Look up cached metadata for a model.
    ///
    /// Returns `None` on cache miss.
    pub fn get(&self, model: &str) -> Option<ModelMetadata> {
        self.entries.get(model)
    }

    /// Insert (or overwrite) metadata, keyed on `metadata.info.id`.
    pub fn insert(&self, metadata: ModelMetadata) {
        self.entries.insert(metadata.info.id.clone(), metadata);
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> u64 {
        self.entries.entry_count()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evict all entries.
    pub fn clear(&self) {
        self.entries.invalidate_all();
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}
