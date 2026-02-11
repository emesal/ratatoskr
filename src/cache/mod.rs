//! Caching subsystem.
//!
//! Two independent caches:
//!
//! - [`ModelCache`] — ephemeral model metadata fetched from providers at
//!   runtime. Populated by [`ModelGateway::fetch_model_metadata()`](crate::ModelGateway::fetch_model_metadata),
//!   consulted as a fallback when the registry has no entry.
//!
//! - [`response::ResponseCache`] — opt-in LRU + TTL cache for deterministic
//!   operations (embeddings, NLI). Activated via the builder's
//!   `.response_cache()` method. See [`response`] module docs for
//!   architecture and future extensibility notes.

pub mod response;
pub use response::{CacheConfig, ResponseCache};

use std::collections::HashMap;
use std::sync::RwLock;

use crate::types::ModelMetadata;

/// Thread-safe ephemeral store for provider-fetched model metadata.
///
/// Keyed on model ID. Entries are cloned on read to avoid holding
/// locks across async boundaries.
pub struct ModelCache {
    entries: RwLock<HashMap<String, ModelMetadata>>,
}

impl ModelCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Look up cached metadata for a model.
    ///
    /// Returns `None` on cache miss. Clones the entry to release the
    /// read lock immediately.
    pub fn get(&self, model: &str) -> Option<ModelMetadata> {
        self.entries
            .read()
            .expect("cache lock poisoned")
            .get(model)
            .cloned()
    }

    /// Insert (or overwrite) metadata, keyed on `metadata.info.id`.
    pub fn insert(&self, metadata: ModelMetadata) {
        self.entries
            .write()
            .expect("cache lock poisoned")
            .insert(metadata.info.id.clone(), metadata);
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}
