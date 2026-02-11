//! Opt-in response cache for deterministic operations.
//!
//! [`ResponseCache`] caches responses from embedding and NLI operations,
//! which are deterministic (same input → same output). Chat and generate
//! operations are intentionally excluded — provider-side caching (e.g.
//! `cache_prompt`) is more appropriate for non-deterministic responses.
//!
//! # Architecture
//!
//! The cache sits in [`EmbeddedGateway`](crate::gateway::EmbeddedGateway),
//! above the [`ProviderRegistry`](crate::providers::ProviderRegistry) fallback
//! chain. A cache hit bypasses retry logic, provider selection, and metrics
//! for provider calls entirely. Cache hit/miss metrics are emitted separately.
//!
//! # Future extensibility: shared/distributed caching
//!
//! The current implementation uses moka's in-memory LRU cache, owned
//! per-gateway instance. When shared caching is needed (e.g. redis-backed
//! for multiple ratd instances or cross-process deduplication), extract a
//! trait and inject implementations via the builder:
//!
//! ```rust,ignore
//! #[async_trait]
//! trait CacheBackend: Send + Sync {
//!     async fn get(&self, key: u64) -> Option<CachedResponse>;
//!     async fn insert(&self, key: u64, value: CachedResponse);
//! }
//! ```
//!
//! The key design (content hash of operation + model + input) is
//! backend-agnostic, so the refactor is straightforward. The insertion
//! point is [`EmbeddedGateway`](crate::gateway::EmbeddedGateway) — replace
//! the `Option<ResponseCache>` field with `Option<Arc<dyn CacheBackend>>`.
//! All cache interactions go through the same gateway methods (`embed`,
//! `embed_batch`, `infer_nli`), so no other modules need changes.
//!
//! # Batch decomposition
//!
//! `embed_batch` uses per-item cache lookup: each text in the batch is
//! checked individually, only cache misses are forwarded to the provider,
//! and results are reassembled in original order. This means a single
//! `embed("hello", model)` call populates the cache entry that a later
//! `embed_batch(["hello", "world"], model)` can partially hit.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use moka::future::Cache;

use crate::telemetry;
use crate::types::{Embedding, NliResult};

/// Configuration for the response cache.
///
/// Enables caching of deterministic responses (embeddings, NLI).
/// Pass to [`RatatoskrBuilder::response_cache()`](crate::RatatoskrBuilder::response_cache)
/// to activate. Without this, no cache is allocated (zero overhead).
///
/// ```rust
/// # use ratatoskr::CacheConfig;
/// # use std::time::Duration;
/// let config = CacheConfig::new()
///     .max_entries(10_000)
///     .ttl(Duration::from_secs(3600));
/// ```
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached entries. Default: 10,000.
    pub max_entries: u64,
    /// Time-to-live for cached entries. Default: 1 hour.
    pub ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: Duration::from_secs(3600),
        }
    }
}

impl CacheConfig {
    /// Create a new config with sensible defaults.
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

/// Cached response value — either an embedding or an NLI result.
#[derive(Clone, Debug)]
pub(crate) enum CachedResponse {
    Embedding(Embedding),
    Nli(NliResult),
}

/// In-memory response cache for deterministic operations.
///
/// Uses moka's async-friendly LRU + TTL cache. Keyed on a content hash
/// of (operation, model, input). See module docs for architecture and
/// future extensibility notes.
pub struct ResponseCache {
    cache: Cache<u64, CachedResponse>,
}

impl ResponseCache {
    /// Create a new response cache with the given configuration.
    pub fn new(config: &CacheConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(config.max_entries)
            .time_to_live(config.ttl)
            .build();
        Self { cache }
    }

    /// Look up a cached embedding.
    ///
    /// Returns `None` on cache miss. Emits cache hit/miss metrics.
    pub async fn get_embedding(&self, model: &str, text: &str) -> Option<Embedding> {
        let key = cache_key("embed", model, &[text]);
        match self.cache.get(&key).await {
            Some(CachedResponse::Embedding(e)) => {
                metrics::counter!(telemetry::CACHE_HITS_TOTAL, "operation" => "embed").increment(1);
                Some(e)
            }
            _ => {
                metrics::counter!(telemetry::CACHE_MISSES_TOTAL, "operation" => "embed")
                    .increment(1);
                None
            }
        }
    }

    /// Insert a cached embedding.
    pub async fn insert_embedding(&self, model: &str, text: &str, embedding: Embedding) {
        let key = cache_key("embed", model, &[text]);
        self.cache
            .insert(key, CachedResponse::Embedding(embedding))
            .await;
    }

    /// Look up a cached NLI result.
    ///
    /// Returns `None` on cache miss. Emits cache hit/miss metrics.
    pub async fn get_nli(&self, model: &str, premise: &str, hypothesis: &str) -> Option<NliResult> {
        let key = cache_key("nli", model, &[premise, hypothesis]);
        match self.cache.get(&key).await {
            Some(CachedResponse::Nli(r)) => {
                metrics::counter!(telemetry::CACHE_HITS_TOTAL, "operation" => "nli").increment(1);
                Some(r)
            }
            _ => {
                metrics::counter!(telemetry::CACHE_MISSES_TOTAL, "operation" => "nli").increment(1);
                None
            }
        }
    }

    /// Insert a cached NLI result.
    pub async fn insert_nli(
        &self,
        model: &str,
        premise: &str,
        hypothesis: &str,
        result: NliResult,
    ) {
        let key = cache_key("nli", model, &[premise, hypothesis]);
        self.cache.insert(key, CachedResponse::Nli(result)).await;
    }

    /// Check cache for multiple texts, returning hits and miss indices.
    ///
    /// Used by `embed_batch` for per-item decomposition. Returns a vec
    /// of `Option<Embedding>` in the same order as `texts` — `Some` for
    /// hits, `None` for misses. The caller forwards only misses to the
    /// provider, then reassembles using [`merge_batch_results`].
    pub async fn get_embedding_batch(&self, model: &str, texts: &[&str]) -> Vec<Option<Embedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.get_embedding(model, text).await);
        }
        results
    }

    /// Insert multiple embeddings into the cache.
    ///
    /// `texts` and `embeddings` must be the same length and correspond
    /// positionally (i.e. `embeddings[i]` is the result for `texts[i]`).
    pub async fn insert_embedding_batch(
        &self,
        model: &str,
        texts: &[&str],
        embeddings: &[Embedding],
    ) {
        for (text, embedding) in texts.iter().zip(embeddings.iter()) {
            self.insert_embedding(model, text, embedding.clone()).await;
        }
    }
}

/// Compute a cache key from operation, model, and input strings.
///
/// Uses `DefaultHasher` (SipHash) for a reasonable collision-resistance /
/// performance trade-off. The hash is deterministic within a process
/// lifetime, which is sufficient for an in-memory cache.
///
/// For a future distributed backend, replace with a stable hash (e.g.
/// xxhash or SHA-256 prefix) that is consistent across processes.
fn cache_key(operation: &str, model: &str, input: &[&str]) -> u64 {
    let mut hasher = DefaultHasher::new();
    operation.hash(&mut hasher);
    model.hash(&mut hasher);
    for s in input {
        s.hash(&mut hasher);
    }
    hasher.finish()
}

/// Merge cached hits with provider results for a batch operation.
///
/// Given the original `cached` lookup (Some = hit, None = miss) and the
/// provider `results` for just the misses, reassembles the full output
/// in original order.
///
/// # Panics
///
/// Panics if `results.len()` doesn't match the number of `None` entries
/// in `cached` (i.e. the provider didn't return the expected count).
pub(crate) fn merge_batch_results(
    cached: Vec<Option<Embedding>>,
    results: Vec<Embedding>,
) -> Vec<Embedding> {
    let mut result_iter = results.into_iter();
    cached
        .into_iter()
        .map(|opt| {
            opt.unwrap_or_else(|| {
                result_iter
                    .next()
                    .expect("provider returned fewer results than expected")
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_deterministic() {
        let k1 = cache_key("embed", "model-a", &["hello"]);
        let k2 = cache_key("embed", "model-a", &["hello"]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_operation() {
        let k1 = cache_key("embed", "model-a", &["hello"]);
        let k2 = cache_key("nli", "model-a", &["hello"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_model() {
        let k1 = cache_key("embed", "model-a", &["hello"]);
        let k2 = cache_key("embed", "model-b", &["hello"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_input() {
        let k1 = cache_key("embed", "model-a", &["hello"]);
        let k2 = cache_key("embed", "model-a", &["world"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_nli_order_matters() {
        let k1 = cache_key("nli", "model", &["premise", "hypothesis"]);
        let k2 = cache_key("nli", "model", &["hypothesis", "premise"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn merge_batch_all_cached() {
        let cached = vec![
            Some(Embedding {
                values: vec![1.0],
                model: "m".into(),
                dimensions: 1,
            }),
            Some(Embedding {
                values: vec![2.0],
                model: "m".into(),
                dimensions: 1,
            }),
        ];
        let result = merge_batch_results(cached, vec![]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].values, vec![1.0]);
        assert_eq!(result[1].values, vec![2.0]);
    }

    #[test]
    fn merge_batch_all_misses() {
        let cached = vec![None, None];
        let provider_results = vec![
            Embedding {
                values: vec![1.0],
                model: "m".into(),
                dimensions: 1,
            },
            Embedding {
                values: vec![2.0],
                model: "m".into(),
                dimensions: 1,
            },
        ];
        let result = merge_batch_results(cached, provider_results);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].values, vec![1.0]);
    }

    #[test]
    fn merge_batch_mixed() {
        let cached = vec![
            Some(Embedding {
                values: vec![1.0],
                model: "m".into(),
                dimensions: 1,
            }),
            None,
            Some(Embedding {
                values: vec![3.0],
                model: "m".into(),
                dimensions: 1,
            }),
            None,
        ];
        let provider_results = vec![
            Embedding {
                values: vec![2.0],
                model: "m".into(),
                dimensions: 1,
            },
            Embedding {
                values: vec![4.0],
                model: "m".into(),
                dimensions: 1,
            },
        ];
        let result = merge_batch_results(cached, provider_results);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].values, vec![1.0]); // cached
        assert_eq!(result[1].values, vec![2.0]); // from provider
        assert_eq!(result[2].values, vec![3.0]); // cached
        assert_eq!(result[3].values, vec![4.0]); // from provider
    }
}
