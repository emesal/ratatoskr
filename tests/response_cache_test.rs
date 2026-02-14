//! Tests for [`ResponseCache`] — opt-in LRU + TTL cache for embed/NLI.

use std::time::Duration;

use ratatoskr::cache::{CacheConfig, ResponseCache};
use ratatoskr::types::{Embedding, NliLabel, NliResult};

fn make_embedding(text: &str) -> Embedding {
    Embedding {
        values: vec![text.len() as f32; 3],
        model: "test-model".into(),
        dimensions: 3,
    }
}

fn make_nli_result() -> NliResult {
    NliResult {
        entailment: 0.8,
        contradiction: 0.1,
        neutral: 0.1,
        label: NliLabel::Entailment,
    }
}

// =========================================================================
// CacheConfig
// =========================================================================

#[test]
fn cache_config_defaults() {
    let config = CacheConfig::default();
    assert_eq!(config.max_entries, 10_000);
    assert_eq!(config.ttl, Duration::from_secs(3600));
}

#[test]
fn cache_config_builder() {
    let config = CacheConfig::new()
        .max_entries(500)
        .ttl(Duration::from_secs(60));
    assert_eq!(config.max_entries, 500);
    assert_eq!(config.ttl, Duration::from_secs(60));
}

// =========================================================================
// Embedding caching
// =========================================================================

#[tokio::test]
async fn embed_cache_miss_then_hit() {
    let cache = ResponseCache::new(&CacheConfig::default());

    // Miss
    assert!(cache.get_embedding("model-a", "hello").await.is_none());

    // Insert
    let embedding = make_embedding("hello");
    cache
        .insert_embedding("model-a", "hello", embedding.clone())
        .await;

    // Hit
    let cached = cache.get_embedding("model-a", "hello").await;
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().values, embedding.values);
}

#[tokio::test]
async fn embed_cache_different_text_is_miss() {
    let cache = ResponseCache::new(&CacheConfig::default());

    cache
        .insert_embedding("model-a", "hello", make_embedding("hello"))
        .await;

    assert!(cache.get_embedding("model-a", "world").await.is_none());
}

#[tokio::test]
async fn embed_cache_different_model_is_miss() {
    let cache = ResponseCache::new(&CacheConfig::default());

    cache
        .insert_embedding("model-a", "hello", make_embedding("hello"))
        .await;

    assert!(cache.get_embedding("model-b", "hello").await.is_none());
}

#[tokio::test]
async fn embed_cache_ttl_expiry() {
    let config = CacheConfig::new().ttl(Duration::from_millis(50));
    let cache = ResponseCache::new(&config);

    cache
        .insert_embedding("model", "text", make_embedding("text"))
        .await;

    // Should be present immediately
    assert!(cache.get_embedding("model", "text").await.is_some());

    // Wait for TTL + some margin
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Should be expired
    assert!(cache.get_embedding("model", "text").await.is_none());
}

// =========================================================================
// NLI caching
// =========================================================================

#[tokio::test]
async fn nli_cache_miss_then_hit() {
    let cache = ResponseCache::new(&CacheConfig::default());

    assert!(
        cache
            .get_nli("model", "premise", "hypothesis")
            .await
            .is_none()
    );

    let result = make_nli_result();
    cache
        .insert_nli("model", "premise", "hypothesis", result.clone())
        .await;

    let cached = cache.get_nli("model", "premise", "hypothesis").await;
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().label, NliLabel::Entailment);
}

#[tokio::test]
async fn nli_cache_order_matters() {
    let cache = ResponseCache::new(&CacheConfig::default());

    cache.insert_nli("model", "A", "B", make_nli_result()).await;

    // (A, B) cached but (B, A) should miss — order is semantic
    assert!(cache.get_nli("model", "A", "B").await.is_some());
    assert!(cache.get_nli("model", "B", "A").await.is_none());
}

#[tokio::test]
async fn nli_cache_ttl_expiry() {
    let config = CacheConfig::new().ttl(Duration::from_millis(50));
    let cache = ResponseCache::new(&config);

    cache.insert_nli("model", "p", "h", make_nli_result()).await;

    assert!(cache.get_nli("model", "p", "h").await.is_some());

    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(cache.get_nli("model", "p", "h").await.is_none());
}

// =========================================================================
// Batch decomposition
// =========================================================================

#[tokio::test]
async fn batch_cache_all_misses() {
    let cache = ResponseCache::new(&CacheConfig::default());

    let results = cache.get_embedding_batch("model", &["a", "b", "c"]).await;
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.is_none()));
}

#[tokio::test]
async fn batch_cache_partial_hits() {
    let cache = ResponseCache::new(&CacheConfig::default());

    // Pre-populate "hello" and "world"
    cache
        .insert_embedding("model", "hello", make_embedding("hello"))
        .await;
    cache
        .insert_embedding("model", "world", make_embedding("world"))
        .await;

    let results = cache
        .get_embedding_batch("model", &["hello", "new", "world"])
        .await;

    assert!(results[0].is_some()); // "hello" hit
    assert!(results[1].is_none()); // "new" miss
    assert!(results[2].is_some()); // "world" hit
}

#[tokio::test]
async fn batch_insert_then_individual_hit() {
    let cache = ResponseCache::new(&CacheConfig::default());

    let texts = ["alpha", "beta"];
    let embeddings = [make_embedding("alpha"), make_embedding("beta")];
    cache
        .insert_embedding_batch("model", &texts, &embeddings)
        .await;

    // Individual lookups should hit
    assert!(cache.get_embedding("model", "alpha").await.is_some());
    assert!(cache.get_embedding("model", "beta").await.is_some());
    assert!(cache.get_embedding("model", "gamma").await.is_none());
}

// =========================================================================
// Builder integration (compilation tests)
// =========================================================================

#[test]
fn builder_with_response_cache_compiles() {
    let gateway = ratatoskr::Ratatoskr::builder()
        .openrouter(Some("fake-key"))
        .response_cache(
            CacheConfig::new()
                .max_entries(100)
                .ttl(Duration::from_secs(60)),
        )
        .build();

    assert!(gateway.is_ok());
}

#[test]
fn builder_without_response_cache_compiles() {
    let gateway = ratatoskr::Ratatoskr::builder()
        .openrouter(Some("fake-key"))
        .build();

    assert!(gateway.is_ok());
}

// =========================================================================
// Metrics (no-op without recorder — just verify no panics)
// =========================================================================

#[tokio::test]
async fn metrics_emitted_without_panic() {
    // Without a metrics recorder installed, all metric calls should be no-ops
    let cache = ResponseCache::new(&CacheConfig::default());

    // Miss should emit cache_misses_total
    cache.get_embedding("model", "text").await;

    // Insert + hit should emit cache_hits_total
    cache
        .insert_embedding("model", "text", make_embedding("text"))
        .await;
    cache.get_embedding("model", "text").await;

    // NLI miss + insert + hit
    cache.get_nli("model", "p", "h").await;
    cache.insert_nli("model", "p", "h", make_nli_result()).await;
    cache.get_nli("model", "p", "h").await;
}

/// Runs async cache operations within a local recorder scope.
///
/// Uses `block_in_place` + `block_on` pattern to keep `with_local_recorder`
/// on the same thread (it's a thread-local recorder).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn metrics_with_recorder() {
    use metrics_util::MetricKind;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    metrics::with_local_recorder(&recorder, || {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let cache = ResponseCache::new(&CacheConfig::default());

                // Miss
                cache.get_embedding("model", "text").await;

                // Insert + hit
                cache
                    .insert_embedding("model", "text", make_embedding("text"))
                    .await;
                cache.get_embedding("model", "text").await;
            })
        })
    });

    let snapshot = snapshotter.snapshot().into_vec();

    let miss_count: u64 = snapshot
        .iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Counter && key.key().name() == "ratatoskr_cache_misses_total"
        })
        .map(|(_, _, _, val)| match val {
            DebugValue::Counter(c) => *c,
            _ => 0,
        })
        .sum();

    let hit_count: u64 = snapshot
        .iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Counter && key.key().name() == "ratatoskr_cache_hits_total"
        })
        .map(|(_, _, _, val)| match val {
            DebugValue::Counter(c) => *c,
            _ => 0,
        })
        .sum();

    assert_eq!(miss_count, 1, "expected 1 cache miss");
    assert_eq!(hit_count, 1, "expected 1 cache hit");
}
