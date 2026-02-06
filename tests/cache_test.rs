//! Tests for [`ModelCache`] — ephemeral model metadata storage.

use ratatoskr::cache::ModelCache;
use ratatoskr::{ModelInfo, ModelMetadata};

fn make_metadata(id: &str) -> ModelMetadata {
    ModelMetadata::from_info(ModelInfo::new(id, "test"))
}

#[test]
fn cache_miss_returns_none() {
    let cache = ModelCache::new();
    assert!(cache.get("nonexistent").is_none());
}

#[test]
fn insert_then_get() {
    let cache = ModelCache::new();
    let m = make_metadata("test-model");
    cache.insert(m);

    let got = cache.get("test-model");
    assert!(got.is_some());
    assert_eq!(got.unwrap().info.id, "test-model");
}

#[test]
fn overwrite_replaces_entry() {
    let cache = ModelCache::new();

    let m1 = make_metadata("model-a").with_max_output_tokens(100);
    cache.insert(m1);

    let m2 = make_metadata("model-a").with_max_output_tokens(999);
    cache.insert(m2);

    let got = cache.get("model-a").unwrap();
    assert_eq!(got.max_output_tokens, Some(999));
}

#[test]
fn independent_keys() {
    let cache = ModelCache::new();
    cache.insert(make_metadata("alpha"));
    cache.insert(make_metadata("beta"));

    assert!(cache.get("alpha").is_some());
    assert!(cache.get("beta").is_some());
    assert!(cache.get("gamma").is_none());
}

#[test]
fn thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let cache = Arc::new(ModelCache::new());
    let mut handles = Vec::new();

    // Spawn writers
    for i in 0..10 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            cache.insert(make_metadata(&format!("model-{i}")));
        }));
    }

    // Spawn concurrent readers
    for i in 0..10 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            // May or may not see the entry yet — shouldn't panic
            let _ = cache.get(&format!("model-{i}"));
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // After all writers finish, all entries should be present
    for i in 0..10 {
        assert!(cache.get(&format!("model-{i}")).is_some());
    }
}

#[test]
fn default_creates_empty_cache() {
    let cache = ModelCache::default();
    assert!(cache.get("anything").is_none());
}
