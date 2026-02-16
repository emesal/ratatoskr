//! Tests for metrics integration (phase 7, task 7).
//!
//! Uses `metrics_util::debugging::DebuggingRecorder` to capture and assert
//! on emitted metrics without needing a real exporter.

use std::sync::Arc;

use async_trait::async_trait;
use metrics_util::MetricKind;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

use ratatoskr::providers::registry::ProviderRegistry;
use ratatoskr::providers::traits::EmbeddingProvider;
use ratatoskr::telemetry;
use ratatoskr::types::Embedding;
use ratatoskr::{RatatoskrError, Result};

// ============================================================================
// Mock providers
// ============================================================================

struct MockEmbeddingProvider {
    name: &'static str,
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn embed(&self, _text: &str, _model: &str) -> Result<Embedding> {
        Ok(Embedding {
            values: vec![0.1, 0.2],
            model: "test-model".to_string(),
            dimensions: 2,
        })
    }
}

struct FailingEmbeddingProvider;

#[async_trait]
impl EmbeddingProvider for FailingEmbeddingProvider {
    fn name(&self) -> &str {
        "failing"
    }

    async fn embed(&self, _text: &str, _model: &str) -> Result<Embedding> {
        Err(RatatoskrError::AuthenticationFailed)
    }
}

// ============================================================================
// Snapshot type alias for readability
// ============================================================================

type SnapshotVec = Vec<(
    metrics_util::CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
)>;

// ============================================================================
// Helpers
// ============================================================================

/// Sum all counter values matching a given metric name.
fn counter_total(snapshot: &SnapshotVec, name: &str) -> u64 {
    snapshot
        .iter()
        .filter(|(key, _, _, _)| key.kind() == MetricKind::Counter && key.key().name() == name)
        .map(|(_, _, _, value)| match value {
            DebugValue::Counter(v) => *v,
            _ => 0,
        })
        .sum()
}

/// Check if any histogram entries exist for a given metric name.
fn has_histogram(snapshot: &SnapshotVec, name: &str) -> bool {
    snapshot
        .iter()
        .any(|(key, _, _, _)| key.kind() == MetricKind::Histogram && key.key().name() == name)
}

// ============================================================================
// Tests
// ============================================================================

/// Runs async code within a local recorder scope on the multi-thread runtime.
///
/// `block_in_place` ensures the sync `with_local_recorder` closure stays
/// on the current thread while `block_on` drives the inner async work.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn successful_request_records_metrics() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let result = metrics::with_local_recorder(&recorder, || {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut registry = ProviderRegistry::new();
                registry.add_embedding(Arc::new(MockEmbeddingProvider { name: "test-embed" }));
                registry.embed("hello", "test-model", None).await
            })
        })
    });
    assert!(result.is_ok());

    let snapshot = snapshotter.snapshot().into_vec();

    let count = counter_total(&snapshot, telemetry::REQUESTS_TOTAL);
    assert_eq!(count, 1, "expected 1 request counter");

    assert!(
        has_histogram(&snapshot, telemetry::REQUEST_DURATION_SECONDS),
        "expected a duration histogram entry"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn failed_request_records_error_metrics() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let _result = metrics::with_local_recorder(&recorder, || {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut registry = ProviderRegistry::new();
                registry.add_embedding(Arc::new(FailingEmbeddingProvider));
                registry.embed("hello", "test-model", None).await
            })
        })
    });

    let snapshot = snapshotter.snapshot().into_vec();

    let count = counter_total(&snapshot, telemetry::REQUESTS_TOTAL);
    assert_eq!(count, 1, "expected 1 request counter for error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn no_provider_records_error_metrics() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let _result = metrics::with_local_recorder(&recorder, || {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let registry = ProviderRegistry::new();
                registry.embed("hello", "test-model", None).await
            })
        })
    });

    let snapshot = snapshotter.snapshot().into_vec();

    let count = counter_total(&snapshot, telemetry::REQUESTS_TOTAL);
    assert_eq!(count, 1);
}

#[tokio::test]
async fn metrics_are_noop_without_recorder() {
    // Verify no panics when no recorder is installed.
    let mut registry = ProviderRegistry::new();
    registry.add_embedding(Arc::new(MockEmbeddingProvider { name: "test" }));
    let _result = registry.embed("hello", "test-model", None).await.unwrap();
}
