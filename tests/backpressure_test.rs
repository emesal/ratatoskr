//! Tests for streaming backpressure.
//!
//! Verifies that the bounded channel wrapper correctly applies backpressure
//! to stream producers when consumers fall behind.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures_util::stream::{self, Stream, StreamExt};
use ratatoskr::providers::backpressure::bounded_stream;
use ratatoskr::{ChatEvent, Result};

/// Create a stream that counts how many items have been produced.
fn counting_stream(
    count: u32,
    produced: Arc<AtomicU32>,
) -> Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>> {
    let s = stream::iter((0..count).map(move |i| {
        produced.fetch_add(1, Ordering::SeqCst);
        Ok(ChatEvent::Content(format!("chunk-{i}")))
    }));
    Box::pin(s)
}

#[tokio::test]
async fn bounded_stream_delivers_all_items() {
    let produced = Arc::new(AtomicU32::new(0));
    let inner = counting_stream(10, produced.clone());
    let mut stream = bounded_stream(inner, 4);

    let mut received = 0;
    while let Some(item) = stream.next().await {
        assert!(item.is_ok());
        received += 1;
    }
    assert_eq!(received, 10);
    assert_eq!(produced.load(Ordering::SeqCst), 10);
}

#[tokio::test]
async fn bounded_stream_propagates_errors() {
    let inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>> =
        Box::pin(stream::iter(vec![
            Ok(ChatEvent::Content("ok".into())),
            Err(ratatoskr::RatatoskrError::Stream("boom".into())),
            Ok(ChatEvent::Done),
        ]));

    let mut stream = bounded_stream(inner, 4);

    let first = stream.next().await.unwrap();
    assert!(first.is_ok());

    let second = stream.next().await.unwrap();
    assert!(second.is_err());

    let third = stream.next().await.unwrap();
    assert!(third.is_ok());
}

#[tokio::test]
async fn bounded_stream_handles_empty_stream() {
    let inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>> = Box::pin(stream::empty());

    let mut stream = bounded_stream(inner, 4);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn producer_stops_when_consumer_drops() {
    let produced = Arc::new(AtomicU32::new(0));

    // Create a stream of 1000 items but only consume 2
    let inner = counting_stream(1000, produced.clone());
    let mut stream = bounded_stream(inner, 4);

    // Consume just 2 items
    stream.next().await;
    stream.next().await;

    // Drop the stream (consumer side)
    drop(stream);

    // Give the producer task a moment to notice the dropped receiver
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Producer should have stopped well before 1000
    // (at most buffer_size + a few items ahead)
    let total = produced.load(Ordering::SeqCst);
    assert!(
        total < 20,
        "producer should stop early when consumer drops, but produced {total} items"
    );
}

#[tokio::test]
async fn backpressure_limits_producer_ahead() {
    // Slow consumer: read one item, then pause. The producer shouldn't
    // run unbounded ahead — it should be limited by the buffer size.
    let produced = Arc::new(AtomicU32::new(0));
    let produced_clone = produced.clone();

    // Create a slow-producing stream (each item is instant, but there are many)
    let inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>> =
        Box::pin(stream::iter((0..100).map(move |i| {
            produced_clone.fetch_add(1, Ordering::SeqCst);
            Ok(ChatEvent::Content(format!("chunk-{i}")))
        })));

    let buffer_size = 4;
    let mut stream = bounded_stream(inner, buffer_size);

    // Read one item to kick things off
    let _ = stream.next().await;

    // Give the producer a moment to fill the buffer
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Producer should be limited: consumed 1 + buffer capacity ahead
    // Allow some tolerance for timing
    let total = produced.load(Ordering::SeqCst);
    assert!(
        total <= (buffer_size as u32 + 2),
        "producer should be bounded by buffer, but produced {total} items (buffer={buffer_size})"
    );
}
