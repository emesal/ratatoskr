//! Streaming backpressure via bounded channels.
//!
//! Wraps stream output in a bounded `tokio::sync::mpsc::channel` so that
//! producers block when consumers fall behind. Without this, a fast provider
//! can fill unbounded memory if the consumer is slow (e.g. rate-limited
//! client, busy UI thread).
//!
//! # Usage
//!
//! Applied automatically by [`ProviderRegistry`](super::ProviderRegistry)
//! to `chat_stream` and `generate_stream` results. The buffer size defaults
//! to [`DEFAULT_STREAM_BUFFER`] and can be overridden via the builder.

use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use tokio_stream::wrappers::ReceiverStream;

use crate::Result;

/// Default number of items buffered between producer and consumer.
///
/// 64 balances throughput (enough items to keep the consumer busy)
/// with memory pressure (bounded, not unbounded).
pub const DEFAULT_STREAM_BUFFER: usize = 64;

/// Wrap a stream in a bounded channel for backpressure.
///
/// Spawns a producer task that reads from `inner` and sends items
/// through a bounded `mpsc` channel. When the channel is full, the
/// producer blocks until the consumer reads. If the consumer drops
/// the stream, the producer stops.
///
/// # Panics
///
/// Requires a tokio runtime context (called within an async fn).
pub fn bounded_stream<T: Send + 'static>(
    inner: Pin<Box<dyn Stream<Item = Result<T>> + Send>>,
    buffer_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<T>> + Send>> {
    let (tx, rx) = tokio::sync::mpsc::channel(buffer_size);

    tokio::spawn(async move {
        let mut inner = inner;
        while let Some(item) = inner.next().await {
            if tx.send(item).await.is_err() {
                break; // receiver dropped
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}
