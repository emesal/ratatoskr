# Phase 7: Operational Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** make ratatoskr production-ready with retry logic, streaming backpressure, response caching for deterministic operations, advanced routing, and full observability via tracing spans and metrics.

**Architecture:** five sub-phases build on each other. telemetry lands first for visibility into everything that follows. retry uses the decorator pattern (`RetryingProvider<T>`) with global defaults and per-provider overrides; exhausted retries trigger cross-provider fallback. backpressure adds bounded channels to stream producers. response caching (embeddings + NLI only, opt-in) uses `moka` with LRU + TTL. routing wires up the existing `RoutingConfig` and adds latency tracking.

**Tech Stack:** rust, `metrics` crate (counters/histograms), `moka` (async cache), existing `tracing` + `tokio` deps. no new heavy frameworks.

---

## Design Decisions (Resolved)

| question | decision | rationale |
|----------|----------|-----------|
| retry: decorator vs inline | decorator (`RetryingProvider<T>`) | composable, matches `traits.rs` module doc intent, clean separation |
| retry: per-provider or global | global default + per-provider override | flexibility for mixed providers (e.g. ollama local = no retry, openrouter = aggressive) |
| exhausted retries → fallback | yes | rate-limited provider #1 should fall through to provider #2 after retry exhaustion |
| telemetry: metrics backend | `metrics` crate | purpose-built for counters/histograms, lower overhead than tracing events |
| caching scope | per-gateway, opt-in via builder | zero overhead for short-lived consumers (chibi); valuable for long-running ratd |
| response cache: which ops | embeddings + NLI only | deterministic, high hit rate. chat/generate not cached locally — provider-side caching instead |
| LLM response caching | none | near-zero cache hit rate for dynamic conversations; inherently flaky even at temperature 0. consumers cache at their layer; we facilitate provider-side caching via `cache_prompt` |
| cache backend | `moka` in-memory, per-gateway owned | simple, no coordination. extract `trait ResponseCache` later when shared/redis backends needed (see code comment) |

## Task Overview

```
task 1:  error classification — is_transient() on RatatoskrError                    [x]
task 2:  RetryConfig type + builder integration                                     [x]
task 3:  RetryingProvider<T> decorator for ChatProvider                              [x]
task 4:  RetryingProvider<T> for EmbeddingProvider, NliProvider, GenerateProvider    [x]
task 5:  cross-provider retry fallback in ProviderRegistry                          [x]
task 6:  telemetry — tracing spans with #[instrument] on dispatch paths             [ ]
task 7:  telemetry — metrics crate integration (counters, histograms)               [ ]
task 8:  streaming backpressure — bounded channels in chat_stream/generate_stream   [ ]
task 9:  response cache — CacheConfig + ResponseCache for embed/NLI                 [ ]
task 10: routing — wire up RoutingConfig + latency tracking                         [ ]
task 11: update AGENTS.md, ROADMAP.md, and docs                                    [ ]
```

Dependencies: task 1 → 2 → 3 → 4. task 5 depends on 3. tasks 6–7 are independent. task 8 is independent. task 9 depends on 7 (metrics for cache hit/miss). task 10 depends on 6–7 (latency data). task 11 is last.

Recommended execution order: 1 → 2 → 3 → 4 → 5, then 6 + 7 in parallel, then 8, then 9, then 10, then 11.

---

## Task 1: Error Classification — `is_transient()`

**Files:**
- Modify: `src/error.rs`
- Test: `tests/error_test.rs`

### Step 1: Write failing tests

```rust
// tests/error_test.rs
use std::time::Duration;
use ratatoskr::RatatoskrError;

#[test]
fn transient_errors() {
    assert!(RatatoskrError::RateLimited { retry_after: None }.is_transient());
    assert!(RatatoskrError::RateLimited { retry_after: Some(Duration::from_secs(1)) }.is_transient());
    assert!(RatatoskrError::Http("connection reset".into()).is_transient());
    assert!(RatatoskrError::Api { status: 500, message: "internal".into() }.is_transient());
    assert!(RatatoskrError::Api { status: 502, message: "bad gateway".into() }.is_transient());
    assert!(RatatoskrError::Api { status: 503, message: "unavailable".into() }.is_transient());
    assert!(RatatoskrError::Api { status: 504, message: "timeout".into() }.is_transient());
    assert!(RatatoskrError::Stream("connection reset".into()).is_transient());
    assert!(RatatoskrError::EmptyResponse.is_transient());
}

#[test]
fn permanent_errors() {
    assert!(!RatatoskrError::AuthenticationFailed.is_transient());
    assert!(!RatatoskrError::ModelNotFound("x".into()).is_transient());
    assert!(!RatatoskrError::InvalidInput("x".into()).is_transient());
    assert!(!RatatoskrError::NoProvider.is_transient());
    assert!(!RatatoskrError::Configuration("x".into()).is_transient());
    assert!(!RatatoskrError::ContentFiltered { reason: "x".into() }.is_transient());
    assert!(!RatatoskrError::ContextLengthExceeded { limit: 4096 }.is_transient());
    assert!(!RatatoskrError::Api { status: 400, message: "bad request".into() }.is_transient());
    assert!(!RatatoskrError::Api { status: 401, message: "unauth".into() }.is_transient());
    assert!(!RatatoskrError::Api { status: 403, message: "forbidden".into() }.is_transient());
    assert!(!RatatoskrError::Api { status: 404, message: "not found".into() }.is_transient());
    assert!(!RatatoskrError::UnsupportedParameter {
        param: "x".into(),
        model: "y".into(),
        provider: "z".into(),
    }.is_transient());
}

#[test]
fn llm_error_transient_heuristic() {
    // network-sounding errors are transient
    assert!(RatatoskrError::Llm("connection reset by peer".into()).is_transient());
    assert!(RatatoskrError::Llm("timeout".into()).is_transient());
    assert!(RatatoskrError::Llm("connection refused".into()).is_transient());
    // generic errors are not
    assert!(!RatatoskrError::Llm("invalid model format".into()).is_transient());
}
```

### Step 2: Implement `is_transient()`

Add to `src/error.rs`:

```rust
impl RatatoskrError {
    /// Whether this error is transient and the request may succeed on retry.
    ///
    /// Used by `RetryingProvider` to decide whether to retry a failed request.
    /// Permanent errors (auth, validation, model not found) return `false`.
    pub fn is_transient(&self) -> bool {
        match self {
            // Always transient
            Self::RateLimited { .. } => true,
            Self::Http(_) => true,
            Self::Stream(_) => true,
            Self::EmptyResponse => true,

            // Server errors are transient; client errors are not
            Self::Api { status, .. } => *status >= 500,

            // Heuristic: network-sounding messages are transient
            Self::Llm(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("timeout")
                    || lower.contains("connection")
                    || lower.contains("reset")
                    || lower.contains("refused")
            }

            // Everything else is permanent
            Self::Json(_)
            | Self::AuthenticationFailed
            | Self::ModelNotFound(_)
            | Self::InvalidInput(_)
            | Self::NoProvider
            | Self::ModelNotAvailable
            | Self::Configuration(_)
            | Self::NotImplemented(_)
            | Self::Unsupported
            | Self::UnsupportedParameter { .. }
            | Self::DataError(_)
            | Self::ContentFiltered { .. }
            | Self::ContextLengthExceeded { .. } => false,
        }
    }

    /// For `RateLimited` errors, the duration the provider suggests waiting.
    ///
    /// Returns `None` for non-rate-limit errors or when the provider didn't
    /// specify a retry-after duration.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }
}
```

### Step 3: Run tests

Run: `cargo test --test error_test`
Expected: all PASS

### Step 4: Run full suite

Run: `just pre-push`
Expected: all PASS, no clippy warnings

### Step 5: Commit

```bash
git add src/error.rs tests/error_test.rs
git commit -m "feat: add is_transient() and retry_after() to RatatoskrError

Classifies errors as transient (retryable) vs permanent. Server 5xx,
rate limits, network errors are transient; auth, validation, not-found
are permanent. Foundation for RetryingProvider decorator."
```

---

## Task 2: RetryConfig Type + Builder Integration

**Files:**
- Create: `src/providers/retry.rs`
- Modify: `src/providers/mod.rs`
- Modify: `src/gateway/builder.rs`
- Modify: `src/lib.rs`
- Test: `tests/retry_config_test.rs`

### Step 1: Write failing tests

```rust
// tests/retry_config_test.rs
use std::time::Duration;
use ratatoskr::RetryConfig;

#[test]
fn retry_config_defaults() {
    let config = RetryConfig::default();
    assert_eq!(config.max_attempts, 3);
    assert_eq!(config.initial_delay, Duration::from_millis(500));
    assert_eq!(config.max_delay, Duration::from_secs(30));
    assert!(config.jitter);
}

#[test]
fn retry_config_builder() {
    let config = RetryConfig::new()
        .max_attempts(5)
        .initial_delay(Duration::from_millis(100))
        .max_delay(Duration::from_secs(10))
        .jitter(false);

    assert_eq!(config.max_attempts, 5);
    assert_eq!(config.initial_delay, Duration::from_millis(100));
    assert_eq!(config.max_delay, Duration::from_secs(10));
    assert!(!config.jitter);
}

#[test]
fn retry_config_disabled() {
    let config = RetryConfig::disabled();
    assert_eq!(config.max_attempts, 1);
}

#[test]
fn retry_config_delay_calculation() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_millis(100))
        .max_delay(Duration::from_secs(10))
        .jitter(false);

    // Exponential backoff: 100ms, 200ms, 400ms, 800ms, ...
    assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
    assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
    assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
    assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
}

#[test]
fn retry_config_delay_capped_at_max() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_secs(1))
        .max_delay(Duration::from_secs(5))
        .jitter(false);

    // attempt 3 = 1 * 2^3 = 8s, but capped at 5s
    assert_eq!(config.delay_for_attempt(3), Duration::from_secs(5));
}

#[test]
fn retry_config_respects_retry_after() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_millis(100))
        .jitter(false);

    // retry_after from provider overrides calculated delay
    let delay = config.effective_delay(0, Some(Duration::from_secs(5)));
    assert_eq!(delay, Duration::from_secs(5));

    // without retry_after, uses calculated delay
    let delay = config.effective_delay(0, None);
    assert_eq!(delay, Duration::from_millis(100));
}
```

### Step 2: Implement RetryConfig

```rust
// src/providers/retry.rs
//! Retry configuration and delay calculation.
//!
//! Provides [`RetryConfig`] for controlling retry behaviour across providers.
//! Used by `RetryingProvider<T>` decorators (task 3) and the `ProviderRegistry`
//! cross-provider fallback (task 5).

use std::time::Duration;

/// Configuration for retry behaviour on transient errors.
///
/// Uses exponential backoff with optional jitter. The builder supports
/// both global defaults and per-provider overrides:
///
/// ```rust
/// # use ratatoskr::RetryConfig;
/// # use std::time::Duration;
/// let config = RetryConfig::new()
///     .max_attempts(5)
///     .initial_delay(Duration::from_millis(200))
///     .jitter(true);
/// ```
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the initial request).
    /// 1 = no retry. Default: 3.
    pub max_attempts: u32,
    /// Base delay before the first retry. Default: 500ms.
    pub initial_delay: Duration,
    /// Maximum delay between retries (caps exponential growth). Default: 30s.
    pub max_delay: Duration,
    /// Whether to add random jitter to delays. Default: true.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new config with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config that disables retries (single attempt).
    pub fn disabled() -> Self {
        Self {
            max_attempts: 1,
            ..Self::default()
        }
    }

    /// Set maximum attempts (including the initial request).
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    /// Set the base delay before the first retry.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay between retries.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Enable or disable jitter.
    pub fn jitter(mut self, enabled: bool) -> Self {
        self.jitter = enabled;
        self
    }

    /// Calculate the delay for a given attempt number (0-indexed).
    ///
    /// Uses exponential backoff: `initial_delay * 2^attempt`, capped at `max_delay`.
    /// Does NOT include jitter — see `effective_delay()` for the full calculation.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay = self.initial_delay.saturating_mul(2u32.saturating_pow(attempt));
        delay.min(self.max_delay)
    }

    /// Calculate the effective delay, respecting provider `retry_after` hints.
    ///
    /// If a `retry_after` duration is provided (from a `RateLimited` error),
    /// it takes precedence over the calculated backoff.
    pub fn effective_delay(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        retry_after.unwrap_or_else(|| self.delay_for_attempt(attempt))
    }
}
```

### Step 3: Wire up module + builder

Add to `src/providers/mod.rs`:
```rust
pub mod retry;
pub use retry::RetryConfig;
```

Add to `src/lib.rs` re-exports:
```rust
pub use providers::RetryConfig;
```

Add to `RatatoskrBuilder` in `src/gateway/builder.rs`:
- Field: `retry_config: RetryConfig`
- Builder method: `.retry(config: RetryConfig)`
- Default in `new()`: `RetryConfig::default()`

### Step 4: Run tests

Run: `cargo test --test retry_config_test`
Expected: all PASS

### Step 5: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 6: Commit

```bash
git add src/providers/retry.rs src/providers/mod.rs src/gateway/builder.rs src/lib.rs tests/retry_config_test.rs
git commit -m "feat: add RetryConfig with exponential backoff

Global default + per-provider override support. Respects provider
retry_after hints. Builder integration for gateway configuration."
```

---

## Task 3: RetryingProvider<T> Decorator for ChatProvider

**Files:**
- Modify: `src/providers/retry.rs`
- Modify: `Cargo.toml` (add `tokio/time` feature)
- Test: `tests/retry_test.rs`

### Step 1: Add `tokio/time` feature

In `Cargo.toml`, change:
```toml
tokio = { version = "1", features = ["rt", "sync"] }
```
to:
```toml
tokio = { version = "1", features = ["rt", "sync", "time"] }
```

### Step 2: Write failing tests

```rust
// tests/retry_test.rs
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ratatoskr::providers::retry::{RetryConfig, RetryingChatProvider};
use ratatoskr::providers::traits::ChatProvider;
use ratatoskr::{
    ChatEvent, ChatOptions, ChatResponse, Message, ModelMetadata, ParameterName,
    RatatoskrError, Result, ToolDefinition,
};

/// Mock provider that fails N times then succeeds.
struct FailThenSucceed {
    fail_count: AtomicU32,
    fail_with: fn() -> RatatoskrError,
    total_calls: AtomicU32,
}

impl FailThenSucceed {
    fn new(failures: u32, fail_with: fn() -> RatatoskrError) -> Self {
        Self {
            fail_count: AtomicU32::new(failures),
            fail_with,
            total_calls: AtomicU32::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        self.total_calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ChatProvider for FailThenSucceed {
    fn name(&self) -> &str { "mock-retry" }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        let remaining = self.fail_count.load(Ordering::Relaxed);
        if remaining > 0 {
            self.fail_count.fetch_sub(1, Ordering::Relaxed);
            return Err((self.fail_with)());
        }
        Ok(ChatResponse {
            content: "ok".into(),
            model: "test".into(),
            tool_calls: vec![],
            usage: None,
        })
    }

    async fn chat_stream(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<ChatEvent>> + Send>>> {
        Err(RatatoskrError::NotImplemented("stream"))
    }
}

#[tokio::test]
async fn retries_on_transient_error_then_succeeds() {
    let inner = Arc::new(FailThenSucceed::new(
        2,
        || RatatoskrError::RateLimited { retry_after: None },
    ));
    let provider = RetryingChatProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_ok());
    assert_eq!(inner.call_count(), 3); // 2 failures + 1 success
}

#[tokio::test]
async fn gives_up_after_max_attempts() {
    let inner = Arc::new(FailThenSucceed::new(
        10,
        || RatatoskrError::Http("timeout".into()),
    ));
    let provider = RetryingChatProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert_eq!(inner.call_count(), 3);
}

#[tokio::test]
async fn does_not_retry_permanent_errors() {
    let inner = Arc::new(FailThenSucceed::new(
        1,
        || RatatoskrError::AuthenticationFailed,
    ));
    let provider = RetryingChatProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(5)
            .initial_delay(Duration::from_millis(1)),
    );

    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert_eq!(inner.call_count(), 1); // no retry
}

#[tokio::test]
async fn respects_retry_after_duration() {
    let inner = Arc::new(FailThenSucceed::new(
        1,
        || RatatoskrError::RateLimited {
            retry_after: Some(Duration::from_millis(50)),
        },
    ));
    let provider = RetryingChatProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(2)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let start = std::time::Instant::now();
    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    // Should have waited at least 50ms (the retry_after), not 1ms (initial_delay)
    assert!(elapsed >= Duration::from_millis(40)); // some tolerance
}

#[tokio::test]
async fn disabled_config_no_retry() {
    let inner = Arc::new(FailThenSucceed::new(
        1,
        || RatatoskrError::RateLimited { retry_after: None },
    ));
    let provider = RetryingChatProvider::new(inner.clone(), RetryConfig::disabled());

    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert_eq!(inner.call_count(), 1);
}
```

### Step 3: Implement RetryingChatProvider

Add to `src/providers/retry.rs`:

```rust
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use tracing::warn;

use super::traits::ChatProvider;
use crate::{
    ChatEvent, ChatOptions, ChatResponse, Message, ModelMetadata, ParameterName,
    RatatoskrError, Result, ToolDefinition,
};

/// Decorator that wraps a `ChatProvider` with retry logic.
///
/// On transient errors (as classified by `RatatoskrError::is_transient()`),
/// retries with exponential backoff up to `config.max_attempts`. Respects
/// provider `retry_after` hints from `RateLimited` errors.
///
/// Non-transient errors and `ModelNotAvailable` are returned immediately
/// (the latter to preserve fallback chain semantics in `ProviderRegistry`).
pub struct RetryingChatProvider {
    inner: Arc<dyn ChatProvider>,
    config: RetryConfig,
}

impl RetryingChatProvider {
    /// Wrap a chat provider with retry logic.
    pub fn new(inner: Arc<dyn ChatProvider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

#[async_trait]
impl ChatProvider for RetryingChatProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let mut last_err = None;
        for attempt in 0..self.config.max_attempts {
            match self.inner.chat(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(e) if e.is_transient() => {
                    if attempt + 1 < self.config.max_attempts {
                        let delay = self.config.effective_delay(attempt, e.retry_after());
                        warn!(
                            provider = self.inner.name(),
                            attempt = attempt + 1,
                            max_attempts = self.config.max_attempts,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "retrying after transient error"
                        );
                        tokio::time::sleep(delay).await;
                    }
                    last_err = Some(e);
                }
                Err(e) => return Err(e), // permanent error, no retry
            }
        }
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        // Retry only the initial connection, not mid-stream failures.
        // Once the stream is established, errors are propagated to the consumer.
        let mut last_err = None;
        for attempt in 0..self.config.max_attempts {
            match self.inner.chat_stream(messages, tools, options).await {
                Ok(stream) => return Ok(stream),
                Err(e) if e.is_transient() => {
                    if attempt + 1 < self.config.max_attempts {
                        let delay = self.config.effective_delay(attempt, e.retry_after());
                        warn!(
                            provider = self.inner.name(),
                            attempt = attempt + 1,
                            max_attempts = self.config.max_attempts,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "retrying stream after transient error"
                        );
                        tokio::time::sleep(delay).await;
                    }
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    fn supported_chat_parameters(&self) -> Vec<ParameterName> {
        self.inner.supported_chat_parameters()
    }

    async fn fetch_metadata(&self, model: &str) -> Result<ModelMetadata> {
        // Metadata fetches are retried too — they're network calls
        let mut last_err = None;
        for attempt in 0..self.config.max_attempts {
            match self.inner.fetch_metadata(model).await {
                Ok(metadata) => return Ok(metadata),
                Err(e) if e.is_transient() => {
                    if attempt + 1 < self.config.max_attempts {
                        let delay = self.config.effective_delay(attempt, e.retry_after());
                        tokio::time::sleep(delay).await;
                    }
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }
}
```

### Step 4: Run tests

Run: `cargo test --test retry_test`
Expected: all PASS

### Step 5: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 6: Commit

```bash
git add src/providers/retry.rs Cargo.toml tests/retry_test.rs
git commit -m "feat: add RetryingChatProvider decorator

Wraps ChatProvider with exponential backoff retry on transient errors.
Respects retry_after hints, preserves ModelNotAvailable for fallback."
```

---

## Task 4: RetryingProvider Decorators for Other Provider Traits

**Files:**
- Modify: `src/providers/retry.rs`
- Test: `tests/retry_test.rs` (extend)

### Implementation

Add `RetryingEmbeddingProvider`, `RetryingNliProvider`, and `RetryingGenerateProvider` following the exact same pattern as `RetryingChatProvider`. Each wraps `Arc<dyn XxxProvider>` + `RetryConfig`, retries on `is_transient()`, and returns permanent errors immediately.

Use a private helper function to avoid code duplication in the retry loop:

```rust
/// Execute an async operation with retry logic.
///
/// Retries on transient errors up to `config.max_attempts`, using
/// exponential backoff and respecting `retry_after` hints.
async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    provider_name: &str,
    operation: &str,
    f: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_err = None;
    for attempt in 0..config.max_attempts {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if e.is_transient() => {
                if attempt + 1 < config.max_attempts {
                    let delay = config.effective_delay(attempt, e.retry_after());
                    warn!(
                        provider = provider_name,
                        operation,
                        attempt = attempt + 1,
                        max_attempts = config.max_attempts,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "retrying after transient error"
                    );
                    tokio::time::sleep(delay).await;
                }
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or(RatatoskrError::NoProvider))
}
```

Then `RetryingChatProvider` and the new decorators all delegate to `with_retry()`. This keeps the retry logic in one place (single source of truth) and the decorators thin.

### Tests

Add tests for `RetryingEmbeddingProvider` and `RetryingGenerateProvider` following the same pattern as the chat tests: mock provider that fails N times, verify retry count, verify permanent errors aren't retried.

### Commit

```bash
git commit -m "feat: add RetryingProvider decorators for embed, NLI, generate

All provider retry decorators delegate to shared with_retry() helper.
Single source of truth for retry logic across all capability types."
```

---

## Task 5: Cross-Provider Retry Fallback in ProviderRegistry

**Files:**
- Modify: `src/providers/registry.rs`
- Test: `tests/registry_retry_test.rs`

### Design

Currently, the fallback loop in `ProviderRegistry` only falls through on `ModelNotAvailable`. After retry exhaustion, the error is still the last transient error (e.g. `RateLimited`), which is terminal.

The change: when `RetryingProvider` returns a transient error (meaning all retry attempts were exhausted), the registry should treat it as a fallback trigger — same as `ModelNotAvailable`. The registry wraps each provider in `RetryingProvider` at registration time (using the global config, overridable per-provider).

**Approach:**
1. `ProviderRegistry` stores a `RetryConfig` (global default).
2. Registration methods (`add_chat`, etc.) wrap providers in `RetryingProvider<T>` automatically.
3. Fallback loop treats exhausted transient errors as fallback triggers:

```rust
for provider in &self.chat {
    match provider.chat(messages, tools, options).await {
        Ok(result) => return Ok(result),
        Err(RatatoskrError::ModelNotAvailable) => continue,
        Err(e) if e.is_transient() => continue,  // NEW: retry exhausted, try next provider
        Err(e) => return Err(e),
    }
}
```

The `RetryingProvider` already exhausted retries internally, so if the error is still transient at the registry level, it means all attempts for that provider failed → fall through.

### Tests

```rust
// tests/registry_retry_test.rs
// Test: provider #1 rate-limited (retries exhausted) → falls through to provider #2 which succeeds
// Test: all providers fail transiently → final error is the last transient error
// Test: provider #1 permanent error → NOT a fallback trigger (stops immediately)
```

### Commit

```bash
git commit -m "feat: exhausted retry triggers cross-provider fallback

ProviderRegistry now falls through to next provider when retries are
exhausted on transient errors, not just on ModelNotAvailable."
```

---

## Task 6: Telemetry — Tracing Spans

**Files:**
- Modify: `src/providers/registry.rs`
- Modify: `src/gateway/embedded.rs`
- Modify: `src/providers/llm_chat.rs`

### Design

Add `#[instrument]` spans to the hot paths:

1. **ProviderRegistry dispatch methods** — span per capability dispatch with `provider`, `model`, `operation` fields
2. **EmbeddedGateway trait methods** — top-level span per gateway call
3. **LlmChatProvider** — span around the llm crate call

No new dependencies — `tracing` is already in `Cargo.toml`.

### Example

```rust
use tracing::instrument;

#[instrument(skip(self, messages, tools), fields(model = %options.model, provider))]
pub async fn chat(
    &self,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    options: &ChatOptions,
) -> Result<ChatResponse> {
    // ...
}
```

### Tests

Tracing spans are observability — tested via integration tests that verify no panics with a `tracing_subscriber` installed. No assertion on span content (that's the consumer's job).

### Commit

```bash
git commit -m "feat: add tracing spans to dispatch and gateway paths

#[instrument] on ProviderRegistry dispatch, EmbeddedGateway methods,
and LlmChatProvider calls. Zero new dependencies."
```

---

## Task 7: Telemetry — Metrics Crate Integration

**Files:**
- Modify: `Cargo.toml` (add `metrics` dependency)
- Create: `src/telemetry.rs`
- Modify: `src/providers/registry.rs`
- Modify: `src/lib.rs`
- Test: `tests/metrics_test.rs`

### Design

Add the `metrics` crate and emit counters/histograms at the registry dispatch level:

```rust
// Counters
metrics::counter!("ratatoskr_requests_total", "provider" => name, "operation" => "chat", "status" => "ok").increment(1);
metrics::counter!("ratatoskr_requests_total", "provider" => name, "operation" => "chat", "status" => "error").increment(1);
metrics::counter!("ratatoskr_retries_total", "provider" => name, "operation" => "chat").increment(1);

// Histograms
metrics::histogram!("ratatoskr_request_duration_seconds", "provider" => name, "operation" => "chat").record(elapsed.as_secs_f64());

// Token counters (from ChatResponse.usage when available)
metrics::counter!("ratatoskr_tokens_total", "provider" => name, "direction" => "prompt").increment(prompt_tokens);
metrics::counter!("ratatoskr_tokens_total", "provider" => name, "direction" => "completion").increment(completion_tokens);
```

The `metrics` crate is recorder-agnostic — consumers install their own recorder (prometheus, statsd, etc.). If no recorder is installed, all calls are no-ops.

### Relationship to open issues

- **#12 (runtime parameter telemetry & discovery)** — this task builds the local telemetry foundation that #12's feedback loop will consume. the metric signals emitted here (request counts by status, `UnsupportedParameter` errors) are exactly the raw data #13 needs for local runtime discovery and #14 needs for centralised collection. designing metric names and labels with this future consumption in mind.
- **#14 (centralised telemetry collection)** — #14 will add a `metrics` recorder that ships anonymised data to an aggregation endpoint. the metric names defined in `src/telemetry.rs` below will be the exact signals #14 collects. no special accommodation needed — the recorder-agnostic design handles this naturally.

### Metric names module

Create `src/telemetry.rs` with metric name constants to avoid string duplication:

```rust
//! Telemetry metric name constants.
//!
//! Centralised metric names for ratatoskr operations. Consumers install
//! their own `metrics` recorder (e.g. prometheus, statsd); without a
//! recorder installed, all metric calls are no-ops.

pub const REQUESTS_TOTAL: &str = "ratatoskr_requests_total";
pub const REQUEST_DURATION_SECONDS: &str = "ratatoskr_request_duration_seconds";
pub const RETRIES_TOTAL: &str = "ratatoskr_retries_total";
pub const TOKENS_TOTAL: &str = "ratatoskr_tokens_total";
pub const CACHE_HITS_TOTAL: &str = "ratatoskr_cache_hits_total";
pub const CACHE_MISSES_TOTAL: &str = "ratatoskr_cache_misses_total";
```

### Tests

Verify metrics compile and don't panic without a recorder (the crate's design guarantee). Use `metrics::with_local_recorder` in tests to capture and assert on emitted metrics.

### Commit

```bash
git commit -m "feat: add metrics crate integration for request telemetry

Counters for requests, retries, tokens. Histograms for duration.
Recorder-agnostic — consumers install their own backend."
```

---

## Task 8: Streaming Backpressure

**Files:**
- Modify: `src/providers/registry.rs` (or create `src/providers/backpressure.rs`)
- Test: `tests/backpressure_test.rs`

### Design

Wrap stream output in `chat_stream` and `generate_stream` with a bounded `tokio::sync::mpsc::channel`. The producer task reads from the inner stream and sends to the channel; the consumer receives from the channel. When the channel is full, the producer blocks (backpressure).

```rust
const DEFAULT_STREAM_BUFFER: usize = 64;

fn bounded_stream<T: Send + 'static>(
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

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}
```

### Dependencies

Add `tokio-stream` to `Cargo.toml` for `ReceiverStream`.

### Tests

Test with a slow consumer (delayed reads) and verify the producer doesn't run unbounded ahead.

### Commit

```bash
git commit -m "feat: add bounded channel backpressure to streaming

Wraps chat_stream/generate_stream output in bounded mpsc channel.
Producer blocks when consumer falls behind (default buffer: 64)."
```

---

## Task 9: Response Cache for Embeddings + NLI

**Files:**
- Create: `src/cache/response.rs`
- Modify: `src/cache/mod.rs`
- Modify: `src/gateway/builder.rs`
- Modify: `src/gateway/embedded.rs`
- Modify: `src/providers/registry.rs`
- Modify: `Cargo.toml` (add `moka`)
- Test: `tests/response_cache_test.rs`

### Design

Opt-in response cache for deterministic operations. Only activated when the consumer calls `.response_cache(config)` on the builder. Uses `moka::future::Cache` for async-friendly LRU + TTL.

```rust
/// Configuration for the response cache.
///
/// Enables caching of deterministic responses (embeddings, NLI).
/// Without this, no cache is allocated (zero overhead for short-lived
/// consumers like chibi).
///
/// # Future extensibility
///
/// The current implementation uses moka's in-memory LRU cache, owned
/// per-gateway. When shared caching is needed (e.g. redis-backed for
/// ratd clusters), extract a `trait ResponseCache: Send + Sync` and
/// inject implementations via the builder. The key design (content hash)
/// is backend-agnostic, so the refactor is straightforward:
///
/// ```rust,ignore
/// trait ResponseCache: Send + Sync {
///     async fn get(&self, key: &CacheKey) -> Option<CachedResponse>;
///     async fn insert(&self, key: CacheKey, value: CachedResponse);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached entries. Default: 10_000.
    pub max_entries: u64,
    /// Time-to-live for cached entries. Default: 1 hour.
    pub ttl: Duration,
}
```

Cache key:
```rust
/// Cache key — hash of (operation, model, input).
///
/// For embeddings: hash(model, text)
/// For NLI: hash(model, premise, hypothesis)
fn cache_key(operation: &str, model: &str, input: &[&str]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    operation.hash(&mut hasher);
    model.hash(&mut hasher);
    for s in input {
        s.hash(&mut hasher);
    }
    hasher.finish()
}
```

### Builder integration

```rust
Ratatoskr::builder()
    .openrouter(key)
    .response_cache(CacheConfig { max_entries: 10_000, ttl: Duration::from_secs(3600) })
    .build()?
```

No `.response_cache()` → no cache allocated → zero overhead.

### Metrics integration

Emit `ratatoskr_cache_hits_total` and `ratatoskr_cache_misses_total` counters (from task 7).

### Tests

- Cache hit on repeated embedding request
- Cache miss on different input
- TTL expiry
- No cache when `CacheConfig` not set

### Commit

```bash
git commit -m "feat: add opt-in response cache for embeddings and NLI

moka-backed LRU + TTL cache, activated via builder. Zero overhead
when not configured. Emits cache hit/miss metrics."
```

---

## Task 10: Routing — Wire Up RoutingConfig + Latency Tracking

**Files:**
- Modify: `src/providers/registry.rs`
- Modify: `src/server/config.rs`
- Modify: `src/bin/ratd.rs`
- Create: `src/providers/routing.rs`
- Test: `tests/routing_test.rs`

### Design

1. **Wire up existing `RoutingConfig`** — `ratd.rs` already parses `[routing]` from TOML but doesn't use it. Wire preferred-provider fields into `ProviderRegistry` construction so that the preferred provider is moved to position 0 in the fallback chain.

2. **Latency tracking** — record per-provider request durations (already emitted via metrics in task 7). Expose a `ProviderLatency` struct that tracks a rolling average, accessible for future latency-based routing.

3. **Cost-aware routing** (foundation) — add a `route_by_cost(model)` method that consults `ModelMetadata.pricing` from the registry to suggest the cheapest provider for a model. This is informational for now — consumers can use it, but automatic routing is deferred.

### Note

Full automatic latency-based and cost-based routing is deferred to a future phase. Task 10 lays the foundation (tracking, wiring) without over-engineering.

### Commit

```bash
git commit -m "feat: wire RoutingConfig and add latency tracking

Preferred-provider config now affects fallback chain order.
Per-provider latency tracking via rolling average.
Cost-aware routing method using ModelMetadata pricing."
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/ROADMAP.md`

### Changes

1. **AGENTS.md** — add phase 7 types and concepts:
   - `RetryConfig` — retry configuration with exponential backoff
   - `RetryingProvider<T>` — decorator pattern for retry logic
   - `CacheConfig` — opt-in response cache configuration
   - `ResponseCache` — embedding/NLI response cache (moka-backed)
   - Telemetry section: metric names, tracing spans
   - Update builder pattern example with `.retry()` and `.response_cache()`
   - Update project structure for new files

2. **ROADMAP.md** — mark phase 7 complete, add link to this plan doc.

### Commit

```bash
git commit -m "docs: update AGENTS.md and ROADMAP.md for phase 7"
```
