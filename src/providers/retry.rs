//! Retry configuration, delay calculation, and provider decorators.
//!
//! Provides [`RetryConfig`] for controlling retry behaviour and
//! `Retrying*Provider` decorators that wrap provider traits with
//! automatic retry on transient errors.
//!
//! All decorators delegate to the shared `with_retry()` helper,
//! keeping retry logic in a single place.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::Stream;
use tracing::warn;

use crate::telemetry;

use super::traits::{ChatProvider, EmbeddingProvider, GenerateProvider, NliProvider};
use crate::types::{
    ChatEvent, ChatOptions, ChatResponse, Embedding, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, ModelMetadata, NliResult, ParameterName, ToolDefinition,
};
use crate::{RatatoskrError, Result};

/// Configuration for retry behaviour on transient errors.
///
/// Uses exponential backoff with optional jitter. Supports both global
/// defaults and per-provider overrides via the builder:
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
    /// Does NOT include jitter — see [`effective_delay()`](Self::effective_delay)
    /// for the full calculation.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay = self
            .initial_delay
            .saturating_mul(2u32.saturating_pow(attempt));
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

// ============================================================================
// Shared retry helper
// ============================================================================

/// Execute an async operation with retry logic.
///
/// Retries on transient errors (as classified by [`RatatoskrError::is_transient()`])
/// up to `config.max_attempts`, using exponential backoff and respecting
/// `retry_after` hints from `RateLimited` errors.
///
/// Permanent errors are returned immediately without retry.
pub(crate) async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    provider_name: &str,
    operation: &str,
    f: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_err = None;
    for attempt in 0..config.max_attempts {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if e.is_transient() => {
                metrics::counter!(telemetry::RETRIES_TOTAL,
                    "provider" => provider_name.to_owned(),
                    "operation" => operation.to_owned(),
                )
                .increment(1);
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
            Err(e) => return Err(e), // permanent error, no retry
        }
    }
    Err(last_err.unwrap_or(RatatoskrError::NoProvider))
}

// ============================================================================
// RetryingChatProvider
// ============================================================================

/// Decorator that wraps a [`ChatProvider`] with retry logic.
///
/// On transient errors (as classified by [`RatatoskrError::is_transient()`]),
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
        with_retry(&self.config, self.inner.name(), "chat", || {
            self.inner.chat(messages, tools, options)
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        // Retry only the initial connection, not mid-stream failures.
        with_retry(&self.config, self.inner.name(), "chat_stream", || {
            self.inner.chat_stream(messages, tools, options)
        })
        .await
    }

    fn supported_chat_parameters(&self) -> Vec<ParameterName> {
        self.inner.supported_chat_parameters()
    }

    async fn fetch_metadata(&self, model: &str) -> Result<ModelMetadata> {
        with_retry(&self.config, self.inner.name(), "fetch_metadata", || {
            self.inner.fetch_metadata(model)
        })
        .await
    }
}

// ============================================================================
// RetryingEmbeddingProvider
// ============================================================================

/// Decorator that wraps an [`EmbeddingProvider`] with retry logic.
///
/// Same semantics as [`RetryingChatProvider`] — retries transient errors,
/// returns permanent errors immediately.
pub struct RetryingEmbeddingProvider {
    inner: Arc<dyn EmbeddingProvider>,
    config: RetryConfig,
}

impl RetryingEmbeddingProvider {
    /// Wrap an embedding provider with retry logic.
    pub fn new(inner: Arc<dyn EmbeddingProvider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

#[async_trait]
impl EmbeddingProvider for RetryingEmbeddingProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        with_retry(&self.config, self.inner.name(), "embed", || {
            self.inner.embed(text, model)
        })
        .await
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        with_retry(&self.config, self.inner.name(), "embed_batch", || {
            self.inner.embed_batch(texts, model)
        })
        .await
    }
}

// ============================================================================
// RetryingNliProvider
// ============================================================================

/// Decorator that wraps an [`NliProvider`] with retry logic.
///
/// Same semantics as [`RetryingChatProvider`] — retries transient errors,
/// returns permanent errors immediately.
pub struct RetryingNliProvider {
    inner: Arc<dyn NliProvider>,
    config: RetryConfig,
}

impl RetryingNliProvider {
    /// Wrap an NLI provider with retry logic.
    pub fn new(inner: Arc<dyn NliProvider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

#[async_trait]
impl NliProvider for RetryingNliProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        with_retry(&self.config, self.inner.name(), "infer_nli", || {
            self.inner.infer_nli(premise, hypothesis, model)
        })
        .await
    }

    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        with_retry(&self.config, self.inner.name(), "infer_nli_batch", || {
            self.inner.infer_nli_batch(pairs, model)
        })
        .await
    }
}

// ============================================================================
// RetryingGenerateProvider
// ============================================================================

/// Decorator that wraps a [`GenerateProvider`] with retry logic.
///
/// Same semantics as [`RetryingChatProvider`] — retries transient errors,
/// returns permanent errors immediately. Stream retry covers only the
/// initial connection, not mid-stream failures.
pub struct RetryingGenerateProvider {
    inner: Arc<dyn GenerateProvider>,
    config: RetryConfig,
}

impl RetryingGenerateProvider {
    /// Wrap a generate provider with retry logic.
    pub fn new(inner: Arc<dyn GenerateProvider>, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

#[async_trait]
impl GenerateProvider for RetryingGenerateProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        with_retry(&self.config, self.inner.name(), "generate", || {
            self.inner.generate(prompt, options)
        })
        .await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        with_retry(&self.config, self.inner.name(), "generate_stream", || {
            self.inner.generate_stream(prompt, options)
        })
        .await
    }

    fn supported_generate_parameters(&self) -> Vec<ParameterName> {
        self.inner.supported_generate_parameters()
    }
}
