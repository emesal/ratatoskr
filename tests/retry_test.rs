use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use ratatoskr::providers::retry::{
    RetryConfig, RetryingChatProvider, RetryingEmbeddingProvider, RetryingGenerateProvider,
};
use ratatoskr::providers::traits::{ChatProvider, EmbeddingProvider, GenerateProvider};
use ratatoskr::{
    ChatEvent, ChatOptions, ChatResponse, Embedding, FinishReason, GenerateEvent, GenerateOptions,
    GenerateResponse, Message, RatatoskrError, Result, ToolDefinition,
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
    fn name(&self) -> &str {
        "mock-retry"
    }

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
            model: Some("test".into()),
            tool_calls: vec![],
            usage: None,
            reasoning: None,
            finish_reason: FinishReason::Stop,
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
    let inner = Arc::new(FailThenSucceed::new(2, || RatatoskrError::RateLimited {
        retry_after: None,
    }));
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
    let inner = Arc::new(FailThenSucceed::new(10, || {
        RatatoskrError::Http("timeout".into())
    }));
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
    let inner = Arc::new(FailThenSucceed::new(1, || {
        RatatoskrError::AuthenticationFailed
    }));
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
    let inner = Arc::new(FailThenSucceed::new(1, || RatatoskrError::RateLimited {
        retry_after: Some(Duration::from_millis(50)),
    }));
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
    let inner = Arc::new(FailThenSucceed::new(1, || RatatoskrError::RateLimited {
        retry_after: None,
    }));
    let provider = RetryingChatProvider::new(inner.clone(), RetryConfig::disabled());

    let result = provider
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert_eq!(inner.call_count(), 1);
}

// ============================================================================
// Embedding retry tests
// ============================================================================

/// Mock embedding provider that fails N times then succeeds.
struct FailingEmbeddingProvider {
    fail_count: AtomicU32,
    fail_with: fn() -> RatatoskrError,
    total_calls: AtomicU32,
}

impl FailingEmbeddingProvider {
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
impl EmbeddingProvider for FailingEmbeddingProvider {
    fn name(&self) -> &str {
        "mock-embed"
    }

    async fn embed(&self, _text: &str, model: &str) -> Result<Embedding> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        let remaining = self.fail_count.load(Ordering::Relaxed);
        if remaining > 0 {
            self.fail_count.fetch_sub(1, Ordering::Relaxed);
            return Err((self.fail_with)());
        }
        Ok(Embedding {
            values: vec![0.1, 0.2, 0.3],
            model: model.to_string(),
            dimensions: 3,
        })
    }
}

#[tokio::test]
async fn embedding_retries_on_transient_then_succeeds() {
    let inner = Arc::new(FailingEmbeddingProvider::new(2, || {
        RatatoskrError::Http("timeout".into())
    }));
    let provider = RetryingEmbeddingProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let result = provider.embed("hello", "test-model").await;
    assert!(result.is_ok());
    assert_eq!(inner.call_count(), 3);
}

#[tokio::test]
async fn embedding_no_retry_on_permanent_error() {
    let inner = Arc::new(FailingEmbeddingProvider::new(1, || {
        RatatoskrError::AuthenticationFailed
    }));
    let provider = RetryingEmbeddingProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(5)
            .initial_delay(Duration::from_millis(1)),
    );

    let result = provider.embed("hello", "test-model").await;
    assert!(result.is_err());
    assert_eq!(inner.call_count(), 1);
}

// ============================================================================
// Generate retry tests
// ============================================================================

/// Mock generate provider that fails N times then succeeds.
struct FailingGenerateProvider {
    fail_count: AtomicU32,
    fail_with: fn() -> RatatoskrError,
    total_calls: AtomicU32,
}

impl FailingGenerateProvider {
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
impl GenerateProvider for FailingGenerateProvider {
    fn name(&self) -> &str {
        "mock-generate"
    }

    async fn generate(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<GenerateResponse> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        let remaining = self.fail_count.load(Ordering::Relaxed);
        if remaining > 0 {
            self.fail_count.fetch_sub(1, Ordering::Relaxed);
            return Err((self.fail_with)());
        }
        Ok(GenerateResponse {
            text: "generated".into(),
            model: Some("test".into()),
            usage: None,
            finish_reason: FinishReason::Stop,
        })
    }

    async fn generate_stream(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<GenerateEvent>> + Send>>>
    {
        Err(RatatoskrError::NotImplemented("stream"))
    }
}

#[tokio::test]
async fn generate_retries_on_transient_then_succeeds() {
    let inner = Arc::new(FailingGenerateProvider::new(1, || RatatoskrError::Api {
        status: 503,
        message: "unavailable".into(),
    }));
    let provider = RetryingGenerateProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let result = provider
        .generate("hello", &GenerateOptions::new("test"))
        .await;
    assert!(result.is_ok());
    assert_eq!(inner.call_count(), 2);
}

#[tokio::test]
async fn generate_no_retry_on_permanent_error() {
    let inner = Arc::new(FailingGenerateProvider::new(1, || {
        RatatoskrError::InvalidInput("bad prompt".into())
    }));
    let provider = RetryingGenerateProvider::new(
        inner.clone(),
        RetryConfig::new()
            .max_attempts(5)
            .initial_delay(Duration::from_millis(1)),
    );

    let result = provider
        .generate("hello", &GenerateOptions::new("test"))
        .await;
    assert!(result.is_err());
    assert_eq!(inner.call_count(), 1);
}
