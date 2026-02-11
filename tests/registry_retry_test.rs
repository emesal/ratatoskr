//! Tests for cross-provider retry fallback in ProviderRegistry.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use ratatoskr::providers::ProviderRegistry;
use ratatoskr::providers::retry::RetryConfig;
use ratatoskr::providers::traits::ChatProvider;
use ratatoskr::{
    ChatEvent, ChatOptions, ChatResponse, FinishReason, Message, RatatoskrError, Result,
    ToolDefinition,
};

/// Mock provider that always returns a specific error.
struct AlwaysFailProvider {
    name: &'static str,
    error: fn() -> RatatoskrError,
    call_count: AtomicU32,
}

impl AlwaysFailProvider {
    fn new(name: &'static str, error: fn() -> RatatoskrError) -> Self {
        Self {
            name,
            error,
            call_count: AtomicU32::new(0),
        }
    }

    fn calls(&self) -> u32 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ChatProvider for AlwaysFailProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Err((self.error)())
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

/// Mock provider that always succeeds.
struct SuccessProvider {
    name: &'static str,
    call_count: AtomicU32,
}

impl SuccessProvider {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            call_count: AtomicU32::new(0),
        }
    }

    fn calls(&self) -> u32 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ChatProvider for SuccessProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(ChatResponse {
            content: format!("from-{}", self.name),
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
async fn transient_error_triggers_fallback_to_next_provider() {
    let mut registry = ProviderRegistry::new();
    registry.set_retry_config(
        RetryConfig::new()
            .max_attempts(2)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let failing = Arc::new(AlwaysFailProvider::new("rate-limited", || {
        RatatoskrError::RateLimited { retry_after: None }
    }));
    let success = Arc::new(SuccessProvider::new("backup"));

    registry.add_chat(failing.clone());
    registry.add_chat(success.clone());

    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.content, "from-backup");

    // failing provider was called max_attempts times (retry exhausted)
    assert_eq!(failing.calls(), 2);
    // success provider was called once
    assert_eq!(success.calls(), 1);
}

#[tokio::test]
async fn all_providers_fail_transiently_returns_last_error() {
    let mut registry = ProviderRegistry::new();
    registry.set_retry_config(
        RetryConfig::new()
            .max_attempts(1)
            .initial_delay(Duration::from_millis(1))
            .jitter(false),
    );

    let provider1 = Arc::new(AlwaysFailProvider::new("p1", || {
        RatatoskrError::Http("timeout".into())
    }));
    let provider2 = Arc::new(AlwaysFailProvider::new("p2", || {
        RatatoskrError::RateLimited { retry_after: None }
    }));

    registry.add_chat(provider1.clone());
    registry.add_chat(provider2.clone());

    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    // Last error should be from provider2 (the last in chain)
    assert!(matches!(result, Err(RatatoskrError::RateLimited { .. })));
    assert_eq!(provider1.calls(), 1);
    assert_eq!(provider2.calls(), 1);
}

#[tokio::test]
async fn permanent_error_stops_chain_immediately() {
    let mut registry = ProviderRegistry::new();
    registry.set_retry_config(
        RetryConfig::new()
            .max_attempts(3)
            .initial_delay(Duration::from_millis(1)),
    );

    let failing = Arc::new(AlwaysFailProvider::new("auth-fail", || {
        RatatoskrError::AuthenticationFailed
    }));
    let success = Arc::new(SuccessProvider::new("backup"));

    registry.add_chat(failing.clone());
    registry.add_chat(success.clone());

    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert!(matches!(result, Err(RatatoskrError::AuthenticationFailed)));
    // Permanent error: no retry, no fallback
    assert_eq!(failing.calls(), 1);
    assert_eq!(success.calls(), 0);
}

#[tokio::test]
async fn model_not_available_still_triggers_fallback() {
    let mut registry = ProviderRegistry::new();
    // Even without retry config, ModelNotAvailable should fall through
    let failing = Arc::new(AlwaysFailProvider::new("wrong-model", || {
        RatatoskrError::ModelNotAvailable
    }));
    let success = Arc::new(SuccessProvider::new("right-model"));

    registry.add_chat(failing.clone());
    registry.add_chat(success.clone());

    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_ok());
    assert_eq!(failing.calls(), 1);
    assert_eq!(success.calls(), 1);
}

#[tokio::test]
async fn without_retry_config_transient_errors_are_terminal() {
    // Without retry config, transient errors are terminal — no fallback,
    // since there's no guarantee retries were attempted.
    let mut registry = ProviderRegistry::new();

    let failing = Arc::new(AlwaysFailProvider::new("flaky", || {
        RatatoskrError::Http("connection reset".into())
    }));
    let success = Arc::new(SuccessProvider::new("stable"));

    registry.add_chat(failing.clone());
    registry.add_chat(success.clone());

    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await;

    assert!(result.is_err());
    assert_eq!(failing.calls(), 1);
    // Second provider never reached
    assert_eq!(success.calls(), 0);
}
