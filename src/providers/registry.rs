//! Provider registry with fallback chain semantics.
//!
//! The `ProviderRegistry` stores providers in priority order (index 0 = highest).
//! When a capability is requested, it tries providers in order until one succeeds
//! or returns a non-fallback error.
//!
//! # Fallback Triggers
//!
//! The registry falls through to the next provider on:
//! - `ModelNotAvailable` — provider can't handle this model
//! - Transient errors after retry exhaustion — all retry attempts failed
//!
//! Permanent errors (auth, validation, etc.) are terminal and stop the chain.
//!
//! # Retry Wrapping
//!
//! When a `RetryConfig` is set, providers are automatically wrapped in
//! `Retrying*Provider` decorators at registration time. This means each
//! provider retries internally before the registry sees a transient error
//! as exhausted.
//!
//! # Fallback Chain Flow
//!
//! ```text
//! User: gateway.embed("hello", "all-MiniLM-L6-v2")
//!                     │
//!                     ▼
//!         ┌─────────────────────┐
//!         │  ProviderRegistry   │
//!         │  embedding providers│
//!         └─────────┬───────────┘
//!                   │ try in order
//!                   ▼
//!         ┌─────────────────────┐
//!         │  FastEmbedProvider  │ ──► checks: right model? RAM available?
//!         │  (priority 0)       │ ──► if no: return ModelNotAvailable
//!         └─────────┬───────────┘
//!                   │ ModelNotAvailable
//!                   ▼
//!         ┌─────────────────────┐
//!         │  HuggingFaceClient  │ ──► calls API with model string
//!         │  (priority 1)       │ ──► returns embedding
//!         └─────────────────────┘
//! ```

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures_util::Stream;
use tracing::{instrument, warn};

use crate::telemetry;

use super::backpressure;
use super::backpressure::DEFAULT_STREAM_BUFFER;
use super::retry::{
    RetryConfig, RetryingChatProvider, RetryingEmbeddingProvider, RetryingGenerateProvider,
    RetryingNliProvider,
};
use super::traits::{
    ChatProvider, ClassifyProvider, EmbeddingProvider, GenerateProvider, NliProvider,
    StanceProvider,
};
use crate::types::{
    ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, GenerateEvent,
    GenerateOptions, GenerateResponse, Message, ModelMetadata, NliResult,
    ParameterValidationPolicy, StanceResult, ToolDefinition,
};
use crate::{RatatoskrError, Result};

/// Registry of providers with fallback chain semantics.
///
/// Providers are stored in priority order (index 0 = highest priority).
/// When a capability is requested, the registry tries providers in order
/// until one succeeds or returns a non-fallback error.
///
/// # Retry + Fallback
///
/// When a [`RetryConfig`] is set, providers are wrapped in `Retrying*Provider`
/// decorators at registration time. Transient errors after retry exhaustion
/// trigger fallback to the next provider in the chain.
///
/// # Parameter Validation
///
/// When a provider declares its supported parameters via `supported_chat_parameters()`
/// or `supported_generate_parameters()`, the registry can validate incoming requests.
/// Set `validation_policy` to control behaviour when unsupported parameters are found.
pub struct ProviderRegistry {
    embedding: Vec<Arc<dyn EmbeddingProvider>>,
    nli: Vec<Arc<dyn NliProvider>>,
    classify: Vec<Arc<dyn ClassifyProvider>>,
    stance: Vec<Arc<dyn StanceProvider>>,
    chat: Vec<Arc<dyn ChatProvider>>,
    generate: Vec<Arc<dyn GenerateProvider>>,
    retry_config: Option<RetryConfig>,
    validation_policy: ParameterValidationPolicy,
    stream_buffer_size: usize,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self {
            embedding: Vec::new(),
            nli: Vec::new(),
            classify: Vec::new(),
            stance: Vec::new(),
            chat: Vec::new(),
            generate: Vec::new(),
            retry_config: None,
            validation_policy: ParameterValidationPolicy::default(),
            stream_buffer_size: DEFAULT_STREAM_BUFFER,
        }
    }
}

impl ProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the retry configuration.
    ///
    /// When set, providers registered after this call are automatically
    /// wrapped in `Retrying*Provider` decorators. Transient errors after
    /// retry exhaustion trigger fallback to the next provider.
    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = Some(config);
    }

    /// Set the stream buffer size for backpressure.
    ///
    /// Controls the bounded channel capacity between stream producers
    /// and consumers. Default: [`DEFAULT_STREAM_BUFFER`] (64).
    pub fn set_stream_buffer_size(&mut self, size: usize) {
        self.stream_buffer_size = size;
    }

    /// Set the parameter validation policy.
    ///
    /// Controls how the registry handles requests containing parameters
    /// not declared as supported by the provider.
    pub fn set_validation_policy(&mut self, policy: ParameterValidationPolicy) {
        self.validation_policy = policy;
    }

    /// Get the current validation policy.
    pub fn validation_policy(&self) -> ParameterValidationPolicy {
        self.validation_policy
    }

    // ========================================================================
    // Registration methods (appends to end = lowest priority)
    // Call in priority order: first registered = highest priority
    // ========================================================================

    /// Add an embedding provider (appended to end of chain).
    ///
    /// If a retry config is set, the provider is automatically wrapped
    /// in [`RetryingEmbeddingProvider`].
    pub fn add_embedding(&mut self, provider: Arc<dyn EmbeddingProvider>) {
        self.embedding.push(self.maybe_wrap_embedding(provider));
    }

    /// Add an NLI provider (appended to end of chain).
    ///
    /// If a retry config is set, the provider is automatically wrapped
    /// in [`RetryingNliProvider`].
    pub fn add_nli(&mut self, provider: Arc<dyn NliProvider>) {
        self.nli.push(self.maybe_wrap_nli(provider));
    }

    /// Add a classification provider (appended to end of chain).
    ///
    /// Classification providers are not retry-wrapped (typically local).
    pub fn add_classify(&mut self, provider: Arc<dyn ClassifyProvider>) {
        self.classify.push(provider);
    }

    /// Add a stance provider (appended to end of chain).
    ///
    /// Stance providers are not retry-wrapped (typically local).
    pub fn add_stance(&mut self, provider: Arc<dyn StanceProvider>) {
        self.stance.push(provider);
    }

    /// Add a chat provider (appended to end of chain).
    ///
    /// If a retry config is set, the provider is automatically wrapped
    /// in [`RetryingChatProvider`].
    pub fn add_chat(&mut self, provider: Arc<dyn ChatProvider>) {
        self.chat.push(self.maybe_wrap_chat(provider));
    }

    /// Add a generate provider (appended to end of chain).
    ///
    /// If a retry config is set, the provider is automatically wrapped
    /// in [`RetryingGenerateProvider`].
    pub fn add_generate(&mut self, provider: Arc<dyn GenerateProvider>) {
        self.generate.push(self.maybe_wrap_generate(provider));
    }

    // ========================================================================
    // Fallback chain execution
    // Tries providers in order; ModelNotAvailable and exhausted transient
    // errors trigger fallback to the next provider.
    // ========================================================================

    /// Embed text using the fallback chain.
    #[instrument(skip(self, text), fields(operation = "embed"))]
    pub async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.embedding {
            match provider.embed(text, model).await {
                Ok(result) => {
                    Self::record_request("embed", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("embed", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("embed", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Embed batch of texts using the fallback chain.
    #[instrument(skip(self, texts), fields(operation = "embed_batch", batch_size = texts.len()))]
    pub async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.embedding {
            match provider.embed_batch(texts, model).await {
                Ok(result) => {
                    Self::record_request("embed_batch", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("embed_batch", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("embed_batch", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Infer NLI using the fallback chain.
    #[instrument(skip(self, premise, hypothesis), fields(operation = "infer_nli"))]
    pub async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<NliResult> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.nli {
            match provider.infer_nli(premise, hypothesis, model).await {
                Ok(result) => {
                    Self::record_request("infer_nli", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("infer_nli", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("infer_nli", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Batch NLI inference using the fallback chain.
    #[instrument(skip(self, pairs), fields(operation = "infer_nli_batch", batch_size = pairs.len()))]
    pub async fn infer_nli_batch(
        &self,
        pairs: &[(&str, &str)],
        model: &str,
    ) -> Result<Vec<NliResult>> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.nli {
            match provider.infer_nli_batch(pairs, model).await {
                Ok(result) => {
                    Self::record_request("infer_nli_batch", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("infer_nli_batch", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("infer_nli_batch", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Zero-shot classification using the fallback chain.
    #[instrument(skip(self, text, labels), fields(operation = "classify_zero_shot"))]
    pub async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.classify {
            match provider.classify_zero_shot(text, labels, model).await {
                Ok(result) => {
                    Self::record_request("classify_zero_shot", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("classify_zero_shot", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("classify_zero_shot", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Stance detection using the fallback chain.
    #[instrument(skip(self, text, target), fields(operation = "classify_stance"))]
    pub async fn classify_stance(
        &self,
        text: &str,
        target: &str,
        model: &str,
    ) -> Result<StanceResult> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.stance {
            match provider.classify_stance(text, target, model).await {
                Ok(result) => {
                    Self::record_request("classify_stance", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("classify_stance", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("classify_stance", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Chat completion using the fallback chain.
    ///
    /// If a provider declares `supported_chat_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    #[instrument(skip(self, messages, tools, options), fields(operation = "chat", model = %options.model))]
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.chat {
            // Validate parameters before calling provider
            self.validate_chat_params(provider.as_ref(), options)?;

            match provider.chat(messages, tools, options).await {
                Ok(result) => {
                    let name = provider.name();
                    Self::record_request("chat", name, start, true);
                    if let Some(ref usage) = result.usage {
                        Self::record_token_usage(name, usage);
                    }
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("chat", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("chat", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Streaming chat using the fallback chain.
    ///
    /// If a provider declares `supported_chat_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    #[instrument(skip(self, messages, tools, options), fields(operation = "chat_stream", model = %options.model))]
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.chat {
            // Validate parameters before calling provider
            self.validate_chat_params(provider.as_ref(), options)?;

            match provider.chat_stream(messages, tools, options).await {
                Ok(stream) => {
                    Self::record_request("chat_stream", provider.name(), start, true);
                    return Ok(backpressure::bounded_stream(
                        stream,
                        self.stream_buffer_size,
                    ));
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("chat_stream", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("chat_stream", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Text generation using the fallback chain.
    ///
    /// If a provider declares `supported_generate_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    #[instrument(skip(self, prompt, options), fields(operation = "generate", model = %options.model))]
    pub async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<GenerateResponse> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.generate {
            // Validate parameters before calling provider
            self.validate_generate_params(provider.as_ref(), options)?;

            match provider.generate(prompt, options).await {
                Ok(result) => {
                    Self::record_request("generate", provider.name(), start, true);
                    return Ok(result);
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("generate", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("generate", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    /// Streaming text generation using the fallback chain.
    ///
    /// If a provider declares `supported_generate_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    #[instrument(skip(self, prompt, options), fields(operation = "generate_stream", model = %options.model))]
    pub async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.generate {
            // Validate parameters before calling provider
            self.validate_generate_params(provider.as_ref(), options)?;

            match provider.generate_stream(prompt, options).await {
                Ok(stream) => {
                    Self::record_request("generate_stream", provider.name(), start, true);
                    return Ok(backpressure::bounded_stream(
                        stream,
                        self.stream_buffer_size,
                    ));
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("generate_stream", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("generate_stream", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    // ========================================================================
    // Metadata fetch (walks chat provider fallback chain)
    // ========================================================================

    /// Fetch model metadata from the chat provider fallback chain.
    ///
    /// Walks providers in priority order. `ModelNotAvailable`, `NotImplemented`,
    /// and exhausted transient errors trigger fallback; other errors are terminal.
    #[instrument(skip(self), fields(operation = "fetch_chat_metadata"))]
    pub async fn fetch_chat_metadata(&self, model: &str) -> Result<ModelMetadata> {
        let start = Instant::now();
        let mut last_err = None;
        for provider in &self.chat {
            match provider.fetch_metadata(model).await {
                Ok(metadata) => {
                    Self::record_request("fetch_chat_metadata", provider.name(), start, true);
                    return Ok(metadata);
                }
                Err(RatatoskrError::NotImplemented(_)) => {
                    last_err = Some(RatatoskrError::NotImplemented("fetch_metadata"));
                    continue;
                }
                Err(e) if Self::is_fallback_trigger(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => {
                    Self::record_request("fetch_chat_metadata", provider.name(), start, false);
                    return Err(e);
                }
            }
        }
        Self::record_request("fetch_chat_metadata", "none", start, false);
        Err(last_err.unwrap_or(RatatoskrError::NoProvider))
    }

    // ========================================================================
    // Capability introspection
    // ========================================================================

    /// Check if any embedding providers are registered.
    pub fn has_embedding(&self) -> bool {
        !self.embedding.is_empty()
    }

    /// Check if any NLI providers are registered.
    pub fn has_nli(&self) -> bool {
        !self.nli.is_empty()
    }

    /// Check if any classification providers are registered.
    pub fn has_classify(&self) -> bool {
        !self.classify.is_empty()
    }

    /// Check if any stance providers are registered.
    pub fn has_stance(&self) -> bool {
        !self.stance.is_empty()
    }

    /// Check if any chat providers are registered.
    pub fn has_chat(&self) -> bool {
        !self.chat.is_empty()
    }

    /// Check if any generate providers are registered.
    pub fn has_generate(&self) -> bool {
        !self.generate.is_empty()
    }

    /// List all registered provider names per capability (in priority order).
    pub fn provider_names(&self) -> ProviderNames {
        ProviderNames {
            embedding: self
                .embedding
                .iter()
                .map(|p| p.name().to_string())
                .collect(),
            nli: self.nli.iter().map(|p| p.name().to_string()).collect(),
            classify: self.classify.iter().map(|p| p.name().to_string()).collect(),
            stance: self.stance.iter().map(|p| p.name().to_string()).collect(),
            chat: self.chat.iter().map(|p| p.name().to_string()).collect(),
            generate: self.generate.iter().map(|p| p.name().to_string()).collect(),
        }
    }

    // ========================================================================
    // Metrics recording
    // ========================================================================

    /// Record request outcome metrics (counter + histogram).
    fn record_request(operation: &'static str, provider: &str, start: Instant, ok: bool) {
        let status = if ok { "ok" } else { "error" };
        let elapsed = start.elapsed().as_secs_f64();
        metrics::counter!(telemetry::REQUESTS_TOTAL,
            "provider" => provider.to_owned(),
            "operation" => operation,
            "status" => status,
        )
        .increment(1);
        metrics::histogram!(telemetry::REQUEST_DURATION_SECONDS,
            "provider" => provider.to_owned(),
            "operation" => operation,
        )
        .record(elapsed);
    }

    /// Record token usage metrics from a chat response.
    fn record_token_usage(provider: &str, usage: &crate::types::Usage) {
        metrics::counter!(telemetry::TOKENS_TOTAL,
            "provider" => provider.to_owned(),
            "direction" => "prompt",
        )
        .increment(u64::from(usage.prompt_tokens));
        metrics::counter!(telemetry::TOKENS_TOTAL,
            "provider" => provider.to_owned(),
            "direction" => "completion",
        )
        .increment(u64::from(usage.completion_tokens));
    }

    // ========================================================================
    // Retry wrapping helpers
    // ========================================================================

    /// Whether an error should trigger fallback to the next provider.
    ///
    /// `ModelNotAvailable` always triggers fallback. Transient errors trigger
    /// fallback when retry is configured (meaning retries were exhausted
    /// inside the `RetryingProvider` wrapper).
    fn is_fallback_trigger(e: &RatatoskrError) -> bool {
        matches!(e, RatatoskrError::ModelNotAvailable) || e.is_transient()
    }

    /// Wrap an embedding provider in retry decorator if config is set.
    fn maybe_wrap_embedding(
        &self,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Arc<dyn EmbeddingProvider> {
        match &self.retry_config {
            Some(config) => Arc::new(RetryingEmbeddingProvider::new(provider, config.clone())),
            None => provider,
        }
    }

    /// Wrap an NLI provider in retry decorator if config is set.
    fn maybe_wrap_nli(&self, provider: Arc<dyn NliProvider>) -> Arc<dyn NliProvider> {
        match &self.retry_config {
            Some(config) => Arc::new(RetryingNliProvider::new(provider, config.clone())),
            None => provider,
        }
    }

    /// Wrap a chat provider in retry decorator if config is set.
    fn maybe_wrap_chat(&self, provider: Arc<dyn ChatProvider>) -> Arc<dyn ChatProvider> {
        match &self.retry_config {
            Some(config) => Arc::new(RetryingChatProvider::new(provider, config.clone())),
            None => provider,
        }
    }

    /// Wrap a generate provider in retry decorator if config is set.
    fn maybe_wrap_generate(
        &self,
        provider: Arc<dyn GenerateProvider>,
    ) -> Arc<dyn GenerateProvider> {
        match &self.retry_config {
            Some(config) => Arc::new(RetryingGenerateProvider::new(provider, config.clone())),
            None => provider,
        }
    }

    // ========================================================================
    // Parameter validation
    // ========================================================================

    /// Validate parameters for a chat request against provider-declared support.
    ///
    /// Returns `Ok(())` if validation passes or if the policy is `Ignore`.
    /// Returns `Err(UnsupportedParameter)` if policy is `Error` and unsupported params found.
    /// Logs warnings if policy is `Warn` and returns `Ok(())`.
    fn validate_chat_params(
        &self,
        provider: &dyn ChatProvider,
        options: &ChatOptions,
    ) -> Result<()> {
        let supported = provider.supported_chat_parameters();

        // If provider doesn't declare parameters, skip validation (legacy)
        if supported.is_empty() {
            return Ok(());
        }

        let requested = options.set_parameters();
        let unsupported: Vec<_> = requested
            .iter()
            .filter(|p| !supported.contains(p))
            .collect();

        if unsupported.is_empty() {
            return Ok(());
        }

        match self.validation_policy {
            ParameterValidationPolicy::Ignore => Ok(()),
            ParameterValidationPolicy::Warn => {
                for param in &unsupported {
                    warn!(
                        param = %param,
                        model = %options.model,
                        provider = provider.name(),
                        "unsupported parameter"
                    );
                }
                Ok(())
            }
            ParameterValidationPolicy::Error => {
                // Report first unsupported param in error
                Err(RatatoskrError::UnsupportedParameter {
                    param: unsupported[0].to_string(),
                    model: options.model.clone(),
                    provider: provider.name().to_string(),
                })
            }
        }
    }

    /// Validate parameters for a generate request against provider-declared support.
    fn validate_generate_params(
        &self,
        provider: &dyn GenerateProvider,
        options: &GenerateOptions,
    ) -> Result<()> {
        let supported = provider.supported_generate_parameters();

        // If provider doesn't declare parameters, skip validation (legacy)
        if supported.is_empty() {
            return Ok(());
        }

        let requested = options.set_parameters();
        let unsupported: Vec<_> = requested
            .iter()
            .filter(|p| !supported.contains(p))
            .collect();

        if unsupported.is_empty() {
            return Ok(());
        }

        match self.validation_policy {
            ParameterValidationPolicy::Ignore => Ok(()),
            ParameterValidationPolicy::Warn => {
                for param in &unsupported {
                    warn!(
                        param = %param,
                        model = %options.model,
                        provider = provider.name(),
                        "unsupported parameter"
                    );
                }
                Ok(())
            }
            ParameterValidationPolicy::Error => Err(RatatoskrError::UnsupportedParameter {
                param: unsupported[0].to_string(),
                model: options.model.clone(),
                provider: provider.name().to_string(),
            }),
        }
    }
}

/// Provider names per capability, in priority order.
#[derive(Debug, Clone, Default)]
pub struct ProviderNames {
    /// Embedding provider names.
    pub embedding: Vec<String>,
    /// NLI provider names.
    pub nli: Vec<String>,
    /// Classification provider names.
    pub classify: Vec<String>,
    /// Stance provider names.
    pub stance: Vec<String>,
    /// Chat provider names.
    pub chat: Vec<String>,
    /// Generate provider names.
    pub generate: Vec<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NliLabel;

    /// Mock embedding provider that only handles a specific model.
    struct MockEmbeddingProvider {
        name: &'static str,
        supported_model: &'static str,
        dimensions: usize,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        fn name(&self) -> &str {
            self.name
        }

        async fn embed(&self, _text: &str, model: &str) -> Result<Embedding> {
            if model != self.supported_model {
                return Err(RatatoskrError::ModelNotAvailable);
            }
            Ok(Embedding {
                values: vec![0.1; self.dimensions],
                model: model.to_string(),
                dimensions: self.dimensions,
            })
        }
    }

    /// Mock NLI provider for testing.
    struct MockNliProvider {
        name: &'static str,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl NliProvider for MockNliProvider {
        fn name(&self) -> &str {
            self.name
        }

        async fn infer_nli(
            &self,
            _premise: &str,
            _hypothesis: &str,
            _model: &str,
        ) -> Result<NliResult> {
            if self.fail {
                return Err(RatatoskrError::ModelNotAvailable);
            }
            Ok(NliResult {
                entailment: 0.8,
                contradiction: 0.1,
                neutral: 0.1,
                label: NliLabel::Entailment,
            })
        }
    }

    #[tokio::test]
    async fn registry_embed_uses_first_matching_provider() {
        let mut registry = ProviderRegistry::new();

        // Provider 1: only handles "model-a"
        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "provider-a",
            supported_model: "model-a",
            dimensions: 128,
        }));

        // Provider 2: only handles "model-b"
        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "provider-b",
            supported_model: "model-b",
            dimensions: 256,
        }));

        // Request model-a: should use provider-a
        let result = registry.embed("test", "model-a").await.unwrap();
        assert_eq!(result.dimensions, 128);

        // Request model-b: should fallback to provider-b
        let result = registry.embed("test", "model-b").await.unwrap();
        assert_eq!(result.dimensions, 256);
    }

    #[tokio::test]
    async fn registry_returns_last_error_when_all_fail() {
        let mut registry = ProviderRegistry::new();

        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "provider-a",
            supported_model: "model-a",
            dimensions: 128,
        }));

        // Request unknown model — provider returns ModelNotAvailable,
        // which becomes the last error in the fallback chain
        let result = registry.embed("test", "unknown-model").await;
        assert!(matches!(result, Err(RatatoskrError::ModelNotAvailable)));
    }

    #[tokio::test]
    async fn registry_returns_no_provider_when_empty() {
        let registry = ProviderRegistry::new();
        let result = registry.embed("test", "any-model").await;
        assert!(matches!(result, Err(RatatoskrError::NoProvider)));
    }

    #[tokio::test]
    async fn registry_nli_fallback_chain() {
        let mut registry = ProviderRegistry::new();

        // First provider always fails
        registry.add_nli(Arc::new(MockNliProvider {
            name: "failing",
            fail: true,
        }));

        // Second provider succeeds
        registry.add_nli(Arc::new(MockNliProvider {
            name: "working",
            fail: false,
        }));

        let result = registry
            .infer_nli("premise", "hypothesis", "model")
            .await
            .unwrap();
        assert_eq!(result.label, NliLabel::Entailment);
    }

    #[test]
    fn registry_capability_introspection() {
        let mut registry = ProviderRegistry::new();

        assert!(!registry.has_embedding());
        assert!(!registry.has_nli());

        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "test",
            supported_model: "test",
            dimensions: 128,
        }));

        assert!(registry.has_embedding());
        assert!(!registry.has_nli());
    }

    #[test]
    fn registry_provider_names() {
        let mut registry = ProviderRegistry::new();

        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "first",
            supported_model: "a",
            dimensions: 128,
        }));
        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "second",
            supported_model: "b",
            dimensions: 256,
        }));

        let names = registry.provider_names();
        assert_eq!(names.embedding, vec!["first", "second"]);
    }
}
