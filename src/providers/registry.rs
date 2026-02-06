//! Provider registry with fallback chain semantics.
//!
//! The `ProviderRegistry` stores providers in priority order (index 0 = highest).
//! When a capability is requested, it tries providers in order until one succeeds
//! or returns a non-`ModelNotAvailable` error.
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

use futures_util::Stream;
use tracing::warn;

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
/// until one succeeds or returns a non-`ModelNotAvailable` error.
///
/// # Parameter Validation
///
/// When a provider declares its supported parameters via `supported_chat_parameters()`
/// or `supported_generate_parameters()`, the registry can validate incoming requests.
/// Set `validation_policy` to control behaviour when unsupported parameters are found.
#[derive(Default)]
pub struct ProviderRegistry {
    embedding: Vec<Arc<dyn EmbeddingProvider>>,
    nli: Vec<Arc<dyn NliProvider>>,
    classify: Vec<Arc<dyn ClassifyProvider>>,
    stance: Vec<Arc<dyn StanceProvider>>,
    chat: Vec<Arc<dyn ChatProvider>>,
    generate: Vec<Arc<dyn GenerateProvider>>,
    validation_policy: ParameterValidationPolicy,
}

impl ProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
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
    pub fn add_embedding(&mut self, provider: Arc<dyn EmbeddingProvider>) {
        self.embedding.push(provider);
    }

    /// Add an NLI provider (appended to end of chain).
    pub fn add_nli(&mut self, provider: Arc<dyn NliProvider>) {
        self.nli.push(provider);
    }

    /// Add a classification provider (appended to end of chain).
    pub fn add_classify(&mut self, provider: Arc<dyn ClassifyProvider>) {
        self.classify.push(provider);
    }

    /// Add a stance provider (appended to end of chain).
    pub fn add_stance(&mut self, provider: Arc<dyn StanceProvider>) {
        self.stance.push(provider);
    }

    /// Add a chat provider (appended to end of chain).
    pub fn add_chat(&mut self, provider: Arc<dyn ChatProvider>) {
        self.chat.push(provider);
    }

    /// Add a generate provider (appended to end of chain).
    pub fn add_generate(&mut self, provider: Arc<dyn GenerateProvider>) {
        self.generate.push(provider);
    }

    // ========================================================================
    // Fallback chain execution
    // Tries providers in order; ModelNotAvailable means "try next"
    // ========================================================================

    /// Embed text using the fallback chain.
    pub async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        for provider in &self.embedding {
            match provider.embed(text, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Embed batch of texts using the fallback chain.
    pub async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        for provider in &self.embedding {
            match provider.embed_batch(texts, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Infer NLI using the fallback chain.
    pub async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<NliResult> {
        for provider in &self.nli {
            match provider.infer_nli(premise, hypothesis, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Batch NLI inference using the fallback chain.
    pub async fn infer_nli_batch(
        &self,
        pairs: &[(&str, &str)],
        model: &str,
    ) -> Result<Vec<NliResult>> {
        for provider in &self.nli {
            match provider.infer_nli_batch(pairs, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Zero-shot classification using the fallback chain.
    pub async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult> {
        for provider in &self.classify {
            match provider.classify_zero_shot(text, labels, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Stance detection using the fallback chain.
    pub async fn classify_stance(
        &self,
        text: &str,
        target: &str,
        model: &str,
    ) -> Result<StanceResult> {
        for provider in &self.stance {
            match provider.classify_stance(text, target, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Chat completion using the fallback chain.
    ///
    /// If a provider declares `supported_chat_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        for provider in &self.chat {
            // Validate parameters before calling provider
            self.validate_chat_params(provider.as_ref(), options)?;

            match provider.chat(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Streaming chat using the fallback chain.
    ///
    /// If a provider declares `supported_chat_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        for provider in &self.chat {
            // Validate parameters before calling provider
            self.validate_chat_params(provider.as_ref(), options)?;

            match provider.chat_stream(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Text generation using the fallback chain.
    ///
    /// If a provider declares `supported_generate_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    pub async fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<GenerateResponse> {
        for provider in &self.generate {
            // Validate parameters before calling provider
            self.validate_generate_params(provider.as_ref(), options)?;

            match provider.generate(prompt, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    /// Streaming text generation using the fallback chain.
    ///
    /// If a provider declares `supported_generate_parameters()`, the registry validates
    /// the request against that list according to the `validation_policy`.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        for provider in &self.generate {
            // Validate parameters before calling provider
            self.validate_generate_params(provider.as_ref(), options)?;

            match provider.generate_stream(prompt, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    // ========================================================================
    // Metadata fetch (walks chat provider fallback chain)
    // ========================================================================

    /// Fetch model metadata from the chat provider fallback chain.
    ///
    /// Walks providers in priority order. `ModelNotAvailable` and `NotImplemented`
    /// trigger fallback to the next provider; other errors are terminal.
    pub async fn fetch_chat_metadata(&self, model: &str) -> Result<ModelMetadata> {
        for provider in &self.chat {
            match provider.fetch_metadata(model).await {
                Ok(metadata) => return Ok(metadata),
                Err(RatatoskrError::ModelNotAvailable) | Err(RatatoskrError::NotImplemented(_)) => {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
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
    async fn registry_returns_no_provider_when_all_fail() {
        let mut registry = ProviderRegistry::new();

        registry.add_embedding(Arc::new(MockEmbeddingProvider {
            name: "provider-a",
            supported_model: "model-a",
            dimensions: 128,
        }));

        // Request unknown model
        let result = registry.embed("test", "unknown-model").await;
        assert!(matches!(result, Err(RatatoskrError::NoProvider)));
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
