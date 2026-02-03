//! Provider traits for capability-specific implementations.
//!
//! Providers implement capability-specific traits (e.g., `EmbeddingProvider`, `NliProvider`)
//! rather than a single "god trait". This enables:
//! - Decorator patterns: `CachingProvider<T>`, `RetryingProvider<T>`
//! - Fallback chains: try providers in priority order
//! - RAM-aware routing: local providers signal unavailability
//!
//! # Fallback Semantics
//!
//! Providers receive the model string and self-report availability:
//! - Return `ModelNotAvailable` to signal the registry should try the next provider
//! - Other errors are terminal and propagated to the caller
//!
//! # Example
//!
//! ```ignore
//! // Local provider checks if it can handle the model
//! async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
//!     if model != self.supported_model {
//!         return Err(RatatoskrError::ModelNotAvailable);
//!     }
//!     // ... perform embedding
//! }
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;

use crate::Result;
use crate::types::{
    ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, GenerateEvent,
    GenerateOptions, GenerateResponse, Message, NliResult, StanceResult, ToolDefinition,
};

// ============================================================================
// Embedding Provider
// ============================================================================

/// Provider for text embeddings.
///
/// Providers receive the model string and self-report availability.
/// Return `ModelNotAvailable` to signal the registry should try the next provider.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Generate embedding for a single text.
    ///
    /// Returns `ModelNotAvailable` if this provider cannot handle the model
    /// (wrong model, RAM constrained, etc.) — registry will try next provider.
    async fn embed(&self, text: &str, model: &str) -> Result<Embedding>;

    /// Generate embeddings for multiple texts (batch).
    ///
    /// Default implementation calls `embed` sequentially.
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text, model).await?);
        }
        Ok(results)
    }
}

// ============================================================================
// NLI Provider
// ============================================================================

/// Provider for natural language inference.
#[async_trait]
pub trait NliProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Infer entailment/contradiction/neutral between premise and hypothesis.
    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>;

    /// Batch NLI inference.
    ///
    /// Default implementation calls `infer_nli` sequentially.
    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (premise, hypothesis) in pairs {
            results.push(self.infer_nli(premise, hypothesis, model).await?);
        }
        Ok(results)
    }
}

// ============================================================================
// Classification Provider
// ============================================================================

/// Provider for zero-shot text classification.
#[async_trait]
pub trait ClassifyProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Zero-shot classification with candidate labels.
    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult>;
}

// ============================================================================
// Stance Provider
// ============================================================================

/// Provider for stance detection.
///
/// Stance detection determines whether text expresses favor/against/neutral
/// toward a specific target topic.
#[async_trait]
pub trait StanceProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Detect stance toward a target topic.
    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult>;
}

/// Wrapper that implements `StanceProvider` using zero-shot classification.
///
/// Used as fallback when no dedicated stance model is available. Constructs
/// a prompt including the target and classifies against favor/against/neutral labels.
pub struct ZeroShotStanceProvider {
    inner: Arc<dyn ClassifyProvider>,
    /// Model to use for zero-shot classification.
    model: String,
}

impl ZeroShotStanceProvider {
    /// Create a new zero-shot stance provider wrapping a classify provider.
    pub fn new(inner: Arc<dyn ClassifyProvider>, model: impl Into<String>) -> Self {
        Self {
            inner,
            model: model.into(),
        }
    }
}

#[async_trait]
impl StanceProvider for ZeroShotStanceProvider {
    fn name(&self) -> &str {
        "zero-shot-stance"
    }

    async fn classify_stance(
        &self,
        text: &str,
        target: &str,
        _model: &str,
    ) -> Result<StanceResult> {
        // Construct prompt that includes target
        let prompt = format!("{} [Target: {}]", text, target);
        let labels = ["favor", "against", "neutral"];

        // Use the configured model (ignore passed model, we're a fallback)
        let result = self
            .inner
            .classify_zero_shot(&prompt, &labels, &self.model)
            .await?;

        let favor = *result.scores.get("favor").unwrap_or(&0.0);
        let against = *result.scores.get("against").unwrap_or(&0.0);
        let neutral = *result.scores.get("neutral").unwrap_or(&0.0);

        Ok(StanceResult::from_scores(favor, against, neutral, target))
    }
}

// ============================================================================
// Chat Provider
// ============================================================================

/// Provider for multi-turn chat.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Non-streaming chat completion.
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// Streaming chat completion.
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>;
}

// ============================================================================
// Generate Provider
// ============================================================================

/// Provider for single-turn text generation.
#[async_trait]
pub trait GenerateProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Non-streaming text generation.
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse>;

    /// Streaming text generation.
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ClassifyResult, StanceLabel};
    use std::collections::HashMap;

    /// Mock classify provider for testing ZeroShotStanceProvider.
    struct MockClassifyProvider {
        response: ClassifyResult,
    }

    impl MockClassifyProvider {
        fn with_scores(favor: f32, against: f32, neutral: f32) -> Self {
            let mut scores = HashMap::new();
            scores.insert("favor".to_string(), favor);
            scores.insert("against".to_string(), against);
            scores.insert("neutral".to_string(), neutral);

            // Determine top label
            let (top_label, confidence) = if favor >= against && favor >= neutral {
                ("favor", favor)
            } else if against >= favor && against >= neutral {
                ("against", against)
            } else {
                ("neutral", neutral)
            };

            Self {
                response: ClassifyResult {
                    scores,
                    top_label: top_label.to_string(),
                    confidence,
                },
            }
        }
    }

    #[async_trait]
    impl ClassifyProvider for MockClassifyProvider {
        fn name(&self) -> &str {
            "mock-classify"
        }

        async fn classify_zero_shot(
            &self,
            _text: &str,
            _labels: &[&str],
            _model: &str,
        ) -> Result<ClassifyResult> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn zero_shot_stance_provider_returns_favor() {
        let mock = Arc::new(MockClassifyProvider::with_scores(0.8, 0.1, 0.1));
        let provider = ZeroShotStanceProvider::new(mock, "test-model");

        let result = provider
            .classify_stance("I support this policy", "climate action", "ignored")
            .await
            .unwrap();

        assert_eq!(result.label, StanceLabel::Favor);
        assert_eq!(result.target, "climate action");
        assert!((result.favor - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn zero_shot_stance_provider_returns_against() {
        let mock = Arc::new(MockClassifyProvider::with_scores(0.1, 0.7, 0.2));
        let provider = ZeroShotStanceProvider::new(mock, "test-model");

        let result = provider
            .classify_stance("This is terrible", "new tax", "ignored")
            .await
            .unwrap();

        assert_eq!(result.label, StanceLabel::Against);
        assert_eq!(result.target, "new tax");
    }

    #[tokio::test]
    async fn zero_shot_stance_provider_name() {
        let mock = Arc::new(MockClassifyProvider::with_scores(0.3, 0.3, 0.4));
        let provider = ZeroShotStanceProvider::new(mock, "test-model");

        assert_eq!(provider.name(), "zero-shot-stance");
    }
}
