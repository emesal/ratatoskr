//! Core ModelGateway trait

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, GenerateEvent,
    GenerateOptions, GenerateResponse, Message, ModelInfo, ModelMetadata, ModelStatus, NliResult,
    RatatoskrError, Result, StanceResult, Token, ToolDefinition,
};

/// The core gateway trait that all implementations must provide.
///
/// This trait abstracts over different LLM providers, allowing consumers
/// to interact with models without coupling to provider-specific implementations.
#[async_trait]
pub trait ModelGateway: Send + Sync {
    // ===== Phase 1: Chat (must implement) =====

    /// Streaming chat completion
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>;

    /// Non-streaming chat completion
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// What can this gateway do?
    fn capabilities(&self) -> Capabilities;

    // ===== Phase 3+: Default stubs =====

    /// Generate embeddings for text
    async fn embed(&self, _text: &str, _model: &str) -> Result<Embedding> {
        Err(RatatoskrError::NotImplemented("embed".into()))
    }

    /// Batch embedding generation
    async fn embed_batch(&self, _texts: &[&str], _model: &str) -> Result<Vec<Embedding>> {
        Err(RatatoskrError::NotImplemented("embed_batch".into()))
    }

    /// Natural language inference
    async fn infer_nli(
        &self,
        _premise: &str,
        _hypothesis: &str,
        _model: &str,
    ) -> Result<NliResult> {
        Err(RatatoskrError::NotImplemented("infer_nli".into()))
    }

    /// Zero-shot classification
    async fn classify_zero_shot(
        &self,
        _text: &str,
        _labels: &[&str],
        _model: &str,
    ) -> Result<ClassifyResult> {
        Err(RatatoskrError::NotImplemented("classify_zero_shot".into()))
    }

    /// Count tokens for a given model
    fn count_tokens(&self, _text: &str, _model: &str) -> Result<usize> {
        Err(RatatoskrError::NotImplemented("count_tokens".into()))
    }

    /// Batch NLI inference — more efficient for multiple pairs
    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        // Default: sequential fallback
        let mut results = Vec::with_capacity(pairs.len());
        for (premise, hypothesis) in pairs {
            results.push(self.infer_nli(premise, hypothesis, model).await?);
        }
        Ok(results)
    }

    /// Non-streaming text generation
    async fn generate(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<GenerateResponse> {
        Err(RatatoskrError::NotImplemented("generate".into()))
    }

    /// Streaming text generation
    async fn generate_stream(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        Err(RatatoskrError::NotImplemented("generate_stream".into()))
    }

    // ===== Phase 5: Extended capabilities =====

    /// Stance detection toward a target topic.
    ///
    /// Determines whether text expresses favor, against, or neutral stance
    /// toward a specific target topic.
    async fn classify_stance(
        &self,
        _text: &str,
        _target: &str,
        _model: &str,
    ) -> Result<StanceResult> {
        Err(RatatoskrError::NotImplemented("classify_stance".into()))
    }

    /// Tokenize text into detailed Token objects with IDs, text, and byte offsets.
    fn tokenize(&self, _text: &str, _model: &str) -> Result<Vec<Token>> {
        Err(RatatoskrError::NotImplemented("tokenize".into()))
    }

    /// List all available models and their capabilities.
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![]
    }

    /// Get the status of a specific model.
    fn model_status(&self, _model: &str) -> ModelStatus {
        ModelStatus::Unavailable {
            reason: "Not implemented".into(),
        }
    }

    // ===== Phase 6: Model intelligence =====

    /// Get extended metadata for a model, including parameter availability.
    ///
    /// Returns `None` if the model is not known to the registry or cache.
    /// This is a synchronous lookup — see [`fetch_model_metadata`](Self::fetch_model_metadata)
    /// for the async variant that can reach out to provider APIs.
    fn model_metadata(&self, _model: &str) -> Option<ModelMetadata> {
        None
    }

    /// Fetch metadata from the provider that would serve this model.
    ///
    /// Walks the chat provider fallback chain, populates the cache on success,
    /// and returns the result. Use [`model_metadata`](Self::model_metadata) for
    /// cached/registry lookups without network I/O.
    async fn fetch_model_metadata(&self, _model: &str) -> Result<ModelMetadata> {
        Err(RatatoskrError::NotImplemented(
            "fetch_model_metadata".into(),
        ))
    }

    // ===== Autoconfig presets =====

    /// Resolve a preset model ID for the given cost tier and capability.
    ///
    /// Returns `None` if no preset is configured for this combination.
    fn resolve_preset(&self, _tier: &str, _capability: &str) -> Option<String> {
        None
    }
}
