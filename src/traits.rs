//! Core ModelGateway trait

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, Message,
    NliResult, RatatoskrError, Result, ToolDefinition,
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
        Err(RatatoskrError::NotImplemented("embed"))
    }

    /// Batch embedding generation
    async fn embed_batch(&self, _texts: &[&str], _model: &str) -> Result<Vec<Embedding>> {
        Err(RatatoskrError::NotImplemented("embed_batch"))
    }

    /// Natural language inference
    async fn infer_nli(
        &self,
        _premise: &str,
        _hypothesis: &str,
        _model: &str,
    ) -> Result<NliResult> {
        Err(RatatoskrError::NotImplemented("infer_nli"))
    }

    /// Zero-shot classification
    async fn classify_zero_shot(
        &self,
        _text: &str,
        _labels: &[&str],
        _model: &str,
    ) -> Result<ClassifyResult> {
        Err(RatatoskrError::NotImplemented("classify_zero_shot"))
    }

    /// Count tokens for a given model
    fn count_tokens(&self, _text: &str, _model: &str) -> Result<usize> {
        Err(RatatoskrError::NotImplemented("count_tokens"))
    }
}
