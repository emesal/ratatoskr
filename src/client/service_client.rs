//! [`ServiceClient`] — [`ModelGateway`] implementation that connects to ratd over gRPC.
//!
//! All proto ↔ native type conversions are centralized in [`crate::server::convert`].
//!
//! # Synchronous Methods
//!
//! Several [`ModelGateway`] trait methods are synchronous (`count_tokens`, `tokenize`,
//! `list_models`, `model_status`, `model_metadata`). Since the underlying gRPC calls
//! are async, these methods use [`tokio::task::block_in_place`] to safely block
//! without risking deadlocks when called from within an async context.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use tokio::task::block_in_place;
use tonic::transport::Channel;

use crate::server::proto;
use crate::server::proto::ratatoskr_client::RatatoskrClient;
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, GenerateEvent,
    GenerateOptions, GenerateResponse, Message, ModelGateway, ModelInfo, ModelMetadata,
    ModelStatus, NliResult, RatatoskrError, Result, StanceResult, Token, ToolDefinition,
};

/// A [`ModelGateway`] client that connects to a remote ratd server.
///
/// All gateway methods are forwarded over gRPC, with type conversions
/// handled by the shared [`convert`](crate::server::convert) module.
pub struct ServiceClient {
    inner: RatatoskrClient<Channel>,
}

impl ServiceClient {
    /// Connect to a ratd server at the given address.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = ServiceClient::connect("http://127.0.0.1:9741").await?;
    /// ```
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let inner = RatatoskrClient::connect(addr.clone())
            .await
            .map_err(|e| RatatoskrError::Http(format!("failed to connect to {addr}: {e}")))?;
        Ok(Self { inner })
    }
}

/// Convert [`tonic::Status`] to [`RatatoskrError`].
fn from_status(status: tonic::Status) -> RatatoskrError {
    match status.code() {
        tonic::Code::NotFound => RatatoskrError::ModelNotAvailable,
        tonic::Code::ResourceExhausted => RatatoskrError::RateLimited { retry_after: None },
        tonic::Code::Unauthenticated => RatatoskrError::AuthenticationFailed,
        tonic::Code::InvalidArgument => RatatoskrError::InvalidInput(status.message().to_string()),
        tonic::Code::Unimplemented => {
            // Leak the string to get a &'static str — acceptable for error paths
            RatatoskrError::NotImplemented(Box::leak(status.message().to_string().into_boxed_str()))
        }
        _ => RatatoskrError::Http(status.message().to_string()),
    }
}

// =============================================================================
// ModelGateway implementation
// =============================================================================

#[async_trait]
impl ModelGateway for ServiceClient {
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let request = build_chat_request(messages, tools, options);
        let response = self
            .inner
            .clone()
            .chat(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let request = build_chat_request(messages, tools, options);
        let response = self
            .inner
            .clone()
            .chat_stream(request)
            .await
            .map_err(from_status)?;
        let stream = response
            .into_inner()
            .map(|result| result.map(Into::into).map_err(from_status));
        Ok(Box::pin(stream))
    }

    fn capabilities(&self) -> Capabilities {
        // ServiceClient reports all capabilities; the server determines actual availability.
        Capabilities {
            chat: true,
            chat_streaming: true,
            generate: true,
            tool_use: true,
            embeddings: true,
            nli: true,
            classification: true,
            stance: true,
            token_counting: true,
            local_inference: false,
        }
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        let request = proto::EmbedRequest {
            text: text.to_string(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .embed(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        let request = proto::EmbedBatchRequest {
            texts: texts.iter().map(|s| s.to_string()).collect(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .embed_batch(request)
            .await
            .map_err(from_status)?;
        Ok(response
            .into_inner()
            .embeddings
            .into_iter()
            .map(Into::into)
            .collect())
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        let request = proto::NliRequest {
            premise: premise.to_string(),
            hypothesis: hypothesis.to_string(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .infer_nli(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        let request = proto::NliBatchRequest {
            pairs: pairs
                .iter()
                .map(|(p, h)| proto::NliPair {
                    premise: p.to_string(),
                    hypothesis: h.to_string(),
                })
                .collect(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .infer_nli_batch(request)
            .await
            .map_err(from_status)?;
        Ok(response
            .into_inner()
            .results
            .into_iter()
            .map(Into::into)
            .collect())
    }

    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult> {
        let request = proto::ClassifyRequest {
            text: text.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .classify_zero_shot(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        let request = proto::StanceRequest {
            text: text.to_string(),
            target: target.to_string(),
            model: model.to_string(),
        };
        let response = self
            .inner
            .clone()
            .classify_stance(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        // Synchronous trait method — use block_in_place to safely block on async gRPC.
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| RatatoskrError::Configuration("no tokio runtime".into()))?;

        let request = proto::TokenCountRequest {
            text: text.to_string(),
            model: model.to_string(),
        };
        let mut client = self.inner.clone();
        let result = block_in_place(|| rt.block_on(async { client.count_tokens(request).await }))
            .map_err(from_status)?;
        Ok(result.into_inner().count as usize)
    }

    fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| RatatoskrError::Configuration("no tokio runtime".into()))?;

        let request = proto::TokenizeRequest {
            text: text.to_string(),
            model: model.to_string(),
        };
        let mut client = self.inner.clone();
        let result = block_in_place(|| rt.block_on(async { client.tokenize(request).await }))
            .map_err(from_status)?;
        Ok(result
            .into_inner()
            .tokens
            .into_iter()
            .map(Into::into)
            .collect())
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        let request = proto::GenerateRequest {
            prompt: prompt.to_string(),
            options: Some(options.clone().into()),
        };
        let response = self
            .inner
            .clone()
            .generate(request)
            .await
            .map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        let request = proto::GenerateRequest {
            prompt: prompt.to_string(),
            options: Some(options.clone().into()),
        };
        let response = self
            .inner
            .clone()
            .generate_stream(request)
            .await
            .map_err(from_status)?;
        let stream = response
            .into_inner()
            .map(|result| result.map(Into::into).map_err(from_status));
        Ok(Box::pin(stream))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let rt = match tokio::runtime::Handle::try_current() {
            Ok(rt) => rt,
            Err(_) => return vec![],
        };

        let mut client = self.inner.clone();
        match block_in_place(|| {
            rt.block_on(async { client.list_models(proto::ListModelsRequest {}).await })
        }) {
            Ok(response) => response
                .into_inner()
                .models
                .into_iter()
                .map(Into::into)
                .collect(),
            Err(_) => vec![],
        }
    }

    fn model_status(&self, model: &str) -> ModelStatus {
        let rt = match tokio::runtime::Handle::try_current() {
            Ok(rt) => rt,
            Err(_) => {
                return ModelStatus::Unavailable {
                    reason: "no runtime".into(),
                };
            }
        };

        let request = proto::ModelStatusRequest {
            model: model.to_string(),
        };
        let mut client = self.inner.clone();
        match block_in_place(|| rt.block_on(async { client.model_status(request).await })) {
            Ok(response) => response.into_inner().into(),
            Err(e) => ModelStatus::Unavailable {
                reason: e.message().to_string(),
            },
        }
    }

    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        let rt = tokio::runtime::Handle::try_current().ok()?;

        let request = proto::ModelMetadataRequest {
            model: model.to_string(),
        };
        let mut client = self.inner.clone();
        let response =
            block_in_place(|| rt.block_on(async { client.get_model_metadata(request).await }))
                .ok()?;
        let resp = response.into_inner();
        if resp.found {
            resp.metadata.map(Into::into)
        } else {
            None
        }
    }
}

// =============================================================================
// Request builders
// =============================================================================

/// Build a [`proto::ChatRequest`] from native types.
fn build_chat_request(
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    options: &ChatOptions,
) -> proto::ChatRequest {
    proto::ChatRequest {
        messages: messages.iter().map(|m| m.clone().into()).collect(),
        tools: tools
            .map(|ts| ts.iter().map(|t| t.clone().into()).collect())
            .unwrap_or_default(),
        options: Some(options.clone().into()),
    }
}
