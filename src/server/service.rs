//! gRPC service implementation.
//!
//! Wraps any [`ModelGateway`] implementation behind tonic gRPC handlers,
//! mapping native types to/from protobuf via the [`convert`](super::convert) module.

use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use tonic::{Request, Response, Status};

use crate::ModelGateway;

use super::proto;
use super::proto::ratatoskr_server::Ratatoskr;

/// gRPC service that wraps a [`ModelGateway`] implementation.
///
/// Generic over `G` so it works with both [`EmbeddedGateway`](crate::EmbeddedGateway)
/// (for ratd) and any future gateway implementations.
pub struct RatatoskrService<G: ModelGateway> {
    gateway: Arc<G>,
}

impl<G: ModelGateway> RatatoskrService<G> {
    /// Create a new service wrapping the given gateway.
    pub fn new(gateway: Arc<G>) -> Self {
        Self { gateway }
    }
}

type GrpcResult<T> = Result<Response<T>, Status>;
type StreamResponse<T> = Pin<Box<dyn futures_util::Stream<Item = Result<T, Status>> + Send>>;

/// Convert a [`RatatoskrError`](crate::RatatoskrError) to a [`tonic::Status`].
fn to_status(err: crate::RatatoskrError) -> Status {
    use crate::RatatoskrError;
    match err {
        RatatoskrError::ModelNotFound(m) => Status::not_found(format!("Model not found: {m}")),
        RatatoskrError::ModelNotAvailable => Status::not_found("Model not available"),
        RatatoskrError::RateLimited { retry_after } => {
            let msg = match retry_after {
                Some(d) => format!("Rate limited, retry after {d:?}"),
                None => "Rate limited".to_string(),
            };
            Status::resource_exhausted(msg)
        }
        RatatoskrError::AuthenticationFailed => Status::unauthenticated("Authentication failed"),
        RatatoskrError::InvalidInput(msg) => Status::invalid_argument(msg),
        RatatoskrError::NotImplemented(op) => {
            Status::unimplemented(format!("Not implemented: {op}"))
        }
        RatatoskrError::Unsupported => Status::unimplemented("Operation not supported"),
        e => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl<G: ModelGateway + 'static> Ratatoskr for RatatoskrService<G> {
    // =========================================================================
    // Chat
    // =========================================================================

    async fn chat(&self, request: Request<proto::ChatRequest>) -> GrpcResult<proto::ChatResponse> {
        let req = request.into_inner();
        let messages: Vec<_> = req.messages.into_iter().map(Into::into).collect();
        let tools: Vec<_> = req.tools.into_iter().map(Into::into).collect();
        let options = req.options.map(Into::into).unwrap_or_default();

        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        let response = self
            .gateway
            .chat(&messages, tools_ref, &options)
            .await
            .map_err(to_status)?;

        Ok(Response::new(response.into()))
    }

    type ChatStreamStream = StreamResponse<proto::ChatEvent>;

    async fn chat_stream(
        &self,
        request: Request<proto::ChatRequest>,
    ) -> GrpcResult<Self::ChatStreamStream> {
        let req = request.into_inner();
        let messages: Vec<_> = req.messages.into_iter().map(Into::into).collect();
        let tools: Vec<_> = req.tools.into_iter().map(Into::into).collect();
        let options = req.options.map(Into::into).unwrap_or_default();

        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        let stream = self
            .gateway
            .chat_stream(&messages, tools_ref, &options)
            .await
            .map_err(to_status)?;

        let mapped = stream.map(|result| match result {
            Ok(event) => Ok(event.into()),
            Err(e) => Err(to_status(e)),
        });

        Ok(Response::new(Box::pin(mapped) as Self::ChatStreamStream))
    }

    // =========================================================================
    // Generate
    // =========================================================================

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> GrpcResult<proto::GenerateResponse> {
        let req = request.into_inner();
        let options = req
            .options
            .map(Into::into)
            .unwrap_or_else(|| crate::GenerateOptions::new(""));

        let response = self
            .gateway
            .generate(&req.prompt, &options)
            .await
            .map_err(to_status)?;

        Ok(Response::new(response.into()))
    }

    type GenerateStreamStream = StreamResponse<proto::GenerateEvent>;

    async fn generate_stream(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> GrpcResult<Self::GenerateStreamStream> {
        let req = request.into_inner();
        let options = req
            .options
            .map(Into::into)
            .unwrap_or_else(|| crate::GenerateOptions::new(""));

        let stream = self
            .gateway
            .generate_stream(&req.prompt, &options)
            .await
            .map_err(to_status)?;

        let mapped = stream.map(|result| match result {
            Ok(event) => Ok(event.into()),
            Err(e) => Err(to_status(e)),
        });

        Ok(Response::new(Box::pin(mapped) as Self::GenerateStreamStream))
    }

    // =========================================================================
    // Embeddings
    // =========================================================================

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> GrpcResult<proto::EmbedResponse> {
        let req = request.into_inner();
        let embedding = self
            .gateway
            .embed(&req.text, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(embedding.into()))
    }

    async fn embed_batch(
        &self,
        request: Request<proto::EmbedBatchRequest>,
    ) -> GrpcResult<proto::EmbedBatchResponse> {
        let req = request.into_inner();
        let texts: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .gateway
            .embed_batch(&texts, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(proto::EmbedBatchResponse {
            embeddings: embeddings.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // NLI
    // =========================================================================

    async fn infer_nli(
        &self,
        request: Request<proto::NliRequest>,
    ) -> GrpcResult<proto::NliResponse> {
        let req = request.into_inner();
        let result = self
            .gateway
            .infer_nli(&req.premise, &req.hypothesis, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    async fn infer_nli_batch(
        &self,
        request: Request<proto::NliBatchRequest>,
    ) -> GrpcResult<proto::NliBatchResponse> {
        let req = request.into_inner();
        let pairs: Vec<(&str, &str)> = req
            .pairs
            .iter()
            .map(|p| (p.premise.as_str(), p.hypothesis.as_str()))
            .collect();

        let results = self
            .gateway
            .infer_nli_batch(&pairs, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(proto::NliBatchResponse {
            results: results.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // Classification
    // =========================================================================

    async fn classify_zero_shot(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> GrpcResult<proto::ClassifyResponse> {
        let req = request.into_inner();
        let labels: Vec<&str> = req.labels.iter().map(|s| s.as_str()).collect();
        let result = self
            .gateway
            .classify_zero_shot(&req.text, &labels, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    async fn classify_stance(
        &self,
        request: Request<proto::StanceRequest>,
    ) -> GrpcResult<proto::StanceResponse> {
        let req = request.into_inner();
        let result = self
            .gateway
            .classify_stance(&req.text, &req.target, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    async fn count_tokens(
        &self,
        request: Request<proto::TokenCountRequest>,
    ) -> GrpcResult<proto::TokenCountResponse> {
        let req = request.into_inner();
        let count = self
            .gateway
            .count_tokens(&req.text, &req.model)
            .map_err(to_status)?;

        Ok(Response::new(proto::TokenCountResponse {
            count: count as u32,
        }))
    }

    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> GrpcResult<proto::TokenizeResponse> {
        let req = request.into_inner();
        let tokens = self
            .gateway
            .tokenize(&req.text, &req.model)
            .map_err(to_status)?;

        Ok(Response::new(proto::TokenizeResponse {
            tokens: tokens.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // Model Management
    // =========================================================================

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> GrpcResult<proto::ListModelsResponse> {
        let models = self.gateway.list_models();
        Ok(Response::new(proto::ListModelsResponse {
            models: models.into_iter().map(Into::into).collect(),
        }))
    }

    async fn model_status(
        &self,
        request: Request<proto::ModelStatusRequest>,
    ) -> GrpcResult<proto::ModelStatusResponse> {
        let req = request.into_inner();
        let status = self.gateway.model_status(&req.model);
        Ok(Response::new(status.into()))
    }

    // =========================================================================
    // Health
    // =========================================================================

    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> GrpcResult<proto::HealthResponse> {
        Ok(Response::new(proto::HealthResponse {
            healthy: true,
            version: crate::PKG_VERSION.to_string(),
            git_sha: Some(crate::GIT_SHA.to_string()),
        }))
    }
}
