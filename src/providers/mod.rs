//! Provider implementations for non-chat capabilities.
//!
//! This module contains clients for providers that handle specific capabilities
//! like embeddings, NLI, and classification. Chat is handled by the llm crate.

#[cfg(feature = "local-inference")]
pub mod fastembed;
#[cfg(feature = "huggingface")]
pub mod huggingface;
#[cfg(feature = "local-inference")]
pub mod onnx_nli;

#[cfg(feature = "local-inference")]
pub use fastembed::{EmbeddingModelInfo, FastEmbedProvider, LocalEmbeddingModel};
#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceClient;
#[cfg(feature = "local-inference")]
pub use onnx_nli::{LocalNliModel, NliModelInfo, OnnxNliProvider};
