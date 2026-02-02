//! Provider implementations for non-chat capabilities.
//!
//! This module contains clients for providers that handle specific capabilities
//! like embeddings, NLI, and classification. Chat is handled by the llm crate.

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceClient;
