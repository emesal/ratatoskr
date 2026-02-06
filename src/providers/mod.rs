//! Provider implementations for capabilities.
//!
//! This module contains:
//! - Provider traits (`EmbeddingProvider`, `NliProvider`, etc.)
//! - `ProviderRegistry` for fallback chain routing
//! - Concrete provider implementations (HuggingFace, FastEmbed, ONNX, etc.)

pub mod llm_chat;
pub(crate) mod openrouter_models;
pub mod registry;
pub mod traits;

#[cfg(feature = "local-inference")]
pub mod fastembed;
#[cfg(feature = "huggingface")]
pub mod huggingface;
#[cfg(feature = "local-inference")]
pub mod onnx_nli;

// Re-export traits
pub use registry::{ProviderNames, ProviderRegistry};
pub use traits::{
    ChatProvider, ClassifyProvider, EmbeddingProvider, GenerateProvider, NliProvider,
    StanceProvider, ZeroShotStanceProvider,
};

// Re-export concrete providers
#[cfg(feature = "local-inference")]
pub use fastembed::{
    EmbeddingModelInfo, FastEmbedProvider, LocalEmbeddingModel, LocalEmbeddingProvider,
};
#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceClient;
pub use llm_chat::LlmChatProvider;
#[cfg(feature = "local-inference")]
pub use onnx_nli::{LocalNliModel, LocalNliProvider, NliModelInfo, OnnxNliProvider};
