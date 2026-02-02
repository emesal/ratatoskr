//! Ratatoskr - Unified model gateway for LLM APIs
//!
//! This crate provides a stable `ModelGateway` trait that abstracts over
//! different LLM providers, allowing consumers to interact with models
//! without coupling to provider-specific implementations.
//!
//! # Chat Example
//!
//! ```rust,no_run
//! use ratatoskr::{Ratatoskr, Message, ChatOptions, ModelGateway};
//!
//! #[tokio::main]
//! async fn main() -> ratatoskr::Result<()> {
//!     let gateway = Ratatoskr::builder()
//!         .openrouter("sk-or-your-key")
//!         .build()?;
//!
//!     let response = gateway.chat(
//!         &[
//!             Message::system("You are a helpful assistant."),
//!             Message::user("What is the capital of France?"),
//!         ],
//!         None,
//!         &ChatOptions::default().model("anthropic/claude-sonnet-4"),
//!     ).await?;
//!
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```
//!
//! # Embeddings Example (requires `huggingface` feature)
//!
//! ```rust,ignore
//! use ratatoskr::{Ratatoskr, ModelGateway};
//!
//! #[tokio::main]
//! async fn main() -> ratatoskr::Result<()> {
//!     let gateway = Ratatoskr::builder()
//!         .huggingface("hf_your_key")
//!         .build()?;
//!
//!     let embedding = gateway.embed(
//!         "Hello, world!",
//!         "sentence-transformers/all-MiniLM-L6-v2",
//!     ).await?;
//!
//!     println!("Dimensions: {}", embedding.dimensions);
//!     Ok(())
//! }
//! ```

mod convert;
pub mod error;
pub mod gateway;
#[cfg(feature = "local-inference")]
pub mod model;
#[cfg(feature = "huggingface")]
pub mod providers;
#[cfg(feature = "local-inference")]
pub mod tokenizer;
pub mod traits;
pub mod types;

// Re-export main types at crate root
pub use error::{RatatoskrError, Result};
pub use gateway::{EmbeddedGateway, Ratatoskr, RatatoskrBuilder};
pub use traits::ModelGateway;

// Re-export tokenizer types when feature is enabled
#[cfg(feature = "local-inference")]
pub use tokenizer::{HfTokenizer, TokenizerProvider, TokenizerRegistry, TokenizerSource};

// Re-export model types when feature is enabled
#[cfg(feature = "local-inference")]
pub use model::{Device, LoadedModels, ModelManager, ModelManagerConfig, ModelSource};

// Re-export all types
pub use types::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason,
    GenerateEvent, GenerateOptions, GenerateResponse, Message, MessageContent, NliLabel, NliResult,
    ReasoningConfig, ReasoningEffort, ResponseFormat, Role, ToolCall, ToolChoice, ToolDefinition,
    Usage,
};
