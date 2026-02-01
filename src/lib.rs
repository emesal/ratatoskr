//! Ratatoskr - Unified model gateway for LLM APIs
//!
//! This crate provides a stable `ModelGateway` trait that abstracts over
//! different LLM providers, allowing consumers to interact with models
//! without coupling to provider-specific implementations.
//!
//! # Example
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

mod convert;
pub mod error;
pub mod gateway;
pub mod traits;
pub mod types;

// Re-export main types at crate root
pub use error::{RatatoskrError, Result};
pub use gateway::{EmbeddedGateway, Ratatoskr, RatatoskrBuilder};
pub use traits::ModelGateway;

// Re-export all types
pub use types::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason,
    Message, MessageContent, NliLabel, NliResult, ReasoningConfig, ReasoningEffort, ResponseFormat,
    Role, ToolCall, ToolChoice, ToolDefinition, Usage,
};
