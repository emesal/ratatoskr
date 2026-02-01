//! Public types for the Ratatoskr API

mod capabilities;
mod future;
mod message;
mod options;
pub mod response;
mod tool;

pub use capabilities::Capabilities;
pub use future::{ClassifyResult, Embedding, NliLabel, NliResult};
pub use message::{Message, MessageContent, Role};
pub use options::{ChatOptions, ReasoningConfig, ReasoningEffort, ResponseFormat};
pub use response::{ChatEvent, ChatResponse, FinishReason, Usage};
pub use tool::{ToolCall, ToolChoice, ToolDefinition};
