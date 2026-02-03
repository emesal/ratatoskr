//! Public types for the Ratatoskr API.

mod capabilities;
mod future;
mod generate;
mod message;
mod model;
mod options;
pub mod response;
mod stance;
mod token;
mod tool;

pub use capabilities::Capabilities;
pub use future::{ClassifyResult, Embedding, NliLabel, NliResult};
pub use generate::{GenerateEvent, GenerateOptions, GenerateResponse};
pub use message::{Message, MessageContent, Role};
pub use model::{ModelCapability, ModelInfo, ModelStatus};
pub use options::{ChatOptions, ReasoningConfig, ReasoningEffort, ResponseFormat};
pub use response::{ChatEvent, ChatResponse, FinishReason, Usage};
pub use stance::{StanceLabel, StanceResult};
pub use token::Token;
pub use tool::{ToolCall, ToolChoice, ToolDefinition};
