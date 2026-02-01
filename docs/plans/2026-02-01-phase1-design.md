# Ratatoskr Phase 1 Design

> standalone MVP with openrouter, built on `llm` crate

## overview

ratatoskr is a thin abstraction layer providing a stable `ModelGateway` trait for consumers (chibi, orlog), while delegating provider work to the `llm` crate internally.

```
┌─────────────────────────────────────────────────┐
│              consumers (chibi, orlog)           │
└─────────────────────┬───────────────────────────┘
                      │ uses
┌─────────────────────▼───────────────────────────┐
│           ratatoskr::ModelGateway               │
│         (our stable public trait)               │
└─────────────────────┬───────────────────────────┘
                      │ implemented by
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ Embedded  │  │   ONNX    │  │  Service  │
│  Gateway  │  │ Provider  │  │  Client   │
│ (phase 1) │  │ (phase 4) │  │ (phase 5) │
└─────┬─────┘  └───────────┘  └───────────┘
      │ wraps
┌─────▼─────┐
│ llm crate │ ◄── handles openrouter, anthropic,
└───────────┘     ollama, huggingface, etc.
```

**key principle**: consumers only see `ratatoskr::ModelGateway`. the `llm` crate is an implementation detail.

## the `ModelGateway` trait

```rust
#[async_trait]
pub trait ModelGateway: Send + Sync {
    // ===== phase 1: chat =====

    /// streaming chat completion
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>;

    /// non-streaming chat completion
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// what can this gateway do?
    fn capabilities(&self) -> Capabilities;

    // ===== phase 3+: stubs with default impls =====

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        Err(RatatoskrError::NotImplemented("embed"))
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        Err(RatatoskrError::NotImplemented("embed_batch"))
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        Err(RatatoskrError::NotImplemented("infer_nli"))
    }

    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult> {
        Err(RatatoskrError::NotImplemented("classify_zero_shot"))
    }

    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        Err(RatatoskrError::NotImplemented("count_tokens"))
    }
}
```

## core types

### messages

```rust
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub name: Option<String>,
}

pub enum Role {
    System,
    User,
    Assistant,
    Tool { tool_call_id: String },
}

pub enum MessageContent {
    Text(String),
    // future: Parts(Vec<ContentPart>) for multimodal
}
```

### tools

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,  // JSON string
}
```

### streaming events

```rust
pub enum ChatEvent {
    Content(String),
    Reasoning(String),
    ToolCallStart { index: usize, id: String, name: String },
    ToolCallDelta { index: usize, arguments: String },
    Usage(Usage),
    Done,
}

pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
}
```

### responses

```rust
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub finish_reason: FinishReason,
}

pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}
```

### capabilities

```rust
#[derive(Default)]
pub struct Capabilities {
    pub chat: bool,
    pub chat_streaming: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub classification: bool,
    pub token_counting: bool,
}
```

## chat options

```rust
#[derive(Default)]
pub struct ChatOptions {
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: Option<ResponseFormat>,

    // normalized cross-provider options
    pub cache_prompt: Option<bool>,
    pub reasoning: Option<ReasoningConfig>,

    // escape hatch for truly provider-specific options
    pub raw_provider_options: Option<serde_json::Value>,
}

#[derive(Default)]
pub struct ReasoningConfig {
    pub effort: Option<ReasoningEffort>,
    pub max_tokens: Option<usize>,
    pub exclude_from_output: Option<bool>,
}

pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}
```

## error handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum RatatoskrError {
    // provider/network errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    #[error("authentication failed")]
    AuthenticationFailed,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    // streaming errors
    #[error("stream error: {0}")]
    Stream(String),

    // data errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    // configuration errors
    #[error("no provider configured")]
    NoProvider,

    #[error("operation not implemented: {0}")]
    NotImplemented(&'static str),

    #[error("provider does not support this operation")]
    Unsupported,

    // soft errors (API succeeded but response is problematic)
    #[error("empty response from model")]
    EmptyResponse,

    #[error("content filtered: {reason}")]
    ContentFiltered { reason: String },

    #[error("context length exceeded: {limit} tokens")]
    ContextLengthExceeded { limit: usize },
}

pub type Result<T> = std::result::Result<T, RatatoskrError>;
```

## project structure

```
ratatoskr/
├── Cargo.toml
├── CLAUDE.md
├── src/
│   ├── lib.rs                 # public API re-exports
│   ├── error.rs               # RatatoskrError, Result
│   ├── traits.rs              # ModelGateway trait
│   │
│   ├── types/
│   │   ├── mod.rs             # re-exports
│   │   ├── message.rs         # Message, Role, MessageContent
│   │   ├── tool.rs            # ToolDefinition, ToolCall, ToolChoice
│   │   ├── options.rs         # ChatOptions, ReasoningConfig
│   │   ├── response.rs        # ChatResponse, ChatEvent, FinishReason, Usage
│   │   ├── capabilities.rs    # Capabilities
│   │   └── future.rs          # Embedding, NliResult, ClassifyResult (stubs)
│   │
│   ├── gateway/
│   │   ├── mod.rs             # re-exports
│   │   ├── embedded.rs        # EmbeddedGateway (wraps llm crate)
│   │   └── builder.rs         # Ratatoskr::builder()
│   │
│   └── convert/
│       └── mod.rs             # ratatoskr types <-> llm crate types
│
└── tests/
    ├── types_test.rs
    ├── builder_test.rs
    └── integration/
        └── chat_test.rs
```

## dependencies (Cargo.toml)

```toml
[package]
name = "ratatoskr"
version = "0.1.0"
edition = "2024"

[dependencies]
async-trait = "0.1"
futures-util = { version = "0.3", default-features = false, features = ["std"] }
llm = { version = "1.3", features = [
    "openai",      # openrouter, openai, azure, etc.
    "anthropic",   # direct anthropic
    "ollama",      # local models
    "google",      # gemini
] }
pin-project-lite = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
tokio = { version = "1", features = ["rt"] }

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
wiremock = "0.6"
```

## builder pattern

```rust
let gateway = Ratatoskr::builder()
    .openrouter(api_key)
    .anthropic(anthropic_key)      // optional
    .ollama("http://localhost:11434")  // optional
    .build()?;

// model string determines provider routing:
// "anthropic/claude-sonnet-4" -> openrouter
// "claude-sonnet-4" -> direct anthropic (if configured)
// "llama3:8b" -> ollama
```

## testing strategy

1. **unit tests** — types, conversions, builder logic (fast, no I/O)
2. **integration tests** — wiremock with recorded responses (deterministic, tests real parsing)
3. **live tests** — `#[ignore]` by default, run manually to verify against real APIs

full test coverage is required.

## phase 1 deliverables

- [ ] `ratatoskr` crate with `ModelGateway` trait
- [ ] `EmbeddedGateway` implementation wrapping `llm` crate
- [ ] builder pattern for configuration
- [ ] full type system (messages, tools, events, options, errors)
- [ ] stub methods for future capabilities (embed, NLI, etc.)
- [ ] comprehensive test suite with full coverage
- [ ] clean `cargo doc` with examples
- [ ] `CLAUDE.md` documents architecture and conventions

## not in scope

- chibi integration (chibi's roadmap)
- embeddings implementation (phase 3)
- NLI/classification implementation (phase 3)
- ONNX local inference (phase 4)
- service mode (phase 5)

## references

- [llm crate](https://github.com/graniet/llm) — underlying provider abstraction
- [phase-1-plan.md](./phase-1-plan.md) — earlier detailed plan (some parts superseded)
- [ROADMAP.md](./ROADMAP.md) — full multi-phase roadmap
