# Ratatoskr Phase 1: OpenRouter Chat Gateway

> Extract chibi's LLM API code into a standalone crate with a forward-compatible interface.

## Goals

1. **Working embedded-mode chat gateway** with OpenRouter as the first provider
2. **Forward-compatible trait design** that won't break when adding embeddings, NLI, etc.
3. **Clean extraction from chibi** reducing chibi's API code by ~400 lines
4. **Vendor-ready crate** usable by any Rust project for quick LLM access

## Non-Goals (Phase 1)

- Service mode (gRPC/socket server)
- Local inference (ONNX)
- Hugging Face Inference API provider
- Routing/fallback logic
- Caching
- Model lifecycle management

---

## Interface Design

### Core Trait

The trait is designed for the full ratatoskr vision, with Phase 1 implementing only chat methods:

```rust
#[async_trait]
pub trait ModelGateway: Send + Sync {
    // ===== Phase 1: Implement fully =====

    /// Streaming chat completion
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>;

    /// Non-streaming chat completion
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// List available models from configured providers
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Query what this gateway instance can do
    fn capabilities(&self) -> Capabilities;

    // ===== Phase 2+: Default to NotImplemented =====

    /// Generate embeddings for text
    async fn embed(&self, text: &str) -> Result<Embedding> {
        Err(RatatoskrError::NotImplemented("embed"))
    }

    /// Batch embedding generation
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Err(RatatoskrError::NotImplemented("embed_batch"))
    }

    /// Natural language inference (entailment/contradiction/neutral)
    async fn infer_nli(&self, premise: &str, hypothesis: &str) -> Result<NliResult> {
        Err(RatatoskrError::NotImplemented("infer_nli"))
    }

    /// Stance classification relative to a target
    async fn classify_stance(&self, text: &str, target: &str) -> Result<StanceResult> {
        Err(RatatoskrError::NotImplemented("classify_stance"))
    }

    /// Zero-shot classification with custom labels
    async fn classify_zero_shot(&self, text: &str, labels: &[&str]) -> Result<ClassifyResult> {
        Err(RatatoskrError::NotImplemented("classify_zero_shot"))
    }

    /// Count tokens for a given model (local tokenizer)
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        Err(RatatoskrError::NotImplemented("count_tokens"))
    }
}
```

### Capabilities Introspection

```rust
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub chat: bool,
    pub chat_streaming: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub stance: bool,
    pub zero_shot: bool,
    pub token_counting: bool,
}

impl Capabilities {
    /// Phase 1 embedded gateway capabilities
    pub fn chat_only() -> Self {
        Self {
            chat: true,
            chat_streaming: true,
            ..Default::default()
        }
    }
}
```

This allows consumers (like örlög) to check capabilities before calling methods:

```rust
if !gateway.capabilities().nli {
    // Fall back to LLM-based NLI or error
}
```

---

## Core Types

### Messages

```rust
/// A chat message
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    pub tool_calls: Option<Vec<ToolCall>>,  // Present in assistant messages with tool use
    pub name: Option<String>,                // Optional name for multi-agent scenarios
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool { tool_call_id: String },
}

/// Message content (extensible for future multi-modal support)
#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    // Phase 2+: Parts(Vec<ContentPart>) for images, etc.
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }

    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.map(MessageContent::Text).unwrap_or(MessageContent::Text(String::new())),
            tool_calls: Some(tool_calls),
            name: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool { tool_call_id: tool_call_id.into() },
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            name: None,
        }
    }
}
```

### Tool Definitions

```rust
/// Tool definition for function calling
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}

/// A tool call made by the model
#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,  // JSON string
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// For migration: convert from OpenAI-format JSON
impl TryFrom<&serde_json::Value> for ToolDefinition {
    type Error = RatatoskrError;

    fn try_from(value: &serde_json::Value) -> Result<Self> {
        let function = value.get("function")
            .ok_or_else(|| RatatoskrError::InvalidInput("missing 'function' field".into()))?;

        Ok(Self {
            name: function["name"].as_str()
                .ok_or_else(|| RatatoskrError::InvalidInput("missing function name".into()))?
                .to_string(),
            description: function["description"].as_str().unwrap_or("").to_string(),
            parameters: function["parameters"].clone(),
        })
    }
}
```

### Chat Options

```rust
/// Options for chat requests (provider-agnostic)
#[derive(Debug, Clone, Default)]
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
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<ResponseFormat>,

    /// Provider-specific options (escape hatch)
    /// Keys like "openrouter.prompt_caching", "openrouter.reasoning", etc.
    pub provider_options: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}
```

Provider-specific options like OpenRouter's `reasoning` and `prompt_caching` go in `provider_options`:

```rust
let options = ChatOptions {
    model: "anthropic/claude-sonnet-4".into(),
    provider_options: Some(serde_json::json!({
        "openrouter": {
            "prompt_caching": true,
            "reasoning": { "effort": "high" }
        }
    })),
    ..Default::default()
};
```

### Streaming Events

```rust
/// Events emitted during streaming chat
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Text content chunk
    Content(String),

    /// Reasoning/thinking content (extended thinking models)
    Reasoning(String),

    /// Start of a tool call
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },

    /// Incremental tool call arguments
    ToolCallDelta {
        index: usize,
        arguments: String,
    },

    /// Usage statistics
    Usage(Usage),

    /// Model identifier (from response, may differ from requested)
    Model(String),

    /// Request ID for debugging/logging
    RequestId(String),

    /// Stream complete
    Done,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
}
```

### Response Types

```rust
/// Non-streaming chat response
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub request_id: Option<String>,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub provider: String,
    pub context_window: Option<usize>,
    pub capabilities: Vec<String>,  // "chat", "vision", "function_calling", etc.
}
```

### Future Types (stubbed)

```rust
/// Embedding result (Phase 2+)
#[derive(Debug, Clone)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub model: String,
    pub dimensions: usize,
}

/// NLI result (Phase 2+)
#[derive(Debug, Clone)]
pub struct NliResult {
    pub entailment: f32,
    pub contradiction: f32,
    pub neutral: f32,
    pub label: NliLabel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NliLabel {
    Entailment,
    Contradiction,
    Neutral,
}

/// Stance detection result (Phase 2+)
#[derive(Debug, Clone)]
pub struct StanceResult {
    pub favor: f32,
    pub against: f32,
    pub neutral: f32,
    pub label: StanceLabel,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StanceLabel {
    Favor,
    Against,
    Neutral,
}

/// Zero-shot classification result (Phase 2+)
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub scores: std::collections::HashMap<String, f32>,
    pub top_label: String,
    pub confidence: f32,
}
```

---

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum RatatoskrError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<std::time::Duration> },

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("No provider configured")]
    NoProvider,

    #[error("Operation not implemented: {0}")]
    NotImplemented(&'static str),

    #[error("Provider does not support this operation")]
    Unsupported,
}

pub type Result<T> = std::result::Result<T, RatatoskrError>;
```

---

## Builder Pattern

```rust
use ratatoskr::{Ratatoskr, ChatOptions, Message};

// Embedded mode construction
let gateway = Ratatoskr::embedded()
    .with_openrouter("sk-or-...")
    .base_url("https://openrouter.ai/api/v1")  // optional, has default
    .build()?;

// Usage
let response = gateway.chat(
    &[
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ],
    None,  // no tools
    &ChatOptions {
        model: "anthropic/claude-sonnet-4".into(),
        ..Default::default()
    },
).await?;

println!("{}", response.content);

// Streaming
let mut stream = gateway.chat_stream(&messages, None, &options).await?;
while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::Content(text) => print!("{}", text),
        ChatEvent::Done => break,
        _ => {}
    }
}
```

---

## Project Structure

```
ratatoskr/
├── Cargo.toml
├── CLAUDE.md                # Project memory
├── src/
│   ├── lib.rs               # Public API re-exports
│   ├── error.rs             # RatatoskrError
│   ├── traits.rs            # ModelGateway trait
│   ├── types/
│   │   ├── mod.rs           # Re-exports
│   │   ├── message.rs       # Message, Role, MessageContent
│   │   ├── tool.rs          # ToolDefinition, ToolCall, ToolChoice
│   │   ├── options.rs       # ChatOptions, ResponseFormat
│   │   ├── response.rs      # ChatResponse, ChatEvent, Usage
│   │   ├── model.rs         # ModelInfo, Capabilities
│   │   └── future.rs        # Embedding, NliResult, StanceResult (stubbed)
│   │
│   ├── embedded/
│   │   ├── mod.rs           # EmbeddedGateway impl
│   │   └── builder.rs       # Fluent builder
│   │
│   └── providers/
│       ├── mod.rs           # Provider trait
│       └── openrouter.rs    # OpenRouter implementation
│
└── tests/
    ├── types_test.rs        # Unit tests for type conversions
    ├── builder_test.rs      # Builder pattern tests
    └── openrouter_test.rs   # Integration tests (with recorded responses)
```

### Cargo.toml

```toml
[package]
name = "ratatoskr"
version = "0.1.0"
edition = "2024"
description = "Unified model gateway for LLM APIs"
license = "MIT OR Apache-2.0"
repository = "https://github.com/emesal/ratatoskr"
keywords = ["llm", "ai", "openrouter", "api", "gateway"]
categories = ["api-bindings", "asynchronous"]

[features]
default = ["openrouter"]
openrouter = []
huggingface = []    # Phase 2
ollama = []         # Phase 2
onnx = ["ort"]      # Phase 3
service = ["tonic", "prost"] # Phase 3
full = ["openrouter", "huggingface", "ollama", "onnx", "service"]

[dependencies]
async-trait = "0.1"
futures-util = { version = "0.3", default-features = false, features = ["std"] }
pin-project-lite = "0.2"
reqwest = { version = "0.12", default-features = false, features = ["json", "stream", "rustls-tls"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
tokio = { version = "1", features = ["rt"] }

# Phase 2+
# ort = { version = "2", optional = true }
# tonic = { version = "0.12", optional = true }
# prost = { version = "0.13", optional = true }

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
tokio-test = "0.4"
```

---

## Code to Extract from Chibi

### Files Affected

| Chibi File | Lines | Action |
|------------|-------|--------|
| `src/llm.rs` | 49 | Delete entirely |
| `src/api/request.rs` | 154 | Remove `build_request_body`, `extract_choice_content`; keep `PromptOptions` |
| `src/api/mod.rs` | ~1000 | Replace SSE parsing loop (~100 lines) with event handling (~40 lines) |
| `src/api/compact.rs` | 633 | Replace 6 HTTP call sites with `gateway.chat()` |
| `src/config.rs` | ~600 | Remove `ReasoningConfig`, `ToolChoice`, `ResponseFormat` (import from ratatoskr) |

### New Chibi Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/api/adapter.rs` | ~120 | Convert chibi types ↔ ratatoskr types |

### Net Change

- Ratatoskr: ~800 new lines
- Chibi: ~400 fewer lines
- Total ecosystem: +400 lines, but with clean separation

---

## Chibi Adapter Layer

```rust
// src/api/adapter.rs

use ratatoskr::{
    Message as RMessage, MessageContent, Role, ToolDefinition,
    ChatOptions, ToolChoice, ResponseFormat, ChatEvent, ToolCall,
};
use crate::config::ResolvedConfig;

/// Convert chibi's JSON messages to ratatoskr Messages
pub fn to_ratatoskr_messages(messages: &[serde_json::Value]) -> Vec<RMessage> {
    messages.iter().filter_map(|m| {
        let role = match m["role"].as_str()? {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool {
                tool_call_id: m["tool_call_id"].as_str().unwrap_or("").to_string()
            },
            _ => return None,
        };

        let content = m["content"].as_str().unwrap_or("").to_string();

        // Handle assistant tool_calls
        let tool_calls = m.get("tool_calls").and_then(|tc| {
            tc.as_array().map(|arr| {
                arr.iter().filter_map(|t| {
                    Some(ToolCall {
                        id: t["id"].as_str()?.to_string(),
                        name: t["function"]["name"].as_str()?.to_string(),
                        arguments: t["function"]["arguments"].as_str()?.to_string(),
                    })
                }).collect()
            })
        });

        Some(RMessage {
            role,
            content: MessageContent::Text(content),
            tool_calls,
            name: m["name"].as_str().map(String::from),
        })
    }).collect()
}

/// Convert chibi's JSON tools to ratatoskr ToolDefinitions
pub fn to_ratatoskr_tools(tools: &[serde_json::Value]) -> Vec<ToolDefinition> {
    tools.iter().filter_map(|t| {
        ToolDefinition::try_from(t).ok()
    }).collect()
}

/// Convert ResolvedConfig to ChatOptions
pub fn to_chat_options(config: &ResolvedConfig) -> ChatOptions {
    let mut options = ChatOptions {
        model: config.model.clone(),
        temperature: config.api.temperature,
        max_tokens: config.api.max_tokens,
        top_p: config.api.top_p,
        stop: config.api.stop.clone(),
        frequency_penalty: config.api.frequency_penalty,
        presence_penalty: config.api.presence_penalty,
        seed: config.api.seed,
        parallel_tool_calls: config.api.parallel_tool_calls,
        ..Default::default()
    };

    // Tool choice
    if let Some(ref tc) = config.api.tool_choice {
        options.tool_choice = Some(match tc {
            crate::config::ToolChoice::Mode(mode) => match mode {
                crate::config::ToolChoiceMode::Auto => ToolChoice::Auto,
                crate::config::ToolChoiceMode::None => ToolChoice::None,
                crate::config::ToolChoiceMode::Required => ToolChoice::Required,
            },
            crate::config::ToolChoice::Function { function, .. } => {
                ToolChoice::Function { name: function.name.clone() }
            }
        });
    }

    // Response format
    if let Some(ref format) = config.api.response_format {
        options.response_format = Some(match format {
            crate::config::ResponseFormat::Text => ResponseFormat::Text,
            crate::config::ResponseFormat::JsonObject => ResponseFormat::JsonObject,
            crate::config::ResponseFormat::JsonSchema { json_schema } => {
                ResponseFormat::JsonSchema {
                    schema: json_schema.clone().unwrap_or(serde_json::json!({}))
                }
            }
        });
    }

    // Provider-specific options
    let mut provider_opts = serde_json::Map::new();
    let mut openrouter_opts = serde_json::Map::new();

    if let Some(caching) = config.api.prompt_caching {
        openrouter_opts.insert("prompt_caching".into(), caching.into());
    }

    if !config.api.reasoning.is_empty() {
        let mut reasoning = serde_json::Map::new();
        if let Some(ref effort) = config.api.reasoning.effort {
            reasoning.insert("effort".into(), effort.as_str().into());
        }
        if let Some(max_tokens) = config.api.reasoning.max_tokens {
            reasoning.insert("max_tokens".into(), max_tokens.into());
        }
        if let Some(exclude) = config.api.reasoning.exclude {
            reasoning.insert("exclude".into(), exclude.into());
        }
        if let Some(enabled) = config.api.reasoning.enabled {
            reasoning.insert("enabled".into(), enabled.into());
        }
        openrouter_opts.insert("reasoning".into(), reasoning.into());
    }

    if !openrouter_opts.is_empty() {
        provider_opts.insert("openrouter".into(), openrouter_opts.into());
        options.provider_options = Some(provider_opts.into());
    }

    options
}
```

---

## Refactored Chibi Streaming Loop

Before (in `send_prompt_with_depth`, ~100 lines of SSE parsing):

```rust
while let Some(chunk_result) = stream.next().await {
    // UTF-8 decoding, "data: " stripping, JSON parsing,
    // delta extraction, tool call accumulation...
}
```

After (~40 lines of event handling):

```rust
let messages = to_ratatoskr_messages(&messages);
let tools = to_ratatoskr_tools(&all_tools);
let options = to_chat_options(resolved_config);

let mut stream = gateway.chat_stream(&messages, Some(&tools), &options).await
    .map_err(|e| io::Error::other(e.to_string()))?;

let mut full_response = String::new();
let mut tool_calls: Vec<ToolCall> = Vec::new();

while let Some(event) = stream.next().await {
    match event.map_err(|e| io::Error::other(e.to_string()))? {
        ChatEvent::Content(text) => {
            full_response.push_str(&text);
            if !json_mode {
                stdout.write_all(text.as_bytes()).await?;
                stdout.flush().await?;
            }
        }
        ChatEvent::Reasoning(text) => {
            // Handle extended thinking output if needed
            if verbose {
                // Log or display reasoning
            }
        }
        ChatEvent::ToolCallStart { index, id, name } => {
            while tool_calls.len() <= index {
                tool_calls.push(ToolCall::default());
            }
            tool_calls[index].id = id;
            tool_calls[index].name = name;
        }
        ChatEvent::ToolCallDelta { index, arguments } => {
            if let Some(tc) = tool_calls.get_mut(index) {
                tc.arguments.push_str(&arguments);
            }
        }
        ChatEvent::Usage(usage) => {
            if let Some(debug) = options.debug {
                debug_log!(debug, "tokens: {} prompt, {} completion",
                    usage.prompt_tokens, usage.completion_tokens);
            }
        }
        ChatEvent::Done => break,
        _ => {}
    }
}
```

---

## Migration Steps

### Step 1: Build Ratatoskr (Independent)

1. Set up crate structure
2. Implement types (Message, ChatEvent, etc.)
3. Implement OpenRouter provider with SSE parsing
4. Write unit tests for types
5. Write integration tests with recorded API responses
6. Document public API

**Deliverable**: Working `ratatoskr` crate, usable standalone.

### Step 2: Integrate into Chibi

1. Add `ratatoskr` as a path dependency
2. Create `src/api/adapter.rs`
3. Initialize gateway in `main.rs` or state module
4. Refactor `send_prompt_with_depth` to use gateway
5. Refactor compaction code to use `gateway.chat()`
6. Delete `src/llm.rs`
7. Remove migrated types from `src/config.rs`
8. Update tests

**Deliverable**: Chibi using ratatoskr, ~400 fewer lines.

### Step 3: Cleanup

1. Remove dead code
2. Update chibi's CLAUDE.md
3. Add integration tests
4. Verify all existing chibi tests pass

---

## Testing Strategy

### Unit Tests

- Type conversions (`TryFrom<serde_json::Value>` for tools)
- Message builder methods
- ChatOptions serialization
- Error type behavior

### Integration Tests (Recorded Responses)

Record real OpenRouter responses and replay them:

```rust
#[tokio::test]
async fn test_chat_basic() {
    let mock_server = MockServer::start().await;
    mock_server.register(Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(recorded_response())));

    let gateway = Ratatoskr::embedded()
        .with_openrouter("test-key")
        .base_url(&mock_server.uri())
        .build()
        .unwrap();

    let response = gateway.chat(&[Message::user("Hello")], None, &options).await;
    assert!(response.is_ok());
}
```

### Live Tests (Optional, CI-excluded)

For manual verification against real API:

```rust
#[tokio::test]
#[ignore]  // Run manually with: cargo test -- --ignored
async fn test_live_openrouter() {
    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
    // ...
}
```

---

## Success Criteria

1. **Ratatoskr works standalone**: New project can `cargo add ratatoskr` and chat with OpenRouter
2. **Chibi functions identically**: All existing features work, all tests pass
3. **Single gateway touchpoint**: Chibi has one `Arc<dyn ModelGateway>`, no direct HTTP
4. **Forward-compatible trait**: Adding `embed()` later doesn't break existing code
5. **Net code reduction**: Chibi shrinks by 350+ lines
6. **Clean provider abstraction**: Adding Hugging Face later is straightforward

---

## Open Questions (to resolve during implementation)

1. **Connection pooling scope**: One `reqwest::Client` per gateway, or per provider? (Recommend: per gateway, shared across providers)

2. **Retry policy**: Should Phase 1 include any automatic retry for transient failures, or leave entirely to caller? (Recommend: no retry in Phase 1, add as opt-in later)

3. **Timeout configuration**: Expose in `ChatOptions` or builder-level only? (Recommend: builder-level default, per-request override in `provider_options`)

4. **Model validation**: Should `list_models()` cache results? Should `chat()` validate model exists first? (Recommend: no validation in Phase 1, caller's responsibility)
