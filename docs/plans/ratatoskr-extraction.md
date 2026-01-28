# Ratatoskr Extraction Plan

> Extracting chibi's OpenRouter API code into a standalone, vendor-ready crate.

## Executive Summary

This document details the extraction of LLM API communication code from chibi into ratatoskr, a new standalone crate. The goal is twofold:

1. **For chibi**: Reduce complexity by consolidating all API interaction to a single touchpoint
2. **For ratatoskr**: Create a useful, vendor-ready crate that other projects can embed

The extraction is substantial but clean. Chibi's API code is already reasonably modular, making the boundary clear. Post-extraction, chibi loses ~400 lines of HTTP/streaming code and gains a simpler, event-driven interface to LLM services.

---

## Current State: Chibi's API Surface

### Files Involved

| File | Lines | Purpose | Moves to Ratatoskr? |
|------|-------|---------|---------------------|
| `src/llm.rs` | 49 | HTTP client, streaming request | Yes (entirely) |
| `src/api/request.rs` | 152 | Request building, response parsing | Yes (partially) |
| `src/api/compact.rs` | 633 | Context compaction | No (uses ratatoskr) |
| `src/api/mod.rs` | 1011 | Main conversation loop | No (uses ratatoskr) |
| `src/config.rs` | ~200 | API parameter types | Yes (types only) |

### Code That Moves

#### 1. `src/llm.rs` (49 lines) - Moves Entirely

```rust
// Current: chibi-specific
pub struct ToolCallAccumulator {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub async fn send_streaming_request(
    config: &ResolvedConfig,
    request_body: serde_json::Value,
) -> io::Result<reqwest::Response>
```

This is low-level HTTP machinery. Ratatoskr absorbs it entirely and exposes a higher-level streaming interface instead.

#### 2. `src/api/request.rs` - Partial Move

**Moves to ratatoskr:**
- `build_request_body()` - 108 lines of request construction
- `extract_choice_content()` - 8 lines of response parsing

**Stays in chibi:**
- `PromptOptions` struct - chibi-specific options (verbose, json_output, etc.)

#### 3. `src/config.rs` - Types Move

These types define the API contract and belong in ratatoskr:

```rust
// All of these move to ratatoskr
pub enum ReasoningEffort { XHigh, High, Medium, Low, Minimal, None }

pub struct ReasoningConfig {
    pub effort: Option<ReasoningEffort>,
    pub max_tokens: Option<usize>,
    pub exclude: Option<bool>,
    pub enabled: Option<bool>,
}

pub enum ToolChoiceMode { Auto, None, Required }
pub struct ToolChoiceFunction { pub name: String }
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Function { type_: String, function: ToolChoiceFunction },
}

pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Option<serde_json::Value> },
}

pub struct ApiParams {
    pub prompt_caching: Option<bool>,
    pub reasoning: ReasoningConfig,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,
}
```

#### 4. SSE Parsing Loop - Moves (Embedded in `api/mod.rs`)

Lines 556-656 of `send_prompt_with_depth` contain the SSE parsing loop:

```rust
while let Some(chunk_result) = stream.next().await {
    // ... ~100 lines of:
    // - UTF-8 decoding
    // - "data: " prefix stripping
    // - "[DONE]" sentinel handling
    // - JSON parsing
    // - choices[0].delta.content extraction
    // - tool_calls array accumulation
    // - usage/model/id metadata capture
}
```

This is pure API parsing logic with no chibi semantics. It moves entirely to ratatoskr, becoming the implementation of `chat_stream()`.

---

## Target State: Ratatoskr Interface

### Core Types (ratatoskr-owned)

```rust
/// A chat message
pub struct Message {
    pub role: Role,
    pub content: String,
}

pub enum Role {
    System,
    User,
    Assistant,
    Tool { tool_call_id: String },
}

/// Tool definition for the API
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Options for chat requests
pub struct ChatOptions {
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,
    pub reasoning: Option<ReasoningConfig>,
    pub prompt_caching: Option<bool>,
}

/// Events emitted during streaming
pub enum ChatEvent {
    /// Text content chunk
    Content(String),

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

    /// Usage statistics (may arrive mid-stream or at end)
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },

    /// Model identifier (from response)
    Model(String),

    /// Request ID (from response)
    RequestId(String),

    /// Stream complete
    Done,
}

/// Non-streaming response
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

### Gateway Trait

```rust
#[async_trait]
pub trait ModelGateway: Send + Sync {
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

    /// List available models (provider-dependent)
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
}
```

### Builder Pattern

```rust
// Embedded mode (API clients only, no service needed)
let gateway = Ratatoskr::embedded()
    .with_openrouter(api_key)
    .base_url(custom_url)  // optional
    .build()?;

// Or with multiple providers
let gateway = Ratatoskr::embedded()
    .with_openrouter(openrouter_key)
    .with_anthropic(anthropic_key)
    .build()?;
```

---

## Chibi After Extraction

### The Single Touchpoint

Chibi will have exactly one point of contact with ratatoskr:

```rust
// In chibi's startup (main.rs or state.rs)
let gateway: Arc<dyn ModelGateway> = Arc::new(
    Ratatoskr::embedded()
        .with_openrouter(&config.api_key)
        .base_url(&config.base_url)
        .build()?
);

// Pass to AppState or wherever needed
```

### Adapter Layer

Chibi needs a thin adapter to convert its types:

```rust
// src/api/adapter.rs (new file, ~50 lines)

use ratatoskr::{Message as RMessage, Role, ToolDefinition, ChatOptions};

/// Convert chibi's message format to ratatoskr's
pub fn to_ratatoskr_messages(messages: &[serde_json::Value]) -> Vec<RMessage> {
    messages.iter().map(|m| {
        let role = match m["role"].as_str().unwrap_or("user") {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool {
                tool_call_id: m["tool_call_id"].as_str().unwrap_or("").to_string()
            },
            _ => Role::User,
        };
        RMessage {
            role,
            content: m["content"].as_str().unwrap_or("").to_string(),
        }
    }).collect()
}

/// Convert chibi's tool format to ratatoskr's
pub fn to_ratatoskr_tools(tools: &[serde_json::Value]) -> Vec<ToolDefinition> {
    tools.iter().filter_map(|t| {
        Some(ToolDefinition {
            name: t["function"]["name"].as_str()?.to_string(),
            description: t["function"]["description"].as_str().unwrap_or("").to_string(),
            parameters: t["function"]["parameters"].clone(),
        })
    }).collect()
}

/// Convert chibi's ResolvedConfig to ratatoskr's ChatOptions
pub fn to_chat_options(config: &ResolvedConfig) -> ChatOptions {
    ChatOptions {
        model: config.model.clone(),
        temperature: config.api.temperature,
        max_tokens: config.api.max_tokens,
        top_p: config.api.top_p,
        stop: config.api.stop.clone(),
        tool_choice: config.api.tool_choice.clone(),
        parallel_tool_calls: config.api.parallel_tool_calls,
        frequency_penalty: config.api.frequency_penalty,
        presence_penalty: config.api.presence_penalty,
        seed: config.api.seed,
        response_format: config.api.response_format.clone(),
        reasoning: if config.api.reasoning.is_empty() {
            None
        } else {
            Some(config.api.reasoning.clone())
        },
        prompt_caching: config.api.prompt_caching,
    }
}
```

### Refactored `send_prompt_with_depth`

The 737-line function shrinks significantly. The streaming loop becomes:

```rust
// Before: ~150 lines of SSE parsing
// After: ~40 lines of event handling

let messages = to_ratatoskr_messages(&messages);
let tools = to_ratatoskr_tools(&all_tools);
let options = to_chat_options(resolved_config);

let mut stream = gateway.chat_stream(&messages, Some(&tools), &options).await
    .map_err(|e| io::Error::other(format!("API error: {}", e)))?;

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
        ChatEvent::ToolCallStart { index, id, name } => {
            while tool_calls.len() <= index {
                tool_calls.push(ToolCall::default());
            }
            tool_calls[index].id = id;
            tool_calls[index].name = name;
        }
        ChatEvent::ToolCallDelta { index, arguments } => {
            tool_calls[index].arguments.push_str(&arguments);
        }
        ChatEvent::Usage { prompt_tokens, completion_tokens, .. } => {
            // Log if debug enabled
        }
        ChatEvent::Done => break,
        _ => {}
    }
}
```

### Refactored Compaction

The compaction code (`src/api/compact.rs`) has 6 HTTP request sites. All become:

```rust
// Before:
let client = Client::new();
let response = client
    .post(&resolved_config.base_url)
    .header(AUTHORIZATION, format!("Bearer {}", resolved_config.api_key))
    .header(CONTENT_TYPE, "application/json")
    .body(request_body.to_string())
    .send()
    .await?;
// ... error handling, JSON parsing

// After:
let response = gateway.chat(&messages, None, &options).await?;
let content = response.content;
```

---

## Ratatoskr Project Structure

```
ratatoskr/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── error.rs            # RatatoskrError enum
│   ├── types.rs            # Message, Role, ChatEvent, etc.
│   ├── options.rs          # ChatOptions, ToolChoice, ReasoningConfig
│   ├── gateway.rs          # ModelGateway trait
│   │
│   ├── embedded/           # Embedded mode (no service needed)
│   │   ├── mod.rs          # EmbeddedGateway impl
│   │   └── builder.rs      # Fluent builder
│   │
│   └── providers/          # Provider implementations
│       ├── mod.rs          # Provider trait
│       ├── openrouter.rs   # OpenRouter (SSE streaming)
│       └── anthropic.rs    # Direct Anthropic (future)
│
└── tests/
    ├── types.rs            # Unit tests for types
    ├── mock_provider.rs    # Mock for testing
    └── streaming.rs        # Stream parsing tests
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

[features]
default = ["openrouter"]
openrouter = []
anthropic = []
full = ["openrouter", "anthropic"]

[dependencies]
async-trait = "0.1"
futures-util = { version = "0.3", features = ["std"] }
pin-project-lite = "0.2"
reqwest = { version = "0.13", features = ["json", "stream"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
tokio = { version = "1.49", features = ["rt"] }

[dev-dependencies]
tokio = { version = "1.49", features = ["full", "test-util"] }
```

---

## Error Handling

Ratatoskr defines its own error type:

```rust
#[derive(Debug, thiserror::Error)]
pub enum RatatoskrError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    #[error("Invalid API key")]
    InvalidAuth,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("No provider configured")]
    NoProvider,
}
```

Chibi maps these to `io::Error` at the boundary:

```rust
gateway.chat_stream(...).await
    .map_err(|e| io::Error::other(e.to_string()))?
```

---

## Migration Path

### Phase 1: Create Ratatoskr (Independent)

1. Set up crate structure
2. Define types (`Message`, `ChatEvent`, `ChatOptions`, etc.)
3. Implement OpenRouter provider with SSE parsing
4. Write tests with mock responses
5. Publish to crates.io (or keep private initially)

**Deliverable**: Working `ratatoskr` crate that can be used standalone.

### Phase 2: Integrate into Chibi

1. Add `ratatoskr` dependency to chibi
2. Create adapter module (`src/api/adapter.rs`)
3. Refactor `send_prompt_with_depth` to use gateway
4. Refactor compaction functions to use gateway
5. Remove `src/llm.rs` entirely
6. Remove moved types from `src/config.rs` (import from ratatoskr)
7. Update tests

**Deliverable**: Chibi using ratatoskr, with ~400 fewer lines.

### Phase 3: Cleanup and Polish

1. Remove dead code
2. Update CLAUDE.md
3. Add integration tests
4. Document the adapter layer

---

## Lines Changed Summary

### Ratatoskr (New Code)

| File | Lines | Notes |
|------|-------|-------|
| `src/lib.rs` | ~30 | Re-exports |
| `src/types.rs` | ~150 | Core types |
| `src/options.rs` | ~120 | ChatOptions, moved from chibi |
| `src/error.rs` | ~50 | Error enum |
| `src/gateway.rs` | ~30 | Trait definition |
| `src/embedded/mod.rs` | ~100 | Gateway implementation |
| `src/embedded/builder.rs` | ~80 | Builder pattern |
| `src/providers/openrouter.rs` | ~200 | SSE streaming, request building |
| **Total** | **~760** | |

### Chibi Changes

| File | Before | After | Delta |
|------|--------|-------|-------|
| `src/llm.rs` | 49 | 0 | -49 (deleted) |
| `src/api/request.rs` | 152 | ~30 | -122 |
| `src/api/mod.rs` | 1011 | ~900 | -111 |
| `src/api/compact.rs` | 633 | ~550 | -83 |
| `src/config.rs` | ~600 | ~450 | -150 |
| `src/api/adapter.rs` | 0 | ~60 | +60 |
| **Net** | | | **-455** |

---

## Open Questions

### 1. Tool Definition Format

Should ratatoskr accept OpenAI-format tool JSON directly, or define its own `ToolDefinition` struct?

**Option A**: Own struct (cleaner API, validates at compile time)
**Option B**: Accept `serde_json::Value` (easier migration, more flexible)

**Recommendation**: Option A with a `From<serde_json::Value>` impl for migration.

### 2. Message Content Types

OpenRouter supports multi-modal content (text + images). Should ratatoskr support this in v0.1?

**Recommendation**: Start with text-only `String` content. Add `Content` enum later:
```rust
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}
```

### 3. Streaming Backpressure

Should the stream apply backpressure if the consumer is slow?

**Recommendation**: Let tokio's channel handle it naturally. Document that consumers should process events promptly.

### 4. Retry Logic

Should ratatoskr handle retries internally (for rate limits, transient errors)?

**Recommendation**: Not in v0.1. Return `RatatoskrError::RateLimited` and let the caller decide. Add optional retry middleware later.

### 5. Connection Pooling

Should ratatoskr reuse HTTP connections across requests?

**Recommendation**: Yes. Use `reqwest::Client` stored in the gateway, not created per-request. This is a bug fix relative to chibi's current code.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SSE parsing differences across providers | Medium | Test against real API responses, not just OpenRouter docs |
| Breaking chibi during migration | High | Keep old code paths working until ratatoskr integration is complete |
| Ratatoskr API churn affecting chibi | Medium | Stabilize core types before chibi integration |
| Performance regression | Low | Benchmark streaming throughput before/after |

---

## Success Criteria

1. **Ratatoskr works standalone**: Can be used by a new project without chibi knowledge
2. **Chibi functionality unchanged**: All existing features work identically
3. **Single touchpoint achieved**: Chibi has one gateway instance, no direct HTTP calls
4. **Net code reduction**: Chibi shrinks by 400+ lines
5. **Tests pass**: Both ratatoskr and chibi test suites green

---

## Timeline Considerations

This extraction is blocked on chibi's next release (per user request). The work can be staged:

1. **Now**: Finalize this plan, identify any blockers
2. **Pre-release**: Build ratatoskr in parallel, test independently
3. **Post-release**: Integrate ratatoskr into chibi, cut new release

The parallel development approach means ratatoskr can be mature and tested before chibi depends on it.
