# API Reference

Quick reference for Ratatoskr's public API.

## Gateway

### Ratatoskr::builder()

Creates a new gateway builder.

```rust
let gateway = Ratatoskr::builder()
    .openrouter(key)
    .anthropic(key)
    .openai(key)
    .google(key)
    .ollama(url)
    .timeout(seconds)
    .build()?;
```

At least one provider must be configured or `build()` returns `NoProvider` error.

## ModelGateway Trait

### Chat

```rust
async fn chat(&self, messages: &[Message], tools: Option<&[ToolDefinition]>, options: &ChatOptions) -> Result<ChatResponse>
async fn chat_stream(&self, ...) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>
```

### Generate

```rust
async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse>
async fn generate_stream(&self, ...) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>>
```

### Embeddings

```rust
async fn embed(&self, text: &str, model: &str) -> Result<Embedding>
async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>>
```

### NLI

```rust
async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>
async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>>
```

### Classification

```rust
async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult>
async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult>
```

### Tokenization

```rust
fn count_tokens(&self, text: &str, model: &str) -> Result<usize>
fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>>
```

### Introspection

```rust
fn capabilities(&self) -> Capabilities
fn list_models(&self) -> Vec<ModelInfo>
fn model_status(&self, model: &str) -> ModelStatus
```

## Types

### Message

```rust
// Constructors
Message::system(content)
Message::user(content)
Message::assistant(content)
Message::assistant_with_tool_calls(content: Option<&str>, tool_calls: Vec<ToolCall>)
Message::tool_result(tool_call_id, result)

// Modifier
message.with_name(name)  // For multi-agent scenarios

// Fields
pub role: Role
pub content: MessageContent
pub tool_calls: Option<Vec<ToolCall>>
pub name: Option<String>
```

### Role

```rust
pub enum Role {
    System,
    User,
    Assistant,
    Tool { tool_call_id: String },
}
```

### MessageContent

```rust
pub enum MessageContent {
    Text(String),
    // Future: Image, Audio, etc.
}
```

### ChatOptions

```rust
ChatOptions::default()
    .model(name)                    // Required
    .temperature(0.0..2.0)
    .max_tokens(usize)
    .top_p(0.0..1.0)
    .stop(vec!["stop", "sequences"])
    .frequency_penalty(-2.0..2.0)
    .presence_penalty(-2.0..2.0)
    .seed(u64)
    .tool_choice(ToolChoice)
    .response_format(ResponseFormat)
    .cache_prompt(bool)
    .reasoning(ReasoningConfig)
    .raw_provider_options(serde_json::Value)
```

### ToolChoice

```rust
pub enum ToolChoice {
    Auto,                           // Model decides
    None,                           // No tools
    Required,                       // Must use a tool
    Function { name: String },      // Use specific tool
}
```

### ResponseFormat

```rust
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}
```

### ReasoningConfig

```rust
pub struct ReasoningConfig {
    pub effort: Option<ReasoningEffort>,      // Low, Medium, High
    pub max_tokens: Option<usize>,
    pub exclude_from_output: Option<bool>,
}
```

### ChatResponse

```rust
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub finish_reason: FinishReason,
}
```

### ChatEvent

```rust
pub enum ChatEvent {
    Content(String),
    Reasoning(String),
    ToolCallStart { index: usize, id: String, name: String },
    ToolCallDelta { index: usize, arguments: String },
    Usage(Usage),
    Done,
}
```

### FinishReason

```rust
pub enum FinishReason {
    Stop,           // Normal completion
    Length,         // Hit max_tokens
    ToolCalls,      // Made tool calls
    ContentFilter,  // Blocked by policy
}
```

### Usage

```rust
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
}
```

### ToolDefinition

```rust
ToolDefinition::new(name, description, parameters_schema)

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}
```

### ToolCall

```rust
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,  // JSON string
}

// Methods
tool_call.parse_arguments::<T>()?  // Deserialize arguments
```

### Capabilities

```rust
pub struct Capabilities {
    pub chat: bool,
    pub chat_streaming: bool,
    pub generate: bool,
    pub tool_use: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub classification: bool,
    pub stance: bool,
    pub token_counting: bool,
    pub local_inference: bool,
}

// Constructors
Capabilities::chat_only()
Capabilities::full()
```

### Embedding

```rust
pub struct Embedding {
    pub values: Vec<f32>,
    pub model: String,
    pub dimensions: usize,
}
```

### NliResult

```rust
pub struct NliResult {
    pub entailment: f32,
    pub contradiction: f32,
    pub neutral: f32,
    pub label: NliLabel,  // Entailment | Contradiction | Neutral
}
```

### StanceResult

```rust
pub struct StanceResult {
    pub favor: f32,
    pub against: f32,
    pub neutral: f32,
    pub label: StanceLabel,  // Favor | Against | Neutral
    pub target: String,
}
```

### Token

```rust
pub struct Token {
    pub id: u32,
    pub text: String,
    pub start: usize,  // byte offset
    pub end: usize,
}
```

### ModelInfo

```rust
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub capabilities: Vec<ModelCapability>,
    pub context_window: Option<usize>,
    pub dimensions: Option<usize>,  // for embedding models
}
```

### ModelStatus

```rust
pub enum ModelStatus {
    Available,
    Loading,
    Ready,
    Unavailable { reason: String },
}
```

## Errors

### RatatoskrError

```rust
pub enum RatatoskrError {
    // Network/API
    Http(String),
    Api { status: u16, message: String },
    RateLimited { retry_after: Option<Duration> },
    AuthenticationFailed,
    ModelNotFound(String),

    // Provider routing
    NoProvider,                    // No provider registered for capability
    ModelNotAvailable,             // Provider can't handle model (triggers fallback)

    // Streaming
    Stream(String),

    // Data
    Json(serde_json::Error),
    InvalidInput(String),

    // Configuration
    Configuration(String),
    NotImplemented(&'static str),
    Unsupported,

    // Soft errors
    EmptyResponse,
    ContentFiltered { reason: String },
    ContextLengthExceeded { limit: usize },

    // Wrapped
    Llm(String),
}
```

Note: `ModelNotAvailable` is used internally by providers to signal the registry should try the next provider in the fallback chain.

### Result Type

```rust
pub type Result<T> = std::result::Result<T, RatatoskrError>;
```

## Re-exports

The crate root re-exports all public types:

```rust
use ratatoskr::{
    // Gateway
    Ratatoskr,
    ModelGateway,

    // Messages
    Message,
    MessageContent,
    Role,

    // Options
    ChatOptions,
    GenerateOptions,
    ToolChoice,
    ResponseFormat,
    ReasoningConfig,
    ReasoningEffort,

    // Responses
    ChatResponse,
    ChatEvent,
    GenerateResponse,
    GenerateEvent,
    FinishReason,
    Usage,

    // Tools
    ToolDefinition,
    ToolCall,

    // Embeddings & NLI
    Embedding,
    NliResult,
    NliLabel,
    ClassifyResult,
    StanceResult,
    StanceLabel,

    // Tokenization
    Token,

    // Model introspection
    ModelInfo,
    ModelStatus,
    ModelCapability,

    // Capabilities
    Capabilities,

    // Local inference (with feature)
    Device,
    LocalEmbeddingModel,
    LocalNliModel,

    // Errors
    RatatoskrError,
    Result,
};
```
