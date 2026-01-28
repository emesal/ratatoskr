# Ratatoskr Architecture

> Ratatoskr (Old Norse: "drill-tooth") — the squirrel that runs up and down Yggdrasil carrying messages between the eagle at the top and the serpent Níðhöggr at the roots.

A unified model gateway that abstracts over local and remote AI model providers, offering a single interface to any client regardless of where models run.

## Overview

Ratatoskr serves two purposes:

1. **As a crate**: Lightweight, embeddable API client for applications that only need remote model access
2. **As a service**: Shared model gateway that loads local models once and serves multiple clients

Applications use the same interface in both modes — they don't know or care whether inference happens locally or remotely.

## Architecture

### Dual-Mode Operation

```
┌─────────────────────────────────────────────────────────────────┐
│                     ratatoskr service                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Provider Layer                        │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │   ONNX    │  │ OpenRouter│  │  Ollama   │  ...       │   │
│  │  │  Runtime  │  │   API     │  │   API     │            │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘            │   │
│  │        └──────────────┴──────────────┘                   │   │
│  │                       │                                  │   │
│  │           ┌───────────┴───────────┐                      │   │
│  │           │     Router Layer      │                      │   │
│  │           │  (task → provider)    │                      │   │
│  │           └───────────┬───────────┘                      │   │
│  └───────────────────────┼──────────────────────────────────┘   │
│                          │                                      │
│              ┌───────────┴───────────┐                          │
│              │    Server Interface   │                          │
│              │   (gRPC/Unix socket)  │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │  Client A  │   │  Client B  │   │  Client C  │
   │ (chibi +   │   │ (chibi     │   │ (other     │
   │  örlög)    │   │  standalone)│  │  project)  │
   └────────────┘   └────────────┘   └────────────┘
```

### Embedded Mode

For lightweight use without the service:

```
┌────────────────────────────────────────┐
│              Application               │
│  ┌──────────────────────────────────┐  │
│  │     ratatoskr (embedded)         │  │
│  │  ┌────────────┐  ┌────────────┐  │  │
│  │  │ OpenRouter │  │  Anthropic │  │  │
│  │  │   client   │  │   client   │  │  │
│  │  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

No service required. Only API clients are activated — no model loading, negligible memory footprint.

## Core Interface

```rust
/// Unified interface for all model operations.
/// Implemented by both embedded and client modes.
#[async_trait]
pub trait ModelGateway: Send + Sync {
    // Embeddings
    async fn embed(&self, text: &str) -> Result<Embedding>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;

    // Natural Language Inference
    async fn infer_nli(&self, premise: &str, hypothesis: &str) -> Result<NliResult>;
    async fn infer_nli_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliResult>>;

    // Classification
    async fn classify_stance(&self, text: &str, target: &str) -> Result<StanceResult>;
    async fn classify_zero_shot(&self, text: &str, labels: &[&str]) -> Result<ClassifyResult>;

    // Generation
    async fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<String>;
    async fn generate_stream(
        &self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Result<impl Stream<Item = Result<String>>>;

    // Chat
    async fn chat(&self, messages: &[Message], options: ChatOptions) -> Result<Message>;
    async fn chat_stream(
        &self,
        messages: &[Message],
        options: ChatOptions,
    ) -> Result<impl Stream<Item = Result<MessageDelta>>>;

    // Tokenization
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize>;
    fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>>;

    // Model management
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn model_status(&self, model: &str) -> Result<ModelStatus>;
}
```

## Types

```rust
pub struct Embedding {
    pub values: Vec<f32>,
    pub model: String,
    pub dimensions: usize,
}

pub struct NliResult {
    pub entailment: f32,
    pub contradiction: f32,
    pub neutral: f32,
    pub label: NliLabel,  // Entailment | Contradiction | Neutral
}

pub struct StanceResult {
    pub favor: f32,
    pub against: f32,
    pub neutral: f32,
    pub label: StanceLabel,  // Favor | Against | Neutral
    pub target: String,
}

pub struct ClassifyResult {
    pub scores: HashMap<String, f32>,
    pub top_label: String,
    pub confidence: f32,
}

pub struct Message {
    pub role: Role,  // System | User | Assistant
    pub content: String,
}

pub struct GenerateOptions {
    pub model: Option<String>,      // Override configured model
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

pub struct ChatOptions {
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
}

pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub capabilities: Vec<Capability>,  // Embed | Nli | Classify | Generate | Chat
    pub context_window: Option<usize>,
    pub loaded: bool,
}

pub enum ModelStatus {
    Available,           // Can be loaded
    Loading,             // Currently loading
    Ready,               // Loaded and ready
    Unavailable(String), // Error or missing
}
```

## Usage

### Construction

```rust
use ratatoskr::{Ratatoskr, Config};

// Embedded mode — lightweight, API-only
let gateway = Ratatoskr::embedded()
    .with_openrouter(api_key)
    .with_anthropic(api_key)  // Optional additional provider
    .build()?;

// Client mode — connects to service
let gateway = Ratatoskr::connect("unix:///run/ratatoskr.sock").await?;

// Auto mode — service if available, embedded fallback
let gateway = Ratatoskr::auto()
    .service_socket("/run/ratatoskr.sock")
    .fallback_openrouter(api_key)
    .build()
    .await?;
```

### Operations

```rust
// Embeddings
let embedding = gateway.embed("Hello world").await?;
let embeddings = gateway.embed_batch(&["Hello", "World"]).await?;

// NLI
let nli = gateway.infer_nli(
    "The cat sat on the mat",
    "An animal is on the floor"
).await?;
// nli.label == NliLabel::Entailment

// Stance detection
let stance = gateway.classify_stance(
    "Rust's ownership model prevents memory bugs",
    "Rust"
).await?;
// stance.label == StanceLabel::Favor

// Chat
let response = gateway.chat(&[
    Message::user("What is the capital of France?")
], ChatOptions::default()).await?;

// Streaming
let mut stream = gateway.chat_stream(&messages, options).await?;
while let Some(delta) = stream.next().await {
    print!("{}", delta?.content);
}

// Token counting
let count = gateway.count_tokens("Hello world", "claude-sonnet")?;
```

## Providers

### Local Providers (service mode only)

**ONNX Runtime**

Runs transformer models locally via ONNX format.

```toml
[providers.onnx]
device = "cuda"  # or "cpu"
cache_dir = "~/.cache/ratatoskr/models"

[providers.onnx.models.embed]
name = "all-MiniLM-L6-v2"
source = "huggingface"  # auto-downloads

[providers.onnx.models.nli]
name = "nli-deberta-v3-base"
source = "huggingface"
```

Supported tasks:
- Embeddings (sentence-transformers)
- NLI (cross-encoder models)
- Stance/zero-shot classification (DeBERTa, BART-MNLI)

**Ollama**

Local LLM inference via Ollama.

```toml
[providers.ollama]
base_url = "http://localhost:11434"

[providers.ollama.models.chat]
name = "llama3"
```

Supported tasks:
- Chat
- Generate

### Remote Providers (both modes)

**OpenRouter**

Access to 100+ models via single API.

```toml
[providers.openrouter]
api_key = "${OPENROUTER_API_KEY}"
base_url = "https://openrouter.ai/api/v1"  # default

[providers.openrouter.models.chat]
name = "anthropic/claude-sonnet"

[providers.openrouter.models.generate]
name = "anthropic/claude-haiku"

[providers.openrouter.models.embed]
name = "openai/text-embedding-3-small"
```

Supported tasks:
- Chat
- Generate
- Embeddings (select models)

**Anthropic**

Direct Anthropic API access.

```toml
[providers.anthropic]
api_key = "${ANTHROPIC_API_KEY}"

[providers.anthropic.models.chat]
name = "claude-sonnet-4-20250514"
```

Supported tasks:
- Chat
- Generate

## Routing

The router maps tasks to providers based on configuration:

```toml
[routing]
# Task → provider mapping
embed = "onnx"           # Local embeddings
nli = "onnx"             # Local NLI
classify = "onnx"        # Local classification
chat = "openrouter"      # Remote chat
generate = "openrouter"  # Remote generation

[routing.fallbacks]
# Fallback chain if primary fails
chat = ["anthropic", "ollama"]
embed = ["openrouter"]
```

When a task is requested:
1. Router selects configured provider
2. If provider fails, tries fallback chain
3. If all fail, returns error with context

## Service Configuration

Full service configuration:

```toml
[server]
socket = "/run/ratatoskr.sock"
# Alternative: TCP
# host = "127.0.0.1"
# port = 50051

[server.limits]
max_batch_size = 100
request_timeout_ms = 30000
max_concurrent_requests = 50

[cache]
# Response caching for identical requests
enabled = true
ttl_seconds = 3600
max_entries = 10000

[providers.onnx]
device = "cuda"
cache_dir = "~/.cache/ratatoskr/models"

[providers.onnx.models.embed]
name = "all-MiniLM-L6-v2"
source = "huggingface"

[providers.onnx.models.nli]
name = "nli-deberta-v3-base"
source = "huggingface"

[providers.openrouter]
api_key = "${OPENROUTER_API_KEY}"

[providers.openrouter.models.chat]
name = "anthropic/claude-sonnet"

[routing]
embed = "onnx"
nli = "onnx"
chat = "openrouter"
generate = "openrouter"
```

## Resource Management

### Model Lifecycle

```
┌─────────┐    load()    ┌─────────┐   ready    ┌─────────┐
│Available│ ──────────► │ Loading │ ────────► │  Ready  │
└─────────┘              └─────────┘            └────┬────┘
     ▲                                               │
     │                    unload()                   │
     └───────────────────────────────────────────────┘
```

- Models loaded on first use or explicitly via API
- LRU eviction when memory pressure detected
- Manual unload available for resource management

### Memory Profile

| Component | Memory | Notes |
|-----------|--------|-------|
| Service base | ~50MB | Runtime overhead |
| Embedding model | ~100-500MB | Depends on model size |
| NLI model | ~100-400MB | DeBERTa variants |
| Classification | ~400MB-2GB | Shared with NLI if same base |
| API clients | Negligible | Just HTTP clients |

Typical service with embed + NLI: **~500MB-1GB**

### Concurrency

- Thread pool for ONNX inference (CPU-bound)
- Async for API calls (IO-bound)
- Request queuing with configurable limits
- Graceful degradation under load

## Protocol

### Option A: gRPC

Structured, typed, good tooling, streaming support.

```protobuf
service Ratatoskr {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
  rpc InferNli(NliRequest) returns (NliResponse);
  rpc Chat(ChatRequest) returns (ChatResponse);
  rpc ChatStream(ChatRequest) returns (stream ChatDelta);
  // ...
}
```

Pros:
- Type safety across language boundaries
- Efficient binary serialization
- Built-in streaming
- Good Rust support (tonic)

Cons:
- Protobuf codegen step
- Heavier dependency

### Option B: JSON over Unix Socket

Simpler, no codegen, easy debugging.

```json
{"method": "embed", "params": {"text": "Hello world"}}
{"method": "chat", "params": {"messages": [...]}}
```

Pros:
- Simple to implement and debug
- No build-time codegen
- Easy to test with netcat/socat

Cons:
- No type checking at protocol level
- Manual streaming implementation
- Slightly higher overhead

**Recommendation**: Start with JSON for simplicity during development. Migrate to gRPC if performance becomes an issue or if non-Rust clients need good ergonomics.

## CLI

```bash
# Start service
ratatoskr serve --config /etc/ratatoskr/config.toml

# Service management
ratatoskr status
ratatoskr reload-config

# Model management
ratatoskr models list
ratatoskr models load embed
ratatoskr models unload nli

# Testing
ratatoskr embed "Hello world"
ratatoskr nli "The cat sat" "An animal rested"
ratatoskr chat "What is 2+2?"
```

## Integration Examples

### Chibi (standalone, no örlög)

```rust
// In chibi startup
let ratatoskr = Ratatoskr::auto()
    .service_socket("/run/ratatoskr.sock")
    .fallback_openrouter(config.openrouter_api_key)
    .build()
    .await?;

// Use for chat - works whether service is running or not
let response = ratatoskr.chat(&messages, options).await?;
```

### Örlög

```rust
// örlög requires full capabilities
let ratatoskr = Ratatoskr::connect("/run/ratatoskr.sock").await
    .context("örlög requires ratatoskr service for local model inference")?;

// Skuld: stance detection
let stance = ratatoskr.classify_stance(&message.content, "user_goal").await?;

// Urðr: embeddings for retrieval
let embedding = ratatoskr.embed(&query).await?;

// Urðr: NLI for filtering
let nli = ratatoskr.infer_nli(&memory.content, &query).await?;

// Verðandi: token counting for budget
let tokens = ratatoskr.count_tokens(&assembled_prompt, "claude-sonnet")?;
```

### Other Projects

Any Rust project can use ratatoskr as a crate:

```toml
[dependencies]
ratatoskr = { version = "0.1", features = ["client"] }  # Client only
# or
ratatoskr = { version = "0.1", features = ["embedded"] }  # Embedded only
# or
ratatoskr = { version = "0.1", features = ["full"] }  # Both
```

## Project Structure

```
ratatoskr/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API, re-exports
│   ├── traits.rs           # ModelGateway trait
│   ├── types.rs            # Shared types
│   ├── error.rs            # Error types
│   ├── config.rs           # Configuration loading
│   │
│   ├── embedded/           # Embedded mode
│   │   ├── mod.rs
│   │   └── builder.rs      # Embedded builder
│   │
│   ├── client/             # Client mode
│   │   ├── mod.rs
│   │   └── connection.rs   # Service connection
│   │
│   ├── providers/          # Provider implementations
│   │   ├── mod.rs
│   │   ├── openrouter.rs
│   │   ├── anthropic.rs
│   │   ├── ollama.rs
│   │   └── onnx.rs         # Local ONNX inference
│   │
│   ├── routing/            # Task routing
│   │   ├── mod.rs
│   │   └── fallback.rs
│   │
│   └── server/             # Service implementation
│       ├── mod.rs
│       ├── handler.rs
│       └── protocol.rs
│
├── src/bin/
│   └── ratatoskr.rs        # CLI entry point
│
└── tests/
    ├── embedded.rs
    ├── client.rs
    └── integration.rs
```

## Open Questions

1. **Protocol choice**: gRPC vs JSON — start simple or invest in structure?

2. **Model hot-reloading**: Should config changes reload models without restart?

3. **Multi-tenancy**: Should the service support multiple API keys / accounts for different clients?

4. **Caching semantics**: Cache embeddings? Cache chat responses? What invalidation strategy?

5. **Metrics/observability**: Prometheus metrics? Structured logging? Tracing?
