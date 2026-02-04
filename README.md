# Ratatoskr

A unified LLM gateway abstraction layer for Rust.

Ratatoskr provides a stable, provider-agnostic interface for interacting with language models. Named after the Norse squirrel who carries messages between realms in Yggdrasil, it routes your requests to the right provider while keeping your code decoupled from implementation details.

## Features

- **Single trait interface** — `ModelGateway` abstracts all providers
- **Multi-provider support** — OpenRouter, Anthropic, OpenAI, Google, Ollama
- **Streaming & non-streaming** — Both chat interfaces supported
- **Tool calling** — Full function/tool support with JSON schema parameters
- **Extended thinking** — Reasoning config for models that support it
- **Text generation** — Simple prompt-in, text-out interface
- **Embeddings** — Local via fastembed-rs or remote via HuggingFace API
- **NLI** — Natural language inference for semantic analysis
- **Stance detection** — Classify text as favor/against/neutral toward a target
- **Token counting** — HuggingFace tokenizers with model-appropriate defaults
- **Fallback chains** — Automatic local→remote fallback when resources constrained
- **Service mode** — `ratd` daemon + `rat` CLI over gRPC, share a gateway across processes
- **Type-safe** — Strong Rust types for messages, options, and responses

## Quick Start

```rust
use ratatoskr::{Ratatoskr, ChatOptions, Message, ModelGateway};

#[tokio::main]
async fn main() -> ratatoskr::Result<()> {
    // Build a gateway with your providers
    let gateway = Ratatoskr::builder()
        .openrouter(std::env::var("OPENROUTER_API_KEY")?)
        .build()?;

    // Create a conversation
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ];

    // Configure the request
    let options = ChatOptions::default()
        .model("anthropic/claude-sonnet-4")
        .temperature(0.7)
        .max_tokens(1000);

    // Get a response
    let response = gateway.chat(&messages, None, &options).await?;
    println!("{}", response.content);

    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ratatoskr = "0.1"
```

### Feature Flags

Enable only the providers you need:

```toml
[dependencies]
ratatoskr = { version = "0.1", default-features = false, features = ["openrouter", "anthropic"] }

# For HuggingFace API (embeddings, NLI, classification)
ratatoskr = { version = "0.1", features = ["huggingface"] }

# For local inference (no API keys needed)
ratatoskr = { version = "0.1", features = ["local-inference"] }

# GPU support for ONNX
ratatoskr = { version = "0.1", features = ["local-inference", "cuda"] }
```

Available features:
- Chat providers: `openai`, `anthropic`, `openrouter`, `ollama`, `google` (default)
- API inference: `huggingface`
- Local inference: `local-inference`, `cuda`
- Service mode: `server`, `client`

## Provider Configuration

```rust
let gateway = Ratatoskr::builder()
    .openrouter("sk-or-...")           // OpenRouter API key
    .anthropic("sk-ant-...")           // Direct Anthropic API
    .openai("sk-...")                  // Direct OpenAI API
    .google("...")                     // Google/Gemini API
    .ollama("http://localhost:11434")  // Local Ollama instance
    .huggingface("hf_...")             // HuggingFace Inference API
    .timeout(120)                      // Request timeout in seconds
    .build()?;
```

At least one provider must be configured.

### Local Inference Configuration

```rust
use ratatoskr::{Device, LocalEmbeddingModel, LocalNliModel};

let gateway = Ratatoskr::builder()
    .openrouter("sk-or-...")                           // Still need chat provider
    .local_embeddings(LocalEmbeddingModel::AllMiniLmL6V2)
    .local_nli(LocalNliModel::NliDebertaV3Small)
    .device(Device::Cpu)                               // or Device::Cuda { device_id: 0 }
    .cache_dir("/custom/model/cache")                  // Optional
    .ram_budget(1024 * 1024 * 1024)                    // Optional: 1GB limit for local models
    .build()?;
```

When RAM budget is set, local providers automatically fall back to API providers when memory is constrained.

## Model Routing

Models are automatically routed based on their name:

| Pattern | Provider |
|---------|----------|
| `anthropic/claude-*` | OpenRouter |
| `openai/gpt-*` | OpenRouter |
| `claude-*` | Direct Anthropic |
| `gpt-*`, `o1-*`, `o3-*` | Direct OpenAI |
| `gemini-*` | Google |
| `model:tag` | Ollama (local) |
| Other | OpenRouter (default) |

## Streaming

```rust
use futures_util::StreamExt;
use ratatoskr::ChatEvent;

let mut stream = gateway.chat_stream(&messages, None, &options).await?;

while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::Content(text) => print!("{}", text),
        ChatEvent::Reasoning(thought) => eprintln!("[thinking] {}", thought),
        ChatEvent::ToolCallStart { name, .. } => println!("Calling tool: {}", name),
        ChatEvent::Usage(usage) => println!("Tokens: {}", usage.total_tokens),
        ChatEvent::Done => break,
        _ => {}
    }
}
```

## Tool Calling

```rust
use ratatoskr::{ToolDefinition, ToolChoice};
use serde_json::json;

// Define a tool
let weather_tool = ToolDefinition::new(
    "get_weather",
    "Get the current weather for a location",
    json!({
        "type": "object",
        "properties": {
            "location": { "type": "string", "description": "City name" }
        },
        "required": ["location"]
    }),
);

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .tool_choice(ToolChoice::Auto);

let response = gateway.chat(&messages, Some(&[weather_tool]), &options).await?;

// Handle tool calls
for tool_call in &response.tool_calls {
    println!("Tool: {} with args: {}", tool_call.name, tool_call.arguments);

    // Parse arguments into your struct
    let args: WeatherArgs = tool_call.parse_arguments()?;
}
```

## Extended Thinking

For models that support reasoning (Claude with extended thinking, o1, etc.):

```rust
use ratatoskr::{ReasoningConfig, ReasoningEffort};

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .reasoning(ReasoningConfig {
        effort: Some(ReasoningEffort::High),
        max_tokens: Some(10000),
        exclude_from_output: Some(false),  // Include thinking in response
    });

let response = gateway.chat(&messages, None, &options).await?;
if let Some(reasoning) = &response.reasoning {
    println!("Thinking: {}", reasoning);
}
```

## Error Handling

```rust
use ratatoskr::RatatoskrError;

match gateway.chat(&messages, None, &options).await {
    Ok(response) => println!("{}", response.content),
    Err(RatatoskrError::RateLimited { retry_after }) => {
        println!("Rate limited, retry after {:?}", retry_after);
    }
    Err(RatatoskrError::AuthenticationFailed) => {
        eprintln!("Invalid API key");
    }
    Err(RatatoskrError::ModelNotFound(model)) => {
        eprintln!("Unknown model: {}", model);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Architecture

```
Your Application
       │
       ▼
ModelGateway trait      ← stable public API
       │
       ├─► EmbeddedGateway  ← in-process, delegates to ProviderRegistry
       │
       └─► ServiceClient    ← connects to ratd over gRPC
              │
              ▼
           ratd (daemon)
              │
              ▼
           EmbeddedGateway  ← delegates to ProviderRegistry
       │
       ▼
ProviderRegistry    ← fallback chains per capability
       │
       ├─► Embedding Providers
       │     ├── LocalEmbeddingProvider (fastembed) [priority 0]
       │     └── HuggingFaceClient [priority 1, fallback]
       │
       ├─► NLI Providers
       │     ├── LocalNliProvider (ONNX) [priority 0]
       │     └── HuggingFaceClient [priority 1, fallback]
       │
       ├─► Stance Providers
       │     └── ZeroShotStanceProvider (wraps ClassifyProvider)
       │
       └─► Chat/Generate Providers
             ├── OpenRouter
             ├── Anthropic
             ├── OpenAI
             ├── Google
             └── Ollama
```

The `ModelGateway` trait is the stability boundary. Your code depends only on this trait, insulating you from provider changes.

**Fallback behaviour:** When a local provider returns `ModelNotAvailable` (wrong model or RAM budget exceeded), the registry automatically tries the next provider in the chain.

## Service Mode

Service mode lets multiple processes share a single gateway instance over gRPC. The daemon (`ratd`) wraps an `EmbeddedGateway` behind gRPC handlers; clients connect via `ServiceClient` which implements `ModelGateway` transparently.

### Server (`ratd`)

```bash
cargo build --features server
ratd --config config.toml
```

Configuration uses TOML files with provider API keys stored separately in a permissions-checked secrets file (`~/.config/ratatoskr/secrets.toml`, mode 0600).

### Client library

```rust
use ratatoskr::{ServiceClient, ModelGateway, ChatOptions, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ServiceClient::connect("http://127.0.0.1:9741").await?;

    let response = client
        .chat(
            &[Message::user("hello!")],
            None,
            &ChatOptions::default().model("anthropic/claude-sonnet-4"),
        )
        .await?;

    println!("{}", response.content);
    Ok(())
}
```

### CLI (`rat`)

```bash
cargo build --features client

rat health              # check connectivity
rat models              # list available models
rat chat "hello!" -m anthropic/claude-sonnet-4
rat embed "some text"   # generate embeddings
rat nli "it rained" "the ground is wet"
rat tokens "count me"   # token counting
```

### systemd

A systemd unit file is provided at `contrib/systemd/ratd.service` with security hardening (sandboxing, resource limits, separate service user).

## Text Generation

For simple prompt-to-text generation (without the chat/message structure):

```rust
use ratatoskr::GenerateOptions;

let response = gateway.generate(
    "Once upon a time",
    &GenerateOptions::new("llama3.2:1b")
        .max_tokens(100)
        .temperature(0.7),
).await?;

println!("{}", response.text);
```

Streaming generation:

```rust
use ratatoskr::GenerateEvent;
use futures_util::StreamExt;

let mut stream = gateway.generate_stream(
    "Once upon a time",
    &GenerateOptions::new("anthropic/claude-3-haiku").max_tokens(100),
).await?;

while let Some(event) = stream.next().await {
    match event? {
        GenerateEvent::Text(text) => print!("{}", text),
        GenerateEvent::Done => break,
    }
}
```

## Local Inference

With the `local-inference` feature, run embeddings and NLI locally without API keys:

```rust
use ratatoskr::providers::{FastEmbedProvider, LocalEmbeddingModel};

// Local embeddings
let mut provider = FastEmbedProvider::new(LocalEmbeddingModel::AllMiniLmL6V2)?;
let embedding = provider.embed("Hello, world!")?;
println!("Dimensions: {}", embedding.dimensions);  // 384

// Batch embeddings
let embeddings = provider.embed_batch(&["First", "Second", "Third"])?;
```

```rust
use ratatoskr::providers::{OnnxNliProvider, LocalNliModel};
use ratatoskr::Device;

// Local NLI
let mut provider = OnnxNliProvider::new(LocalNliModel::NliDebertaV3Small, Device::Cpu)?;
let result = provider.infer_nli("A cat is sleeping", "An animal is resting")?;
println!("{:?}: {:.2}", result.label, result.entailment);  // Entailment: 0.95
```

### Stance Detection

Classify text as expressing favor, against, or neutral toward a target topic:

```rust
use ratatoskr::ModelGateway;

let stance = gateway.classify_stance(
    "I strongly support renewable energy initiatives.",
    "renewable energy",
    "facebook/bart-large-mnli",
).await?;

println!("{:?}: favor={:.2}, against={:.2}",
    stance.label, stance.favor, stance.against);
// Favor: favor=0.85, against=0.05
```

### Token Counting

```rust
use ratatoskr::tokenizer::TokenizerRegistry;

let registry = TokenizerRegistry::new();
let count = registry.count_tokens("Hello, world!", "claude-sonnet-4")?;
println!("Tokens: {}", count);  // ~3

// Detailed tokenization with offsets
let tokens = gateway.tokenize("Hello, world!", "claude-sonnet-4")?;
for token in tokens {
    println!("{}: bytes {}..{}", token.text, token.start, token.end);
}
```

Supported embedding models: `AllMiniLmL6V2`, `AllMiniLmL12V2`, `BgeSmallEn`, `BgeBaseEn`

Supported NLI models: `NliDebertaV3Base`, `NliDebertaV3Small`, or custom ONNX models

## Roadmap

- **Phase 1**: Chat completions via OpenRouter and direct providers ✓
- **Phase 2**: HuggingFace provider (embeddings, NLI, classification) ✓
- **Phase 3-4**: Local inference (embeddings, NLI, tokenizers, generate) ✓
- **Provider Trait Refactor**: Fallback chains, RAM budget, stance detection ✓
- **Phase 5**: Service mode (gRPC daemon + CLI client) ✓
- **Phase 6**: Caching, metrics, decorator patterns

## Development

```bash
just pre-push    # Format, lint, and test (required before pushing)
just lint        # cargo fmt + clippy
just test        # Run all tests
```

## License

ISC
