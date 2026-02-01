# Ratatoskr

A unified LLM gateway abstraction layer for Rust.

Ratatoskr provides a stable, provider-agnostic interface for interacting with language models. Named after the Norse squirrel who carries messages between realms in Yggdrasil, it routes your requests to the right provider while keeping your code decoupled from implementation details.

## Features

- **Single trait interface** — `ModelGateway` abstracts all providers
- **Multi-provider support** — OpenRouter, Anthropic, OpenAI, Google, Ollama
- **Streaming & non-streaming** — Both chat interfaces supported
- **Tool calling** — Full function/tool support with JSON schema parameters
- **Extended thinking** — Reasoning config for models that support it
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
```

Available features: `openai`, `anthropic`, `openrouter`, `ollama`, `google` (all enabled by default).

## Provider Configuration

```rust
let gateway = Ratatoskr::builder()
    .openrouter("sk-or-...")           // OpenRouter API key
    .anthropic("sk-ant-...")           // Direct Anthropic API
    .openai("sk-...")                  // Direct OpenAI API
    .google("...")                     // Google/Gemini API
    .ollama("http://localhost:11434")  // Local Ollama instance
    .timeout(120)                      // Request timeout in seconds
    .build()?;
```

At least one provider must be configured.

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
ModelGateway trait  ← stable public API
       │
       ▼
EmbeddedGateway     ← routes to providers
       │
       ├─► OpenRouter
       ├─► Anthropic
       ├─► OpenAI
       ├─► Google
       └─► Ollama
```

The `ModelGateway` trait is the stability boundary. Your code depends only on this trait, insulating you from provider changes.

## Roadmap

- **Phase 1** (current): Chat completions via OpenRouter and direct providers
- **Phase 2**: Additional provider integrations
- **Phase 3**: Embeddings and NLI support
- **Phase 4**: Local ONNX inference
- **Phase 5**: Service mode (gRPC/socket)
- **Phase 6**: Caching, metrics, intelligent routing

## Development

```bash
just pre-push    # Format, lint, and test (required before pushing)
just lint        # cargo fmt + clippy
just test        # Run all tests
```

## License

ISC
