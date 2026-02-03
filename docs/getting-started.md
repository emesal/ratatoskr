# Getting Started

This guide walks through setting up Ratatoskr and making your first LLM request.

## Installation

Add Ratatoskr to your project:

```toml
[dependencies]
ratatoskr = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Basic Setup

### 1. Configure a Gateway

The gateway is your entry point to all LLM providers. Configure it with at least one provider:

```rust
use ratatoskr::Ratatoskr;

let gateway = Ratatoskr::builder()
    .openrouter(std::env::var("OPENROUTER_API_KEY")?)
    .build()?;
```

### 2. Create Messages

Build a conversation using the `Message` type:

```rust
use ratatoskr::Message;

let messages = vec![
    Message::system("You are a helpful coding assistant."),
    Message::user("How do I read a file in Rust?"),
];
```

### 3. Configure Options

Specify the model and any optional parameters:

```rust
use ratatoskr::ChatOptions;

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .temperature(0.7)
    .max_tokens(2000);
```

### 4. Make a Request

Use the `ModelGateway` trait to chat:

```rust
use ratatoskr::ModelGateway;

let response = gateway.chat(&messages, None, &options).await?;
println!("{}", response.content);
```

## Complete Example

```rust
use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};

#[tokio::main]
async fn main() -> ratatoskr::Result<()> {
    // Build gateway
    let gateway = Ratatoskr::builder()
        .openrouter(std::env::var("OPENROUTER_API_KEY")?)
        .build()?;

    // Create conversation
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Explain async/await in Rust briefly."),
    ];

    // Configure request
    let options = ChatOptions::default()
        .model("anthropic/claude-sonnet-4")
        .max_tokens(500);

    // Get response
    let response = gateway.chat(&messages, None, &options).await?;

    println!("Response: {}", response.content);

    if let Some(usage) = &response.usage {
        println!("Tokens used: {}", usage.total_tokens);
    }

    Ok(())
}
```

## Environment Variables

Store your API keys in environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## Next Steps

- [Streaming Responses](./streaming.md) — Real-time output
- [Tool Calling](./tools.md) — Function calling with LLMs
- [Provider Configuration](./providers.md) — Multi-provider setup
- [Error Handling](./errors.md) — Handling failures gracefully
