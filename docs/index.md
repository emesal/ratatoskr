# Ratatoskr Documentation

Ratatoskr is a unified LLM gateway abstraction layer for Rust. It provides a stable, provider-agnostic interface for interacting with language models.

## Guides

- **[Getting Started](./getting-started.md)** — Installation and first request
- **[Streaming](./streaming.md)** — Real-time response handling
- **[Tool Calling](./tools.md)** — Function calling with LLMs
- **[Providers](./providers.md)** — Multi-provider configuration
- **[Error Handling](./errors.md)** — Handling failures gracefully

## Reference

- **[API Reference](./api-reference.md)** — Types and methods quick reference

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
       ├─► OpenRouter (100+ models)
       ├─► Anthropic (Claude direct)
       ├─► OpenAI (GPT, o1, o3 direct)
       ├─► Google (Gemini)
       └─► Ollama (local)
```

## Quick Example

```rust
use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};

#[tokio::main]
async fn main() -> ratatoskr::Result<()> {
    let gateway = Ratatoskr::builder()
        .openrouter(std::env::var("OPENROUTER_API_KEY")?)
        .build()?;

    let messages = vec![
        Message::system("You are helpful."),
        Message::user("Hello!"),
    ];

    let options = ChatOptions::default()
        .model("anthropic/claude-sonnet-4");

    let response = gateway.chat(&messages, None, &options).await?;
    println!("{}", response.content);

    Ok(())
}
```

## Project Status

Ratatoskr is in active development. Current phase: **Phase 1 (Chat Completions)**.

| Phase | Status | Features |
|-------|--------|----------|
| 1 | Current | Chat, streaming, tools |
| 2 | Planned | Additional providers |
| 3 | Planned | Embeddings, NLI |
| 4 | Planned | Local ONNX inference |
| 5 | Planned | Service mode (gRPC) |
| 6 | Planned | Caching, metrics |

## Links

- [GitHub Repository](https://github.com/emesal/ratatoskr)
- [API Documentation](https://docs.rs/ratatoskr) (when published)
