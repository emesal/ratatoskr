# Ratatoskr Documentation

Ratatoskr is a unified LLM gateway abstraction layer for Rust. It provides a stable, provider-agnostic interface for interacting with language models.

## Guides

- **[Getting Started](./getting-started.md)** — Installation and first request
- **[Streaming](./streaming.md)** — Real-time response handling
- **[Tool Calling](./tools.md)** — Function calling with LLMs
- **[Providers](./providers.md)** — Multi-provider configuration
- **[Error Handling](./errors.md)** — Handling failures gracefully
- **[Service Mode](./service-mode.md)** — Running ratd as a shared gateway daemon

## Reference

- **[API Reference](./api-reference.md)** — Types and methods quick reference

## Architecture

```
Your Application
       │
       ▼
ModelGateway trait        ← stable public API
       │
       ├─► EmbeddedGateway   ← in-process, delegates to ProviderRegistry
       │
       └─► ServiceClient     ← connects to ratd over gRPC
              │
           ratd (daemon)
              │
           EmbeddedGateway → ProviderRegistry
              │
              ├─► Embeddings: LocalEmbedding → HuggingFace
              ├─► NLI: LocalNli → HuggingFace
              ├─► Stance: ZeroShotStanceProvider
              └─► Chat: OpenRouter, Anthropic, OpenAI, Google, Ollama
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

Ratatoskr is in active development. Current phase: **Pre-alpha (Phases 1-5 complete)**.

| Phase | Status | Features |
|-------|--------|----------|
| 1 | ✓ Complete | Chat, streaming, tools |
| 2 | ✓ Complete | HuggingFace API (embeddings, NLI, classification) |
| 3-4 | ✓ Complete | Local inference, tokenizers, stance detection |
| 5 | ✓ Complete | Service mode (gRPC daemon + CLI client) |
| 6 | Planned | Caching, metrics, decorator patterns |

## Links

- [GitHub Repository](https://github.com/emesal/ratatoskr)
- [API Documentation](https://docs.rs/ratatoskr) (when published)
