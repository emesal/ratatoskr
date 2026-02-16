# Provider Configuration

Ratatoskr supports multiple LLM providers. Configure the ones you need and the gateway handles routing automatically.

## Supported Providers

| Provider | Method | Models |
|----------|--------|--------|
| OpenRouter | `.openrouter(key)` | 100+ models via unified API |
| Anthropic | `.anthropic(key)` | Claude models directly |
| OpenAI | `.openai(key)` | GPT, o1, o3 models directly |
| Google | `.google(key)` | Gemini models |
| Ollama | `.ollama(url)` | Local models |

## Basic Configuration

```rust
use ratatoskr::Ratatoskr;

// Single provider
let gateway = Ratatoskr::builder()
    .openrouter("sk-or-...")
    .build()?;

// Multiple providers
let gateway = Ratatoskr::builder()
    .openrouter("sk-or-...")      // For most models
    .anthropic("sk-ant-...")       // Direct Claude access
    .ollama("http://localhost:11434")  // Local models
    .build()?;
```

## Model Routing

The gateway routes requests based on model name patterns:

```rust
// OpenRouter (prefixed models)
"anthropic/claude-sonnet-4"  // → OpenRouter
"openai/gpt-4o"              // → OpenRouter
"meta-llama/llama-3-70b"     // → OpenRouter

// Direct providers (no prefix)
"claude-sonnet-4"            // → Anthropic direct
"gpt-4o"                     // → OpenAI direct
"o1-preview"                 // → OpenAI direct
"gemini-1.5-pro"             // → Google

// Local models (colon separator)
"llama3:latest"              // → Ollama
"codellama:7b"               // → Ollama
```

### Explicit Provider Routing

When multiple providers can serve the same model, you can bypass the fallback chain and route directly to a specific provider using the `provider:model` syntax:

```rust
// Route to Anthropic direct (skip OpenRouter)
let options = ChatOptions::default()
    .model("anthropic:claude-sonnet-4-20250514");

// Route to OpenRouter explicitly
let options = ChatOptions::default()
    .model("openrouter:anthropic/claude-sonnet-4");

// Route embeddings to a specific provider
gateway.embed("hello world", "huggingface:sentence-transformers/all-MiniLM-L6-v2").await?;
```

The prefix must match a registered provider name (e.g. `openrouter`, `anthropic`, `openai`, `ollama`, `huggingface`). If the prefix is not a known provider, the entire string is treated as a plain model ID and goes through the normal fallback chain.

This works with all `ModelGateway` methods — chat, embed, NLI, classification, metadata — and is transparent over gRPC (`ServiceClient` passes the prefixed string through to the server).

**Explicit vs default routing:**
- `RoutingConfig` sets a global preferred provider per capability (reorders the fallback chain)
- `provider:model` overrides routing on a per-request basis (no fallback)

### Preset URIs

For model-agnostic code, use preset URIs that resolve to curated model IDs:

```rust
// Resolve from the model registry
let options = ChatOptions::default()
    .model("ratatoskr:free/agentic");      // Best free model for agentic use

let options = ChatOptions::default()
    .model("ratatoskr:premium/agentic");   // Best premium model for agentic use

let options = ChatOptions::default()
    .model("ratatoskr:free/text-generation"); // Best free model for general text
```

Preset URIs use the `ratatoskr:<tier>/<capability>` format and are resolved via the model registry before dispatch. They cannot be combined with provider prefixes.

## Provider Details

### OpenRouter

Best for: Accessing many models through one API, cost optimization, fallback routing.

```rust
.openrouter(std::env::var("OPENROUTER_API_KEY")?)
```

OpenRouter provides access to models from Anthropic, OpenAI, Meta, Mistral, and many others through a unified API. Use the `provider/model` format:

```rust
let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4");

let options = ChatOptions::default()
    .model("openai/gpt-4o");

let options = ChatOptions::default()
    .model("meta-llama/llama-3.1-405b-instruct");
```

### Anthropic Direct

Best for: Lowest latency to Claude, extended thinking features, prompt caching.

```rust
.anthropic(std::env::var("ANTHROPIC_API_KEY")?)
```

Use model names without prefix:

```rust
let options = ChatOptions::default()
    .model("claude-sonnet-4");

let options = ChatOptions::default()
    .model("claude-opus-4");
```

### OpenAI Direct

Best for: GPT models, o1/o3 reasoning models, DALL-E integration (future).

```rust
.openai(std::env::var("OPENAI_API_KEY")?)
```

```rust
let options = ChatOptions::default()
    .model("gpt-4o");

let options = ChatOptions::default()
    .model("o1-preview");
```

### Google

Best for: Gemini models, long context (1M+ tokens).

```rust
.google(std::env::var("GOOGLE_API_KEY")?)
```

```rust
let options = ChatOptions::default()
    .model("gemini-1.5-pro");
```

### Ollama

Best for: Local inference, privacy, no API costs, offline use.

```rust
.ollama("http://localhost:11434")
```

Model names use the `model:tag` format:

```rust
let options = ChatOptions::default()
    .model("llama3:latest");

let options = ChatOptions::default()
    .model("codellama:13b");

let options = ChatOptions::default()
    .model("mixtral:8x7b");
```

## Configuration Options

### Timeout

Set request timeout (default varies by provider):

```rust
let gateway = Ratatoskr::builder()
    .openrouter(key)
    .timeout(120)  // seconds
    .build()?;
```

### Feature Flags

Compile only the providers you need:

```toml
[dependencies]
ratatoskr = { version = "0.1", default-features = false, features = ["openrouter"] }
```

Available features:
- `openrouter`
- `anthropic`
- `openai`
- `ollama`
- `google`

## Provider-Specific Options

Use `raw_provider_options` for provider-specific parameters not covered by the standard API:

```rust
use serde_json::json;

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .raw_provider_options(json!({
        "anthropic_beta": ["prompt-caching-2024-07-31"]
    }));
```

## Checking Capabilities

Query what the gateway supports:

```rust
let caps = gateway.capabilities();

println!("Chat: {}", caps.chat);
println!("Streaming: {}", caps.chat_streaming);
println!("Embeddings: {}", caps.embeddings);
```

## Error Handling

Provider-specific errors are normalized:

```rust
use ratatoskr::RatatoskrError;

match gateway.chat(&messages, None, &options).await {
    Err(RatatoskrError::AuthenticationFailed) => {
        // Invalid API key for any provider
    }
    Err(RatatoskrError::ModelNotFound(model)) => {
        // Model not available on the routed provider
    }
    Err(RatatoskrError::NoProvider) => {
        // No provider configured for this model pattern
    }
    _ => {}
}
```

## Best Practices

1. **Use OpenRouter as default** — Simplifies configuration, provides fallbacks
2. **Add direct providers for performance** — Lower latency for frequently-used models
3. **Configure Ollama for development** — Free local testing
4. **Store keys in environment** — Never hardcode API keys
5. **Enable only needed features** — Reduces binary size
