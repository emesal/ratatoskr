# Ratatoskr Phase 2 Design: HuggingFace Provider

> Adding embeddings, NLI, and zero-shot classification via HuggingFace Inference API

## Overview

Phase 1 established the `ModelGateway` trait with chat capabilities via the `llm` crate. Phase 2 adds HuggingFace Inference API support to enable the non-chat capabilities: embeddings, NLI, and zero-shot classification.

**What this enables**:
- `gateway.embed(text, model)` - Text embeddings for semantic search/retrieval
- `gateway.infer_nli(premise, hypothesis, model)` - Natural language inference
- `gateway.classify_zero_shot(text, labels, model)` - Zero-shot classification

These capabilities are needed by downstream consumers (örlög's two-stage retrieval, stance detection).

## Scope

**In scope**:
- HuggingFace Inference API client (serverless endpoints only)
- Implement `embed`, `embed_batch`, `infer_nli`, `classify_zero_shot` methods
- Capability-based routing (builder configures which provider handles which capability)
- Isolated routing module for future extensibility

**Out of scope**:
- Chat via HuggingFace (not a good fit for HF)
- HuggingFace Dedicated Endpoints (paid, no cold starts)
- Verifying Ollama/Anthropic providers (deferred)
- Programmable routing (future enhancement)
- Default model names (consumers specify models explicitly)

## Architecture

### Current (Phase 1)

```
EmbeddedGateway
    └── chat → llm crate
```

### New (Phase 2)

```
EmbeddedGateway
    ├── chat → llm crate (unchanged)
    └── embed/nli/classify → HuggingFaceClient (new)
```

### Key Design Decisions

1. **Explicit model names**: Consumers must specify full HuggingFace model IDs (e.g., `sentence-transformers/all-MiniLM-L6-v2`). No magic defaults inside ratatoskr. Convenience constants can be added later without API changes.

2. **Builder-configured routing**: The builder determines which provider handles which capability. When `.huggingface(key)` is called, HuggingFace becomes the provider for embed/nli/classify.

3. **Isolated routing logic**: The `CapabilityRouter` is a separate module that can be ripped out and replaced with programmable routing later.

4. **Direct HTTP via reqwest**: No third-party HuggingFace client crates (none adequately support our needs). Simple REST calls to `api-inference.huggingface.co`.

## File Structure

```
src/
├── lib.rs                    # Add providers module, feature-gated exports
├── providers/
│   ├── mod.rs                # Provider module exports
│   └── huggingface.rs        # HuggingFaceClient implementation
├── gateway/
│   ├── mod.rs                # (unchanged)
│   ├── builder.rs            # Add .huggingface(key) method
│   ├── embedded.rs           # Add HuggingFaceClient field, delegate methods
│   └── routing.rs            # CapabilityRouter (new)
└── types/
    └── capabilities.rs       # Add helper constructors

tests/
├── huggingface_test.rs       # Unit tests for response parsing
├── routing_test.rs           # Unit tests for capability routing
└── integration/
    ├── huggingface_mock.rs   # Wiremock integration tests
    └── huggingface_live.rs   # Live tests (#[ignore])
```

## HuggingFace Client

### API Endpoints (Serverless)

| Capability | Endpoint | Input | Output |
|------------|----------|-------|--------|
| Embeddings | `POST /pipeline/feature-extraction/{model}` | `{"inputs": "text"}` | `[[f32; dim]]` |
| Embeddings (batch) | `POST /pipeline/feature-extraction/{model}` | `{"inputs": ["t1", "t2"]}` | `[[[f32; dim]]]` |
| NLI | `POST /models/{model}` | `{"inputs": "premise", "parameters": {"candidate_labels": ["entailment", "neutral", "contradiction"]}}` | Scores per label |
| Zero-shot | `POST /models/{model}` | `{"inputs": "text", "parameters": {"candidate_labels": ["l1", "l2"]}}` | `{"labels": [...], "scores": [...]}` |

Base URL: `https://api-inference.huggingface.co`

### Client Structure

```rust
#[cfg(feature = "huggingface")]
pub struct HuggingFaceClient {
    api_key: String,
    http: reqwest::Client,
    base_url: String,  // For testing with wiremock
}

impl HuggingFaceClient {
    pub fn new(api_key: impl Into<String>) -> Self;

    pub async fn embed(&self, text: &str, model: &str) -> Result<Embedding>;
    pub async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>>;
    pub async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>;
    pub async fn classify(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult>;
}
```

### Error Mapping

| HTTP Status | RatatoskrError |
|-------------|----------------|
| 401 | `AuthenticationFailed` |
| 404 | `ModelNotFound(model)` |
| 429 | `RateLimited { retry_after }` |
| 503 (model loading) | `Api { status: 503, message: "Model is loading..." }` |
| Other 4xx/5xx | `Api { status, message }` |

## Routing

### CapabilityRouter

```rust
pub struct CapabilityRouter {
    embed_provider: Option<EmbedProvider>,
    nli_provider: Option<NliProvider>,
    classify_provider: Option<ClassifyProvider>,
}

#[derive(Clone, Copy)]
pub enum EmbedProvider {
    HuggingFace,
    // Future: OpenAI, Ollama, Cohere, etc.
}

// Similar enums for NliProvider, ClassifyProvider
```

The router is intentionally simple - just a lookup table. This makes it easy to replace with programmable routing later.

### Builder Integration

```rust
impl RatatoskrBuilder {
    /// Configure HuggingFace provider for embeddings, NLI, and classification.
    pub fn huggingface(mut self, api_key: impl Into<String>) -> Self {
        self.huggingface_key = Some(api_key.into());
        // Implicitly configures routing:
        // - embed_provider = HuggingFace
        // - nli_provider = HuggingFace
        // - classify_provider = HuggingFace
        self
    }
}
```

### EmbeddedGateway Changes

```rust
pub struct EmbeddedGateway {
    // Existing fields for llm crate...
    openrouter_key: Option<String>,
    anthropic_key: Option<String>,
    // ...

    // New fields
    #[cfg(feature = "huggingface")]
    huggingface: Option<HuggingFaceClient>,
    router: CapabilityRouter,
}

#[async_trait]
impl ModelGateway for EmbeddedGateway {
    // chat() and chat_stream() unchanged

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        match self.router.embed_provider {
            Some(EmbedProvider::HuggingFace) => {
                self.huggingface.as_ref()
                    .ok_or(RatatoskrError::NoProvider)?
                    .embed(text, model)
                    .await
            }
            None => Err(RatatoskrError::NotImplemented("embed")),
        }
    }

    // Similar for embed_batch, infer_nli, classify_zero_shot
}
```

## Dependencies

### Cargo.toml Changes

```toml
[features]
default = ["openai", "anthropic", "openrouter", "ollama", "google", "huggingface"]
huggingface = ["dep:reqwest"]

[dependencies]
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"], optional = true }

[dev-dependencies]
wiremock = "0.6"
```

Using `rustls-tls` for portability (no system OpenSSL required).

## Testing Strategy

### Unit Tests (no network)

**`tests/huggingface_test.rs`**:
- Parse embedding response JSON → `Embedding`
- Parse NLI response JSON → `NliResult`
- Parse zero-shot response JSON → `ClassifyResult`
- Error response parsing

**`tests/routing_test.rs`**:
- Router returns correct provider for each capability
- Router returns None when provider not configured
- Builder correctly sets up routing

### Integration Tests (wiremock)

**`tests/integration/huggingface_mock.rs`**:
- Full embed request/response cycle
- Full NLI request/response cycle
- Full classification request/response cycle
- Error responses: 401, 404, 429, 503
- Batch embedding

### Live Tests (#[ignore])

**`tests/integration/huggingface_live.rs`**:
- Real API calls with `HF_API_KEY` env var
- Known-good models:
  - Embed: `sentence-transformers/all-MiniLM-L6-v2`
  - NLI: `cross-encoder/nli-deberta-v3-base`
  - Classify: `facebook/bart-large-mnli`

Run with: `HF_API_KEY=hf_xxx cargo test --test '*' -- --ignored`

## Example Usage

```rust
use ratatoskr::{Ratatoskr, ModelGateway};

#[tokio::main]
async fn main() -> ratatoskr::Result<()> {
    let gateway = Ratatoskr::builder()
        .openrouter("sk-or-...")      // For chat
        .huggingface("hf_...")        // For embed/nli/classify
        .build()?;

    // Embeddings
    let embedding = gateway
        .embed("Hello world", "sentence-transformers/all-MiniLM-L6-v2")
        .await?;
    println!("Embedding dimensions: {}", embedding.dimensions);

    // NLI
    let nli = gateway
        .infer_nli(
            "A dog is running in the park",
            "An animal is moving outdoors",
            "cross-encoder/nli-deberta-v3-base",
        )
        .await?;
    println!("Entailment score: {}", nli.entailment);

    // Zero-shot classification
    let result = gateway
        .classify_zero_shot(
            "I love this product!",
            &["positive", "negative", "neutral"],
            "facebook/bart-large-mnli",
        )
        .await?;
    println!("Top label: {} ({:.2})", result.top_label, result.confidence);

    Ok(())
}
```

## Implementation Notes

### HuggingFace API Quirks

1. **Model loading**: First request to a model may return 503 while the model loads. The response includes estimated load time. For MVP, we surface this as an error; retry logic is Phase 6.

2. **Embedding dimensions**: Different models produce different dimension embeddings. We return whatever the model produces; consumers handle dimension requirements.

3. **NLI via cross-encoders**: True NLI models (like `cross-encoder/nli-deberta-v3-base`) take premise + hypothesis and return entailment/contradiction/neutral scores directly. This is different from zero-shot classification.

4. **Batch limits**: HuggingFace has batch size limits. For MVP, we pass batches through directly and let HF error if too large. Chunking is a Phase 6 enhancement.

### Future Extensibility

The routing module is intentionally minimal. Future enhancements might include:
- Multiple providers per capability (with fallback chain)
- Cost-based or latency-based routing
- Programmable routing via config file or API
- Per-request provider override

The current design supports all of these by replacing/extending `CapabilityRouter`.

## Checklist

- [ ] Create `src/providers/mod.rs`
- [ ] Create `src/providers/huggingface.rs` with `HuggingFaceClient`
- [ ] Create `src/gateway/routing.rs` with `CapabilityRouter`
- [ ] Update `src/gateway/builder.rs` with `.huggingface()` method
- [ ] Update `src/gateway/embedded.rs` to delegate to HuggingFaceClient
- [ ] Update `src/lib.rs` with feature-gated exports
- [ ] Update `Cargo.toml` with new dependencies and feature
- [ ] Write unit tests for response parsing
- [ ] Write unit tests for routing logic
- [ ] Write wiremock integration tests
- [ ] Write live integration tests (#[ignore])
- [ ] Update `Capabilities` type with new helper methods
- [ ] Update CLAUDE.md with Phase 2 status

## References

- [HuggingFace Inference API docs](https://huggingface.co/docs/api-inference/index)
- [Phase 1 design](./2026-02-01-phase1-design.md)
- [ROADMAP.md](./ROADMAP.md) - Phase 2 section
