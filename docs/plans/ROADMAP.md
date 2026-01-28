# Ratatoskr Roadmap

> From chibi extraction to unified model gateway.

This document sketches the full implementation path for ratatoskr, including significant considerations, open questions, and decision points at each phase.

---

## Phase Overview

```
Phase 1: OpenRouter Chat          ←── You are here
    ↓
Phase 2: Additional Providers
    ↓
Phase 3: Embeddings & Classification
    ↓
Phase 4: Local Inference (ONNX)
    ↓
Phase 5: Service Mode
    ↓
Phase 6: Advanced Features
```

---

## Phase 1: OpenRouter Chat Gateway

**Goal**: Extract chibi's API code into a working embedded-mode crate.

**Deliverables**:
- `ModelGateway` trait with forward-compatible signature
- OpenRouter provider with SSE streaming
- Type system (Message, ChatEvent, ToolCall, etc.)
- Builder pattern for embedded mode
- Chibi integration

**See**: [phase-1-plan.md](./phase-1-plan.md) for full details.

**Duration estimate**: This is a focused extraction with clear boundaries.

---

## Phase 2: Additional Providers

**Goal**: Prove the abstraction works by adding more providers.

### 2a: Hugging Face Inference API

The Hugging Face Inference API provides access to thousands of models, including:
- Text generation (chat/completion)
- Embeddings
- Zero-shot classification
- NLI (via appropriate models)

**Key considerations**:

1. **API differences**: HF Inference API has different endpoints per task, not a unified chat endpoint. Need to map our unified interface to their task-specific endpoints.

2. **Model discovery**: HF has thousands of models. How do we expose this?
   - Option A: User specifies full model ID (`sentence-transformers/all-MiniLM-L6-v2`)
   - Option B: We maintain a curated list of known-good models
   - Option C: Query HF API for models matching capability
   - **Recommendation**: Option A with optional B for defaults

3. **Serverless vs Dedicated**: HF offers serverless (cold start) and dedicated endpoints. Initially support serverless only, add dedicated endpoint configuration later.

4. **Rate limits**: HF free tier has aggressive rate limits. Need clear error handling.

**New types needed**:
```rust
pub struct HuggingFaceOptions {
    pub wait_for_model: bool,  // Wait if model is loading
    pub use_cache: bool,       // Use HF's response cache
}
```

### 2b: Ollama Provider

For local LLM inference via Ollama.

**Key considerations**:

1. **Connection handling**: Ollama runs locally, usually on `localhost:11434`. Need to handle connection failures gracefully (service not running).

2. **Model availability**: Unlike cloud APIs, models must be pulled first. Should we:
   - Just error if model not available?
   - Offer to pull on demand?
   - **Recommendation**: Error with helpful message, don't auto-pull

3. **Streaming differences**: Ollama's streaming format differs from OpenRouter. Need provider-specific SSE parsing.

4. **Capability detection**: Ollama models vary in capabilities (some support tools, some don't). Need runtime capability detection.

### 2c: Direct Anthropic API (Optional)

Lower priority since OpenRouter provides Anthropic access, but useful for:
- Users who want direct relationship with Anthropic
- Potentially lower latency
- Anthropic-specific features (prompt caching semantics differ from OpenRouter's)

**Considerations**:
- Anthropic's tool use format differs slightly from OpenAI's
- Their streaming format uses different event types
- Beta features (computer use, etc.) need careful handling

### Phase 2 Questions

1. **Provider priority**: Which provider after OpenRouter? HF has more capability breadth, Ollama is simpler.

2. **Feature flags vs runtime**: Should providers be compile-time features or runtime-configurable?
   ```toml
   # Compile-time (current plan)
   ratatoskr = { features = ["openrouter", "huggingface"] }

   # vs Runtime
   let gateway = Ratatoskr::embedded()
       .add_provider(OpenRouterProvider::new(key))
       .add_provider(HuggingFaceProvider::new(key))
       .build();
   ```
   **Recommendation**: Start with compile-time features, consider runtime for Phase 5 (service mode).

3. **Provider selection**: When multiple providers can handle a request, how to choose?
   - Explicit per-request: `options.provider = "huggingface"`
   - Model-prefix convention: `hf/meta-llama/Llama-3-8B`
   - Configuration-based routing
   - **Recommendation**: Model-prefix for Phase 2, full routing for Phase 5

---

## Phase 3: Embeddings & Classification

**Goal**: Implement the non-chat methods in `ModelGateway`.

### 3a: Embeddings

**Via Hugging Face Inference API**:
```rust
// User code
let embedding = gateway.embed("Hello world").await?;

// Under the hood: POST to HF inference endpoint for embedding model
```

**Via OpenRouter** (limited model selection):
```rust
// OpenRouter supports some embedding models
// May have different pricing/availability
```

**Key considerations**:

1. **Dimension handling**: Different models produce different dimension embeddings. Should we normalize?
   - Option A: Return raw dimensions, caller handles
   - Option B: Offer dimension reduction as option
   - **Recommendation**: Option A, keep it simple

2. **Batch efficiency**: `embed_batch()` should use provider's native batching, not N separate calls.

3. **Consistency**: Same text + same model should produce same embedding. Need to verify provider behavior.

### 3b: NLI (Natural Language Inference)

For örlög's two-stage retrieval (The Lens).

**Via Hugging Face**:
- Use cross-encoder NLI models like `cross-encoder/nli-deberta-v3-base`
- These give entailment/contradiction/neutral scores directly

**Via LLM fallback**:
- Can prompt any chat model to perform NLI
- Less accurate, more expensive, but available everywhere
- Useful as fallback when no dedicated NLI model configured

**Key considerations**:

1. **Score calibration**: Different models produce different score distributions. Should we calibrate?
   - **Recommendation**: No calibration in ratatoskr, let örlög handle if needed

2. **Batch NLI**: örlög's filtering step checks many premise-hypothesis pairs. Need efficient batching.

### 3c: Zero-Shot Classification

For örlög's stance detection.

**Via Hugging Face**:
- Models like `facebook/bart-large-mnli` support zero-shot classification
- User provides text + candidate labels, model returns scores

**Via LLM fallback**:
- Prompt engineering for classification
- More flexible (any labels) but slower/costlier

### Phase 3 Questions

1. **Model defaults**: Should ratatoskr have built-in defaults for embedding/NLI models, or require explicit configuration?
   ```rust
   // With defaults
   gateway.embed("text").await  // Uses default model

   // Explicit
   gateway.embed_with_model("text", "sentence-transformers/all-MiniLM-L6-v2").await
   ```
   **Recommendation**: Require explicit model selection, no magic defaults

2. **LLM fallback policy**: Should non-chat methods automatically fall back to LLM-based implementation if no dedicated model available?
   - Pro: Always works if you have chat capability
   - Con: Unexpected cost/latency, inconsistent results
   - **Recommendation**: Explicit opt-in, not automatic

3. **Caching**: Should embedding results be cached? Same text → same embedding.
   - Pro: Major cost/latency savings for repeated texts
   - Con: Memory usage, cache invalidation complexity
   - **Recommendation**: Optional, off by default in Phase 3, refine in Phase 6

---

## Phase 4: Local Inference (ONNX)

**Goal**: Run embedding/NLI/classification models locally without external APIs.

This is what enables örlög to work efficiently — fast, free local inference for the high-volume operations (embeddings for retrieval, NLI for filtering).

### ONNX Runtime Integration

Using the `ort` crate (ONNX Runtime Rust bindings):

```rust
pub struct OnnxProvider {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl OnnxProvider {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Load ONNX model and tokenizer
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize, run inference, extract embeddings
    }
}
```

**Key considerations**:

1. **Model acquisition**: Where do models come from?
   - Download from Hugging Face Hub on first use (via `hf-hub` crate)
   - Cache locally (`~/.cache/ratatoskr/models/`)
   - Support pre-downloaded models for air-gapped environments

2. **Model format**: Need ONNX-exported models
   - Many HF models have ONNX exports available
   - Some need conversion (can document process, not automate)

3. **Device selection**: CPU vs CUDA vs other accelerators
   ```rust
   pub enum Device {
       Cpu,
       Cuda { device_id: u32 },
       // Future: CoreML, DirectML, etc.
   }
   ```

4. **Memory management**: ONNX models are large (100MB-2GB)
   - Need explicit load/unload in service mode
   - In embedded mode, load on first use, keep loaded

5. **Thread pool**: ONNX inference is CPU-bound
   - Need separate thread pool, not tokio's async executor
   - Use `spawn_blocking` or dedicated pool

6. **Warm-up**: First inference is slow (JIT compilation, memory allocation)
   - Consider warm-up on load for latency-sensitive deployments

### Tokenization

For `count_tokens()` support:

```rust
use tokenizers::Tokenizer;

pub struct TokenizerRegistry {
    tokenizers: HashMap<String, Tokenizer>,
}

impl TokenizerRegistry {
    pub fn count(&self, text: &str, model: &str) -> Result<usize> {
        let tokenizer = self.get_or_load(model)?;
        Ok(tokenizer.encode(text, false)?.len())
    }
}
```

**Considerations**:
- Different model families use different tokenizers
- Need mapping from model name → tokenizer
- HF tokenizers can be loaded from model repos

### Phase 4 Questions

1. **Embedded mode with ONNX**: Should embedded mode support local inference, or only service mode?
   - Pro: Single-binary deployment, no service to manage
   - Con: Memory usage, startup time
   - **Recommendation**: Support both, but document tradeoffs

2. **Model versioning**: Models get updated. How to handle?
   - Pin specific versions in config
   - Auto-update with cache invalidation
   - **Recommendation**: Pin versions, explicit upgrade

3. **Hybrid routing**: Same capability from multiple sources (e.g., embeddings from both ONNX and HF API). How to choose?
   - Cost-based (local is free)
   - Latency-based (local is faster after warm-up)
   - Availability-based (fallback to API if local fails)
   - **Recommendation**: Configure primary + fallback per capability

4. **GPU memory**: Multiple ONNX models competing for VRAM
   - LRU eviction when memory pressure detected?
   - Explicit memory budget configuration?
   - **Recommendation**: Phase 5 (service mode) concern, keep simple in Phase 4

---

## Phase 5: Service Mode

**Goal**: Shared gateway service that multiple clients can connect to.

This is what makes ratatoskr efficient for multi-instance deployments — load models once, share across clients.

### Architecture

```
┌─────────────────────────────────────────┐
│           ratatoskr service             │
│  ┌─────────────────────────────────┐    │
│  │        Provider Layer           │    │
│  │  ONNX | OpenRouter | HF | ...   │    │
│  └───────────────┬─────────────────┘    │
│                  │                      │
│  ┌───────────────┴─────────────────┐    │
│  │        Protocol Handler         │    │
│  │    (gRPC or JSON-over-socket)   │    │
│  └───────────────┬─────────────────┘    │
└──────────────────┼──────────────────────┘
                   │ Unix socket / TCP
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Client A   Client B   Client C
```

### Protocol Decision

**Option A: gRPC (tonic)**

Pros:
- Strong typing via protobuf
- Built-in streaming
- Good cross-language support (if non-Rust clients needed)
- Efficient binary serialization

Cons:
- Protobuf codegen in build process
- Heavier dependency tree
- Harder to debug (binary protocol)

**Option B: JSON over Unix Socket**

Pros:
- Simple to implement
- Easy to debug (human-readable)
- No codegen
- Minimal dependencies

Cons:
- Manual streaming implementation (newline-delimited JSON)
- No compile-time type checking across boundary
- Slightly higher overhead

**Recommendation**: Start with JSON for simplicity. The performance difference is negligible for typical usage patterns. Migrate to gRPC if:
- Non-Rust clients become important
- Benchmarks show serialization as bottleneck
- Type safety across boundary becomes painful

### Client Mode Implementation

```rust
pub struct ServiceClient {
    socket: UnixStream,  // or TcpStream
}

impl ModelGateway for ServiceClient {
    async fn chat_stream(&self, ...) -> Result<Stream<ChatEvent>> {
        // Send request as JSON
        // Read newline-delimited JSON events
        // Convert to ChatEvent stream
    }
}
```

### Auto Mode

Connect to service if available, fall back to embedded:

```rust
let gateway = Ratatoskr::auto()
    .service_socket("/run/ratatoskr.sock")
    .fallback_openrouter(api_key)
    .build()
    .await?;
```

Implementation:
1. Try to connect to socket
2. If successful, use ServiceClient
3. If failed (service not running), build embedded gateway with fallback config

### Service Configuration

```toml
[server]
socket = "/run/ratatoskr.sock"
# or TCP:
# bind = "127.0.0.1:50051"

[server.limits]
max_concurrent_requests = 100
request_timeout_ms = 30000
max_batch_size = 100

[providers.onnx]
device = "cuda"
models_dir = "~/.cache/ratatoskr/models"

[providers.onnx.models]
embed = { name = "all-MiniLM-L6-v2", source = "huggingface" }
nli = { name = "nli-deberta-v3-base", source = "huggingface" }

[providers.openrouter]
api_key = "${OPENROUTER_API_KEY}"

[routing]
embed = "onnx"
nli = "onnx"
classify = "onnx"
chat = "openrouter"

[routing.fallbacks]
embed = ["openrouter"]
chat = ["ollama"]
```

### Resource Management

**Model lifecycle**:
```
Available → Loading → Ready → (Idle timeout) → Unloaded
                ↑                                  │
                └──────────────────────────────────┘
                         (on next request)
```

**Memory pressure handling**:
- Track loaded model memory usage
- LRU eviction when approaching limit
- Configurable memory budget

### CLI

```bash
# Service management
ratatoskr serve --config /etc/ratatoskr/config.toml
ratatoskr status
ratatoskr reload

# Model management
ratatoskr models list
ratatoskr models load embed
ratatoskr models unload nli
ratatoskr models download all-MiniLM-L6-v2

# Ad-hoc usage (for testing)
ratatoskr embed "Hello world"
ratatoskr chat "What is 2+2?"
```

### Phase 5 Questions

1. **Daemon management**: How should users run the service?
   - Systemd unit file (Linux)
   - launchd plist (macOS)
   - Docker container
   - **Recommendation**: Provide all three, document clearly

2. **Hot reload**: Should config changes take effect without restart?
   - Pro: Better uptime
   - Con: Complexity, state management
   - **Recommendation**: Support `reload` command for config, require restart for model changes

3. **Multi-tenancy**: Should one service support multiple API keys?
   - Scenario: Shared service, different users with different OpenRouter accounts
   - **Recommendation**: Not in initial service mode, add later if needed

4. **Observability**: Metrics and tracing?
   - Prometheus metrics endpoint
   - OpenTelemetry tracing
   - Structured logging
   - **Recommendation**: Start with structured logging, add metrics in Phase 6

---

## Phase 6: Advanced Features

**Goal**: Polish and production-readiness.

### 6a: Caching Layer

**Response caching**:
- Cache deterministic responses (embeddings, possibly NLI)
- Key: hash(model + input)
- Configurable TTL and size limits
- Skip for non-deterministic operations (chat with temperature > 0)

**Implementation options**:
- In-memory LRU cache (embedded mode)
- Redis/similar (service mode, shared cache)
- **Recommendation**: In-memory first, external cache later

### 6b: Retry Logic

**Smart retry for transient failures**:
```rust
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub jitter: bool,
}
```

**Retry conditions**:
- Rate limited (respect `retry-after` header)
- Server errors (5xx)
- Network errors

**Non-retry conditions**:
- Client errors (4xx except 429)
- Authentication failures
- Model not found

### 6c: Request Routing

**Multi-provider routing**:
```rust
pub struct Router {
    routes: HashMap<Capability, Vec<ProviderConfig>>,
}

impl Router {
    pub fn select(&self, capability: Capability) -> &ProviderConfig {
        // Try primary, fall back to alternatives
    }
}
```

**Routing strategies**:
- Priority-based (try in order until success)
- Cost-based (prefer cheaper providers)
- Latency-based (prefer faster providers)
- Load-balanced (distribute across providers)

### 6d: Streaming Backpressure

**Current**: Stream produces events as fast as provider sends them.

**Problem**: Slow consumer + fast producer = memory buildup.

**Solution**: Bounded channel with backpressure:
```rust
let (tx, rx) = tokio::sync::mpsc::channel(100);  // Bounded
// Producer blocks when channel full
```

### 6e: Telemetry

**Metrics** (Prometheus-style):
- `ratatoskr_requests_total{provider, operation, status}`
- `ratatoskr_request_duration_seconds{provider, operation}`
- `ratatoskr_tokens_total{provider, direction}`  // prompt vs completion
- `ratatoskr_model_memory_bytes{model}`

**Tracing** (OpenTelemetry):
- Trace IDs propagated through requests
- Spans for provider calls, model inference

### Phase 6 Questions

1. **Caching scope**: Should cache be per-gateway or shared (service mode)?
2. **Retry configuration**: Per-provider or global?
3. **Metrics cardinality**: How to handle high-cardinality labels (model names)?

---

## Cross-Cutting Concerns

### Testing Strategy

**Unit tests**: Type conversions, builder logic, error handling

**Integration tests** (recorded responses): Provider-specific parsing, streaming behavior

**Contract tests**: Verify provider responses match expected schema

**Live tests** (manual/CI-excluded): Actual API calls for smoke testing

**Load tests** (Phase 5+): Service mode performance characterization

### Documentation

**API docs**: Rustdoc for all public types and methods

**Guide**: Getting started, provider configuration, common patterns

**Examples**: Standalone chat, streaming, tool use, embeddings

**Deployment guide**: Service mode setup, systemd, Docker

### Versioning

Follow semver. The `ModelGateway` trait is the stability boundary:
- Patch: Bug fixes, no API changes
- Minor: New providers, new optional methods with defaults, new types
- Major: Breaking changes to trait signature

### Security Considerations

1. **API key handling**: Never log keys, clear from memory when possible
2. **Input validation**: Sanitize before sending to providers
3. **TLS**: Always use HTTPS for remote providers
4. **Socket permissions**: Unix socket should have appropriate permissions

---

## Dependency Management

### Core Dependencies (all phases)

| Crate | Purpose | Notes |
|-------|---------|-------|
| `tokio` | Async runtime | Features vary by phase |
| `reqwest` | HTTP client | `rustls-tls` for portability |
| `serde` / `serde_json` | Serialization | |
| `thiserror` | Error types | |
| `async-trait` | Async trait support | |
| `futures-util` | Stream utilities | |

### Phase-Specific Dependencies

| Phase | Crate | Purpose |
|-------|-------|---------|
| 4 | `ort` | ONNX Runtime |
| 4 | `tokenizers` | HF tokenizers |
| 4 | `hf-hub` | Model downloading |
| 5 | `tonic` + `prost` | gRPC (if chosen) |
| 6 | `metrics` | Prometheus metrics |
| 6 | `tracing` | Distributed tracing |

### Dependency Concerns

1. **`ort` (ONNX Runtime)**: Large, has native dependencies. Keep behind feature flag.

2. **`tonic`**: Adds protobuf codegen to build. Keep behind feature flag.

3. **`reqwest` TLS**: `rustls-tls` vs `native-tls`
   - `rustls-tls`: Pure Rust, portable, no system OpenSSL needed
   - `native-tls`: Uses system TLS, may be faster
   - **Recommendation**: Default to `rustls-tls` for portability

---

## Open Questions Summary

### Architectural

1. **Protocol**: JSON or gRPC for service mode?
2. **Provider selection**: Compile-time features, runtime config, or both?
3. **Embedded + ONNX**: Support local inference in embedded mode?

### Design

4. **Model defaults**: Built-in defaults or require explicit selection?
5. **LLM fallback**: Auto-fallback for embed/NLI when no dedicated model?
6. **Caching scope**: Per-gateway, per-process, or external (Redis)?

### Operational

7. **Multi-tenancy**: One service supporting multiple API keys?
8. **Hot reload**: Config changes without restart?
9. **Memory budget**: Automatic model eviction or manual management?

### Integration

10. **örlög coordination**: Does örlög need features beyond what's planned?
11. **chibi compatibility**: Any chibi features that need special ratatoskr support?

---

## Milestones

| Milestone | Phase | Key Deliverable |
|-----------|-------|-----------------|
| M1 | 1 | Chibi using ratatoskr for chat |
| M2 | 2 | Second provider working (HF or Ollama) |
| M3 | 3 | Embeddings via API |
| M4 | 4 | Local ONNX embeddings |
| M5 | 5 | Service mode operational |
| M6 | 5 | örlög integration complete |
| M7 | 6 | Production-ready with metrics/caching |

---

## Notes for Collaborator Discussion

Key decisions that benefit from early alignment:

1. **Phase 2 priority**: Hugging Face or Ollama first? HF has more capabilities, Ollama is simpler.

2. **Service mode protocol**: JSON (simpler) or gRPC (more structured)?

3. **örlög requirements**: Any capabilities needed that aren't in this roadmap?

4. **Timeline dependencies**: Does örlög development block on any specific ratatoskr phase?
