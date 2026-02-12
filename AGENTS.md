## Project Overview

Ratatoskr is a unified LLM gateway abstraction layer. The core idea: consumers (chibi, orlog) interact only with the `ModelGateway` trait while the `llm` crate is an internal implementation detail.

**Current Status**: Phase 7 (operational hardening) complete. See `docs/plans/2026-02-11-phase7-operational-hardening.md` for the implementation plan.

## Principles

- Establish patterns now that scale well, refactor liberally when beneficial.
- Backwards compatibility not a priority, legacy code unwanted. (Pre-alpha.)
- Self-documenting code; keep symbols, comments, and docs consistent.
- Missing or incorrent documentation including code comments are critical bugs.
- Comprehensive tests including edge cases.
- Remind user about `just pre-push` before pushing and `just merge-to-dev` when merging feature branches.

## Build & Test Commands

```bash
just pre-push          # Format + clippy + test (run before pushing)
just lint              # cargo fmt + clippy
just test              # cargo nextest run --all-targets --locked
cargo doc --no-deps    # Build docs locally
```

## Git Workflow

Branches follow the pattern `feature/*`, `bugfix/*`, `refactor/*`, `chore/*`, `docs/*`, `hotfix/*`. Use justfile commands:

```bash
just feature <name>    # Create feature branch from dev
just pr                # Push and create PR to dev (auto-tags on merge)
just release v0.x      # Squash dev→main, tag release
```

## Architecture

```
consumers (chibi, orlog)
         │ uses
         ▼
ratatoskr::ModelGateway     ← stable public trait
         │ implemented by
         ├─► EmbeddedGateway    → wraps llm crate internally
         └─► ServiceClient      → connects to ratd over gRPC
                │
             ratd (daemon) → EmbeddedGateway behind gRPC handlers
```

### Project Structure

```
src/
├── lib.rs              # Public API re-exports
├── error.rs            # RatatoskrError, Result
├── traits.rs           # ModelGateway trait
├── types/              # Message, Tool, ChatOptions, ChatEvent, GenerateOptions, etc.
│   ├── stance.rs       # StanceResult, StanceLabel
│   ├── model.rs        # ModelInfo, ModelStatus, ModelCapability, ModelMetadata, PricingInfo
│   ├── parameter.rs    # ParameterName, ParameterAvailability, ParameterRange
│   ├── validation.rs   # ParameterValidationPolicy
│   └── token.rs        # Token (detailed tokenization)
├── registry/           # Model metadata registry (Phase 6)
│   ├── mod.rs          # ModelRegistry with layered merge
│   ├── remote.rs       # Remote registry fetch, local cache, versioned format
│   └── seed.json       # Embedded model metadata (compiled-in fallback)
├── cache/
│   ├── mod.rs          # ModelCache — ephemeral provider-fetched metadata store
│   ├── discovery.rs    # ParameterDiscoveryCache — runtime parameter rejection cache (moka)
│   └── response.rs     # ResponseCache — opt-in LRU+TTL cache for embed/NLI (moka)
├── gateway/
│   ├── embedded.rs     # EmbeddedGateway delegating to ProviderRegistry
│   └── builder.rs      # Ratatoskr::builder()
├── telemetry.rs        # Metric name constants (ratatoskr_requests_total, etc.)
├── providers/          # Provider implementations and traits (feature-gated)
│   ├── traits.rs       # EmbeddingProvider, NliProvider, StanceProvider, etc.
│   ├── registry.rs     # ProviderRegistry (fallback chains, latency tracking, cost routing)
│   ├── retry.rs        # RetryConfig, RetryingProvider<T> decorators, with_retry() helper
│   ├── routing.rs      # RoutingConfig, ProviderLatency (EWMA), ProviderCostInfo
│   ├── backpressure.rs # Bounded channel wrapper for streaming backpressure
│   ├── llm_chat.rs     # LlmChatProvider wrapping llm crate (+ fetch_metadata for OpenRouter)
│   ├── workarounds.rs  # Provider-specific parameter translation (parallel_tool_calls, extra_body)
│   ├── openrouter_models.rs  # OpenRouter /api/v1/models response types + conversion
│   ├── huggingface.rs  # HuggingFace Inference API client
│   ├── fastembed.rs    # Local embeddings via fastembed-rs
│   └── onnx_nli.rs     # Local NLI via ONNX Runtime
├── server/             # gRPC server + shared proto types (server/client features)
│   ├── mod.rs          # Proto re-exports, module gating
│   ├── convert.rs      # Bidirectional proto ↔ native conversions
│   ├── service.rs      # RatatoskrService<G> gRPC handler (server feature)
│   └── config.rs       # Config + Secrets loading from TOML (server feature)
├── client/             # gRPC client library (client feature)
│   ├── mod.rs          # ServiceClient re-export
│   └── service_client.rs  # ServiceClient implementing ModelGateway over gRPC
├── bin/
│   ├── ratd.rs         # Daemon entry point — starts gRPC server
│   └── rat.rs          # CLI client — health, models, status, chat, embed, nli, tokens, metadata
├── tokenizer/          # Token counting (local-inference feature)
│   └── mod.rs          # TokenizerRegistry, HfTokenizer
├── model/              # Model management (local-inference feature)
│   ├── manager.rs      # ModelManager with RAM budget tracking
│   └── device.rs       # Device enum (CPU, CUDA)
└── convert/            # ratatoskr ↔ llm type conversions (internal)

proto/
└── ratatoskr.proto     # gRPC service definition (17 RPCs)

contrib/
└── systemd/
    └── ratd.service    # systemd unit with security hardening
```

### Key Types

- `ModelGateway` — async trait with `chat()`, `chat_stream()`, `embed()`, `infer_nli()`, `classify_stance()`, `model_metadata()`, `fetch_model_metadata()`, etc.
- `Message` — role (System/User/Assistant/Tool) + content + optional tool_calls
- `ChatEvent` — streaming events: Content, Reasoning, ToolCallStart, ToolCallDelta, Usage, Done
- `ChatOptions` — model, temperature, max_tokens, top_k, reasoning config, tool_choice, parallel_tool_calls, etc.
- `GenerateOptions` — model, temperature, max_tokens, top_k, frequency/presence penalty, seed, reasoning
- `RatatoskrError` — comprehensive error enum; `ModelNotAvailable` triggers fallback, `UnsupportedParameter` for validation errors
- `StanceResult` — stance detection result (favor/against/neutral scores with label)
- `ProviderRegistry` — fallback chains per capability with opt-in parameter validation; `fetch_chat_metadata()` for on-demand fetch
- `ModelRegistry` — centralized model metadata with three-layer merge (embedded seed → cached remote → live data)
- `RemoteRegistryConfig` — URL + cache path for the remote registry; default: `emesal/ratatoskr-registry` on GitHub
- `RemoteRegistry` — versioned payload wrapper (`{ "version": 1, "models": [...] }`) with legacy bare-array fallback
- `ModelCache` — ephemeral thread-safe cache for provider-fetched metadata (consulted after registry miss)
- `ModelMetadata` — extended model info: capabilities, parameters, pricing, max output tokens
- `ParameterName` — hybrid enum (well-known params + `Custom(String)` escape hatch)
- `ParameterAvailability` — mutable/read-only/opaque/unsupported per parameter
- `ParameterValidationPolicy` — warn/error/ignore for unsupported parameters
- `ServiceClient` — `ModelGateway` impl that forwards to ratd over gRPC (client feature)
- `RatatoskrService<G>` — wraps any `ModelGateway` behind gRPC handlers (server feature)
- `RetryConfig` — retry behaviour (max attempts, exponential backoff, jitter, retry-after support)
- `RetryingProvider<T>` — decorator wrapping any provider trait with retry logic on transient errors
- `RoutingConfig` — preferred provider per capability (reorders fallback chain)
- `ProviderLatency` — EWMA-based per-provider latency tracker (thread-safe atomics)
- `ProviderCostInfo` — cost ranking for providers serving a model (sorted cheapest-first)
- `CacheConfig` — opt-in response cache configuration (max entries, TTL)
- `ResponseCache` — moka-backed LRU+TTL cache for embed/NLI responses
- `DiscoveryConfig` — configuration for runtime parameter discovery cache (max entries, TTL)
- `ParameterDiscoveryCache` — moka-backed cache recording parameter rejections at runtime; consulted during validation to prevent repeated failures
- `DiscoveryRecord` — a single parameter rejection (parameter, provider, model, timestamp, reason); forward-compatible with #14 aggregation

### Builder Pattern

```rust
Ratatoskr::builder()
    .openrouter(api_key)
    .anthropic(anthropic_key)
    .ollama("http://localhost:11434")
    .huggingface(hf_key)              // requires `huggingface` feature
    .local_embeddings(model)          // requires `local-inference` feature
    .local_nli(model)                 // requires `local-inference` feature
    .device(Device::Cpu)              // or Device::Cuda { device_id: 0 }
    .ram_budget(1024 * 1024 * 1024)   // optional: 1GB RAM budget for local models
    .retry(RetryConfig::new().max_attempts(5))      // phase 7: retry on transient errors
    .routing(RoutingConfig::new().chat("anthropic")) // phase 7: preferred provider routing
    .response_cache(CacheConfig::default())          // phase 7: cache embed/NLI responses
    .discovery(DiscoveryConfig::new().ttl(Duration::from_secs(12 * 3600)))  // optional: tune discovery
    // .disable_parameter_discovery()                 // opt-out: disable runtime discovery
    .registry_url("https://...")                     // optional: remote registry (loads cached only)
    // .remote_registry(RemoteRegistryConfig::default()) // full control over URL + cache path
    .build()?
```

**Provider Priority:** Local providers are registered first (higher priority), API providers as fallbacks.
When a local provider returns `ModelNotAvailable` (wrong model or RAM budget exceeded), the registry
tries the next provider in the chain.

### HuggingFace Capabilities (Phase 2)

With the `huggingface` feature enabled, the gateway supports:
- `embed(text, model)` — single text embeddings
- `embed_batch(texts, model)` — batch embeddings
- `infer_nli(premise, hypothesis, model)` — natural language inference
- `classify_zero_shot(text, labels, model)` — zero-shot classification
- `classify_stance(text, target, model)` — stance detection (via ZeroShotStanceProvider wrapper)

Models are HuggingFace model IDs, e.g., `sentence-transformers/all-MiniLM-L6-v2`, `facebook/bart-large-mnli`.

### Local Inference Capabilities (Phase 3-4)

With the `local-inference` feature enabled:
- `generate(prompt, options)` — non-streaming text generation
- `generate_stream(prompt, options)` — streaming text generation
- `count_tokens(text, model)` — token counting with model-appropriate tokenizers
- `tokenize(text, model)` — detailed tokenization with offsets
- Local embeddings via `LocalEmbeddingProvider` (fastembed-rs)
- Local NLI via `LocalNliProvider` (ONNX Runtime)
- `ModelManager` for lazy model loading with RAM budget tracking
- `TokenizerRegistry` with default mappings for common models
- `list_models()` and `model_status()` for introspection

Supported embedding models: `AllMiniLmL6V2`, `AllMiniLmL12V2`, `BgeSmallEn`, `BgeBaseEn`
Supported NLI models: `NliDebertaV3Base`, `NliDebertaV3Small`, or custom ONNX models

### Service Mode (Phase 5)

With the `server` and `client` features enabled:
- `ratd` — daemon binary serving `EmbeddedGateway` over gRPC (default `127.0.0.1:9741`)
- `rat` — CLI client with subcommands: `health`, `models`, `status`, `chat`, `embed`, `nli`, `tokens`, `metadata`, `update-registry`
- `ServiceClient` — implements `ModelGateway` trait, transparently forwarding all calls over gRPC
- TOML configuration with provider/routing/limits sections
- Separate secrets file (`~/.ratatoskr/secrets.toml`) with 0600 permission enforcement
- Proto conversions centralized in `server::convert` (shared by both server and client)

### Model Intelligence (Phase 6)

- `ModelRegistry` — centralized model metadata with three-layer merge (embedded seed → cached remote → live)
- `ModelCache` — ephemeral store for metadata fetched on cache miss from provider APIs
- `model_metadata(model)` — sync lookup: registry (curated) → cache (ephemeral)
- `fetch_model_metadata(model)` — async: walks chat provider chain, populates cache on success
- `ParameterName` — hybrid enum with well-known params + `Custom(String)` for provider-specific options
- `ParameterAvailability` — mutable (with range), read-only, opaque, or unsupported per parameter
- `ParameterValidationPolicy` — opt-in validation (warn/error/ignore) when providers declare their supported params
- `ChatOptions` and `GenerateOptions` at full parity: temperature, top_p, top_k, max_tokens, frequency/presence penalty, seed, reasoning, parallel_tool_calls
- `raw_provider_options` escape hatch for provider-specific JSON options
- `workarounds` module isolates provider-specific parameter translation (e.g. `parallel_tool_calls` via `extra_body` for OpenAI-compatible backends, native flag for Mistral, `UnsupportedParameter` for Anthropic direct)
- Proto `GetModelMetadata` + `FetchModelMetadata` RPCs for remote metadata queries via `ServiceClient`

### Operational Hardening (Phase 7)

- **Retry logic**: `RetryConfig` with exponential backoff + jitter. `RetryingProvider<T>` decorators auto-wrap providers at registration. Transient errors (`is_transient()`) are retried; permanent errors (auth, validation) are not. Exhausted retries trigger cross-provider fallback in the registry.
- **Streaming backpressure**: Bounded `tokio::sync::mpsc::channel` wraps `chat_stream`/`generate_stream` output. Producer blocks when consumer falls behind (default buffer: 64).
- **Response cache**: Opt-in moka-backed LRU+TTL cache for deterministic operations (embed, NLI). Zero overhead when not configured. Cache key = `hash(operation, model, input)`. Emits `ratatoskr_cache_hits_total` / `ratatoskr_cache_misses_total` metrics.
- **Telemetry**: `#[instrument]` tracing spans on dispatch and gateway paths. `metrics` crate integration: `ratatoskr_requests_total`, `ratatoskr_request_duration_seconds`, `ratatoskr_retries_total`, `ratatoskr_tokens_total`. Recorder-agnostic (consumers install their own backend).
- **Routing**: `RoutingConfig` reorders fallback chains per capability (preferred provider first). `ProviderLatency` tracks EWMA per provider. `providers_by_cost()` returns providers sorted cheapest-first via `ModelMetadata.pricing`. ratd TOML `[routing]` section now wired through to the builder.
- **Parameter discovery**: `ParameterDiscoveryCache` records runtime parameter rejections (`UnsupportedParameter` errors) keyed on `(provider, model, parameter)`. Validation consults this cache alongside static declarations. On by default (1,000 entries, 24h TTL); opt-out via `.disable_parameter_discovery()`. Emits `ratatoskr_parameter_discoveries_total` metric. ratd TOML `[discovery]` section for configuration.

## Testing Strategy

1. **Unit tests** — types, conversions, builder (fast, no I/O)
2. **Integration tests** — wiremock with recorded responses
3. **Live tests** — `#[ignore]`, manual verification only

Full test coverage required.

### HuggingFace Live Tests

```bash
HF_API_KEY=hf_xxx cargo test --test huggingface_live_test --features huggingface -- --ignored
```

### Local Inference Live Tests

```bash
cargo test --test local_inference_live_test --features local-inference -- --ignored
```

Note: First run downloads models (~100MB+ for embeddings, ~500MB+ for NLI).

### Service Mode Tests

```bash
cargo test --features server,client --test service_test
```

Live tests require a running ratd instance with valid API keys.

### OpenRouter Metadata Live Tests

```bash
OPENROUTER_API_KEY=sk-or-xxx cargo test --test openrouter_metadata_live_test -- --ignored
```

## Phase Roadmap

- Phase 1: OpenRouter Chat ✓
- Phase 2: HuggingFace provider (embeddings, NLI, classification) ✓
- Phase 3-4: Local inference (embeddings, NLI, tokenizers, generate) ✓
- Phase 5: Service mode (gRPC daemon + CLI client) ✓
- Phase 6: Model intelligence & parameter surface (metadata, registry, validation) ✓
- Phase 7: Operational hardening (retry, telemetry, backpressure, caching, routing) ✓
