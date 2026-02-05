## Project Overview

Ratatoskr is a unified LLM gateway abstraction layer. The core idea: consumers (chibi, orlog) interact only with the `ModelGateway` trait while the `llm` crate is an internal implementation detail.

**Current Status**: Phase 5 (service mode) complete. See `docs/plans/2026-02-04-phase5-implementation.md` for details.

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
│   ├── model.rs        # ModelInfo, ModelStatus, ModelCapability
│   └── token.rs        # Token (detailed tokenization)
├── gateway/
│   ├── embedded.rs     # EmbeddedGateway delegating to ProviderRegistry
│   └── builder.rs      # Ratatoskr::builder()
├── providers/          # Provider implementations and traits (feature-gated)
│   ├── traits.rs       # EmbeddingProvider, NliProvider, StanceProvider, etc.
│   ├── registry.rs     # ProviderRegistry (fallback chains per capability)
│   ├── llm_chat.rs     # LlmChatProvider wrapping llm crate
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
│   └── rat.rs          # CLI client — health, models, chat, embed, nli, tokens
├── tokenizer/          # Token counting (local-inference feature)
│   └── mod.rs          # TokenizerRegistry, HfTokenizer
├── model/              # Model management (local-inference feature)
│   ├── manager.rs      # ModelManager with RAM budget tracking
│   └── device.rs       # Device enum (CPU, CUDA)
└── convert/            # ratatoskr ↔ llm type conversions (internal)

proto/
└── ratatoskr.proto     # gRPC service definition (14 RPCs)

contrib/
└── systemd/
    └── ratd.service    # systemd unit with security hardening
```

### Key Types

- `ModelGateway` — async trait with `chat()`, `chat_stream()`, `embed()`, `infer_nli()`, `classify_stance()`, etc.
- `Message` — role (System/User/Assistant/Tool) + content + optional tool_calls
- `ChatEvent` — streaming events: Content, Reasoning, ToolCallStart, ToolCallDelta, Usage, Done
- `ChatOptions` — model, temperature, max_tokens, reasoning config, tool_choice, etc.
- `RatatoskrError` — comprehensive error enum; `ModelNotAvailable` triggers fallback in registry
- `StanceResult` — stance detection result (favor/against/neutral scores with label)
- `ProviderRegistry` — fallback chains per capability (tries providers in priority order)
- `ServiceClient` — `ModelGateway` impl that forwards to ratd over gRPC (client feature)
- `RatatoskrService<G>` — wraps any `ModelGateway` behind gRPC handlers (server feature)

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
- `rat` — CLI client with subcommands: `health`, `models`, `status`, `chat`, `embed`, `nli`, `tokens`
- `ServiceClient` — implements `ModelGateway` trait, transparently forwarding all calls over gRPC
- TOML configuration with provider/routing/limits sections
- Separate secrets file (`~/.config/ratatoskr/secrets.toml`) with 0600 permission enforcement
- Proto conversions centralized in `server::convert` (shared by both server and client)

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

## Phase Roadmap

- Phase 1: OpenRouter Chat ✓
- Phase 2: HuggingFace provider (embeddings, NLI, classification) ✓
- Phase 3-4: Local inference (embeddings, NLI, tokenizers, generate) ✓
- Phase 5: Service mode (gRPC daemon + CLI client) ✓
- Phase 6: Model intelligence & parameter surface (metadata, registry, validation)
- Phase 7: Operational hardening (caching, retry, routing, telemetry)
