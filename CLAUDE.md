## Project Overview

Ratatoskr is a unified LLM gateway abstraction layer. The core idea: consumers (chibi, orlog) interact only with the `ModelGateway` trait while the `llm` crate is an internal implementation detail.

**Current Status**: Provider Trait Refactor in progress. See `docs/plans/2026-02-03-provider-trait-refactor.md` for details.

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
ratatoskr::ModelGateway  ← stable public trait
         │ implemented by
         ▼
EmbeddedGateway (phase 1) → wraps llm crate internally
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
├── tokenizer/          # Token counting (local-inference feature)
│   └── mod.rs          # TokenizerRegistry, HfTokenizer
├── model/              # Model management (local-inference feature)
│   ├── manager.rs      # ModelManager with RAM budget tracking
│   └── device.rs       # Device enum (CPU, CUDA)
└── convert/            # ratatoskr ↔ llm type conversions (internal)
```

### Key Types

- `ModelGateway` — async trait with `chat()`, `chat_stream()`, `embed()`, `infer_nli()`, `classify_stance()`, etc.
- `Message` — role (System/User/Assistant/Tool) + content + optional tool_calls
- `ChatEvent` — streaming events: Content, Reasoning, ToolCallStart, ToolCallDelta, Usage, Done
- `ChatOptions` — model, temperature, max_tokens, reasoning config, tool_choice, etc.
- `RatatoskrError` — comprehensive error enum; `ModelNotAvailable` triggers fallback in registry
- `StanceResult` — stance detection result (favor/against/neutral scores with label)
- `ProviderRegistry` — fallback chains per capability (tries providers in priority order)

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

## Phase Roadmap

- Phase 1: OpenRouter Chat ✓
- Phase 2: HuggingFace provider (embeddings, NLI, classification) ✓
- Phase 3-4: Local inference (embeddings, NLI, tokenizers, generate) ✓
- Phase 5: Service mode (gRPC/socket)
- Phase 6: Caching, metrics, advanced routing
