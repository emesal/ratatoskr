## Project Overview

Ratatoskr is a unified LLM gateway abstraction layer. The core idea: consumers (chibi, orlog) interact only with the `ModelGateway` trait while the `llm` crate is an internal implementation detail.

**Current Status**: Phase 3-4 (Local Inference) nearly complete. See `docs/plans/2026-02-02-phase3-4-local-inference.md` for details.

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
├── gateway/
│   ├── embedded.rs     # EmbeddedGateway wrapping llm crate
│   ├── builder.rs      # Ratatoskr::builder()
│   └── routing.rs      # CapabilityRouter for provider selection
├── providers/          # External provider clients (feature-gated)
│   ├── huggingface.rs  # HuggingFace Inference API client
│   ├── fastembed.rs    # Local embeddings via fastembed-rs
│   └── onnx_nli.rs     # Local NLI via ONNX Runtime
├── tokenizer/          # Token counting (local-inference feature)
│   └── mod.rs          # TokenizerRegistry, HfTokenizer
├── model/              # Model management (local-inference feature)
│   ├── manager.rs      # ModelManager for lazy loading
│   └── device.rs       # Device enum (CPU, CUDA)
└── convert/            # ratatoskr ↔ llm type conversions (internal)
```

### Key Types

- `ModelGateway` — async trait with `chat()`, `chat_stream()`, `capabilities()`; future methods return `NotImplemented`
- `Message` — role (System/User/Assistant/Tool) + content + optional tool_calls
- `ChatEvent` — streaming events: Content, Reasoning, ToolCallStart, ToolCallDelta, Usage, Done
- `ChatOptions` — model, temperature, max_tokens, reasoning config, tool_choice, etc.
- `RatatoskrError` — comprehensive error enum wrapping provider/network/data/configuration errors

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
    .build()?
```

Model string determines routing: `"anthropic/claude-sonnet-4"` → openrouter, `"claude-sonnet-4"` → direct anthropic.

### HuggingFace Capabilities (Phase 2)

With the `huggingface` feature enabled, the gateway supports:
- `embed(text, model)` — single text embeddings
- `embed_batch(texts, model)` — batch embeddings
- `infer_nli(premise, hypothesis, model)` — natural language inference
- `classify_zero_shot(text, labels, model)` — zero-shot classification

Models are HuggingFace model IDs, e.g., `sentence-transformers/all-MiniLM-L6-v2`, `facebook/bart-large-mnli`.

### Local Inference Capabilities (Phase 3-4)

With the `local-inference` feature enabled:
- `generate(prompt, options)` — non-streaming text generation
- `generate_stream(prompt, options)` — streaming text generation
- `count_tokens(text, model)` — token counting with model-appropriate tokenizers
- Local embeddings via `FastEmbedProvider` (fastembed-rs)
- Local NLI via `OnnxNliProvider` (ONNX Runtime)
- `ModelManager` for lazy model loading with caching
- `TokenizerRegistry` with default mappings for common models

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
