## Project Overview

Ratatoskr is a unified LLM gateway abstraction layer. The core idea: consumers (chibi, orlog) interact only with the `ModelGateway` trait while the `llm` crate is an internal implementation detail.

**Current Status**: Phase 1 (OpenRouter Chat Gateway) is designed but not yet implemented. See `docs/plans/2026-02-01-phase1-implementation.md` for the 13-task TDD implementation plan.

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

### Planned Project Structure

```
src/
├── lib.rs              # Public API re-exports
├── error.rs            # RatatoskrError, Result
├── traits.rs           # ModelGateway trait
├── types/              # Message, Tool, ChatOptions, ChatEvent, etc.
├── gateway/
│   ├── embedded.rs     # EmbeddedGateway wrapping llm crate
│   └── builder.rs      # Ratatoskr::builder()
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
    .build()?
```

Model string determines routing: `"anthropic/claude-sonnet-4"` → openrouter, `"claude-sonnet-4"` → direct anthropic.

## Testing Strategy

1. **Unit tests** — types, conversions, builder (fast, no I/O)
2. **Integration tests** — wiremock with recorded responses
3. **Live tests** — `#[ignore]`, manual verification only

Full test coverage required.

## Phase Roadmap

- Phase 1: OpenRouter Chat (current)
- Phase 2: Additional providers (HuggingFace, Ollama, Anthropic direct)
- Phase 3: Embeddings & NLI
- Phase 4: ONNX local inference
- Phase 5: Service mode (gRPC/socket)
- Phase 6: Caching, metrics, routing
