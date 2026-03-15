# Request Inspector: Wire-Format Request Body Callback

**Issue**: #27
**Date**: 2026-03-15
**Status**: Approved

## Motivation

Consumers (chibi, orlog) need to debug what the LLM actually sees — the exact JSON sent over HTTP. Currently, the llm crate serializes the request body for trace logging, but that's not accessible to consumers programmatically. This feature adds an optional callback that receives the serialized request JSON before it goes to the provider, with zero cost when not set.

## Design Decisions

- **Builder-level only** — one callback for the gateway's lifetime, set at construction. Per-request override is YAGNI; chibi's use case is toggling on `--debug`.
- **Stored on the provider struct** — mirrors builder-level semantics (set once, used for all requests). Minimal diff in the llm fork, easy to maintain.
- **`Fn(&str)` signature** — the caller already knows model/provider from their own context. The callback is purely "here's the wire-format body". A richer struct (`RequestInspection`) or trait is YAGNI for a single-method interface.
- **Type alias** — `RequestInspector = Arc<dyn Fn(&str) + Send + Sync>` for readability. Separate aliases in each crate (decoupled, same shape).

## Architecture

### Call Flow

```
RatatoskrBuilder.request_inspector(cb)
  → EmbeddedGateway { request_inspector: Some(cb) }
    → LlmChatProvider { request_inspector: Some(cb.clone()) }
      → LLMBuilder.request_inspector(cb.clone())
        → OpenAICompatibleProvider { request_inspector: Some(cb) }
```

All `Arc::clone` — cheap pointer bumps. Zero cost when `None`.

### llm Crate Changes

**Type alias** (in `lib.rs` or dedicated module):

```rust
pub type RequestInspector = Arc<dyn Fn(&str) + Send + Sync>;
```

**`OpenAICompatibleProvider<T>`** gains one field:

```rust
request_inspector: Option<RequestInspector>,
```

**Three serialization points** in `openai_compatible.rs` (`chat_with_tools`, `chat_stream_struct`, `chat_stream_with_tools`) change from:

```rust
if log::log_enabled!(log::Level::Trace) {
    if let Ok(json) = serde_json::to_string(&body) {
        log::trace!("...", json);
    }
}
```

to:

```rust
if self.request_inspector.is_some() || log::log_enabled!(log::Level::Trace) {
    if let Ok(json) = serde_json::to_string(&body) {
        if let Some(inspector) = &self.request_inspector {
            (inspector)(&json);
        }
        log::trace!("...", json);
    }
}
```

Serialization only happens if either the callback or trace logging is active. Both coexist. Zero overhead in the default path.

**`LLMBuilder`** gains a `.request_inspector(Arc<...>)` method; `build()` threads it into the provider.

### Ratatoskr Changes

**Type alias** (re-exported from `lib.rs`):

```rust
pub type RequestInspector = Arc<dyn Fn(&str) + Send + Sync>;
```

**`RatatoskrBuilder`** gains:

```rust
request_inspector: Option<RequestInspector>,
```

With builder method `.request_inspector(inspector)`.

**`EmbeddedGateway`** stores `Option<RequestInspector>`, passes it to `LlmChatProvider` at construction.

**`LlmChatProvider`** stores `Option<RequestInspector>`, threads it into `build_provider()` → `LLMBuilder` on every call.

**No changes needed to**: `ProviderRegistry`, `RetryingChatProvider` (transparent wrappers), `ServiceClient` (gRPC doesn't build LLM HTTP requests), proto definitions, feature flags.

### Consumer Usage

```rust
let inspector = Arc::new(|body: &str| {
    eprintln!("wire: {body}");
});

let gateway = Ratatoskr::builder()
    .openrouter(Some(api_key))
    .request_inspector(inspector)
    .build()?;
```

When `--debug` isn't active, omit `.request_inspector()`. Zero overhead.

## Scope

### In Scope

- `RequestInspector` type alias and builder method in ratatoskr
- `request_inspector` field on `EmbeddedGateway` and `LlmChatProvider`
- llm crate: field on `OpenAICompatibleProvider`, builder method on `LLMBuilder`, callback invocation at three serialization points
- Unit tests (callback fires with expected JSON, doesn't fire when unset)
- Integration test (captured JSON matches what wiremock receives)

### Out of Scope

- Response body inspection (future work)
- Per-request callback override
- Generate/embed/NLI inspection (generate routes through chat internally; embed/NLI don't go through the llm crate's HTTP serialization)
- gRPC/proto changes

## Testing Strategy

1. **llm crate unit test**: Construct provider with callback writing to `Arc<Mutex<Vec<String>>>`, fire request against wiremock, assert captured JSON contains expected fields.
2. **Ratatoskr unit test**: Full `Ratatoskr::builder()` → `EmbeddedGateway` path. Verify callback fires when set, doesn't fire when unset.
3. **Integration test**: Wiremock-based roundtrip — captured JSON matches what the server receives.
