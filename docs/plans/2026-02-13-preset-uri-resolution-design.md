# Preset URI Resolution Design

**Status: implemented**

> resolves ratatoskr#23

## goal

model strings with the `ratatoskr:` prefix are resolved transparently to concrete model IDs via the preset table. downstream tools pass preset URIs as ordinary model strings — no special handling needed on their side.

## format

```
ratatoskr:<tier>/<capability>
```

both tier and capability are **required**. examples:

```
ratatoskr:free/agentic          → google/gemini-2.0-flash-001
ratatoskr:free/text-generation  → google/gemini-2.0-flash-001
ratatoskr:budget/agentic        → openai/gpt-4o-mini
ratatoskr:premium/agentic       → anthropic/claude-sonnet-4
ratatoskr:free/embedding        → sentence-transformers/all-MiniLM-L6-v2
```

`ratatoskr:free` (no capability) is an error: "preset URI must be `ratatoskr:<tier>/<capability>`".

## where

resolution lives in `EmbeddedGateway`. it has access to both the `ProviderRegistry` (for API calls) and the `ModelRegistry` (for preset lookups).

### resolution method

private method on `EmbeddedGateway`:

```rust
fn resolve_model_string(&self, model: &str) -> Result<String>
```

- strips `ratatoskr:` prefix
- splits on `/` → `(tier, capability)`
- calls `self.model_registry.preset(tier, capability)`
- returns the concrete model ID, or `PresetNotFound` error

### call sites

resolve before delegating to the provider registry:

- `chat` / `chat_stream`
- `generate` / `generate_stream`
- `model_metadata` / `fetch_model_metadata`

pattern:

```rust
let resolved_model = self.resolve_model_string(&options.model)?;
let options = ChatOptions { model: resolved_model, ..options.clone() };
self.registry.chat(messages, tools, &options).await
```

non-preset model strings pass through unchanged (no clone needed).

### error handling

new error variant:

```rust
RatatoskrError::PresetNotFound { tier: String, capability: String }
```

covers: tier doesn't exist, capability doesn't exist within tier, malformed URI (no `/`).

## testing

- `ratatoskr:free/agentic` → resolves to seeded model
- `ratatoskr:free/nonexistent` → `PresetNotFound`
- `ratatoskr:nonexistent/agentic` → `PresetNotFound`
- `ratatoskr:free` (no capability) → error with clear message
- `ratatoskr:` (empty) → error
- `anthropic/claude-sonnet-4` → passes through unchanged
- resolution works for chat, generate, model_metadata

## context

this enables chibi's zero-config story (emesal/chibi-dev#141). chibi's default model will be `ratatoskr:free/agentic` when no config.toml exists.
