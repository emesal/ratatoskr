# keyless openrouter support (#24)

## summary

enable keyless openrouter access for free-tier models. openrouter allows unauthenticated requests to free models — the llm crate and ratatoskr should model this correctly.

motivation: chibi zero-config (`cargo install chibi && chibi "hello"`) needs a working chat provider with no API key.

## design

### llm crate changes

**`OpenAIProviderConfig` trait** — new associated const:

```rust
const REQUIRES_AUTH: bool = true;  // default preserves existing behaviour
```

**`OpenAICompatibleProvider`** — two mechanical changes at 3 call sites each (chat, chat_with_history, chat_streaming):

1. auth guard: `if T::REQUIRES_AUTH && self.config.api_key.is_empty() { ... }`
2. bearer auth: only add `Authorization` header when key is non-empty

**`OpenRouterConfig`** — override:

```rust
const REQUIRES_AUTH: bool = false;
```

**`OpenRouter::list_models`** — same pattern: drop empty-key guard, conditional `bearer_auth`.

backwards-compatible: all other providers default to `REQUIRES_AUTH = true`.

### ratatoskr changes

**`RatatoskrBuilder`**:

- `.openrouter(api_key: Option<impl Into<String>>)` — `None` = keyless
- new `openrouter_enabled: bool` field, set `true` on any `.openrouter()` call
- `build()`: register openrouter when `openrouter_enabled`, pass `""` as key when `openrouter_key` is `None`

**`LlmChatProvider`**:

- `api_key: String` → `api_key: Option<String>`
- `new()` / `with_http_client()`: `api_key: Option<impl Into<String>>`
- llm builder: conditionally call `.api_key()` only when `Some`
- metadata fetch: conditionally add `.bearer_auth()` only when `Some`

**ratd TOML config**:

- `[providers.openrouter]` section's `api_key` becomes optional
- section present without `api_key` = keyless openrouter

### testing

**llm crate:**

- `OpenRouterConfig::REQUIRES_AUTH == false`
- other configs have `REQUIRES_AUTH == true`
- wiremock: keyless request has no `Authorization` header
- wiremock: keyed request has correct `Authorization` header

**ratatoskr:**

- builder: `.openrouter(None::<String>)` registers chat provider
- builder: `.openrouter(Some("sk-xxx"))` still works
- builder: keyless `.openrouter()` → `has_chat_provider() == true`
- wiremock: keyless chat has no `Authorization` header
- wiremock: keyed chat has `Authorization: Bearer sk-xxx`

### out of scope

- preset URI resolution (`ratatoskr:free/agentic`)
- keyless support for other providers
- chibi integration (downstream consumer)
