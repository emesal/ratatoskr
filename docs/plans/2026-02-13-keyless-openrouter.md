# Keyless OpenRouter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable keyless openrouter access for free-tier models across the llm crate and ratatoskr.

**Architecture:** Add `REQUIRES_AUTH` const to llm's `OpenAIProviderConfig` trait (default `true`, openrouter overrides `false`). Conditional auth guards + `bearer_auth` headers. Ratatoskr builder signature changes to `Option<impl Into<String>>`, `LlmChatProvider.api_key` becomes `Option<String>`.

**Tech Stack:** Rust, llm crate (local fork at `/home/fey/projects/llm/llm-emesal`), ratatoskr, wiremock

**Refs:** Design doc: `docs/plans/2026-02-13-keyless-openrouter-design.md`, issue: #24

---

### Task 1: llm crate — add `REQUIRES_AUTH` to `OpenAIProviderConfig` trait

**Files:**
- Modify: `/home/fey/projects/llm/llm-emesal/src/providers/openai_compatible.rs:89-110` (trait definition)

**Step 1: Add the const to the trait**

In the `OpenAIProviderConfig` trait (line 89), add after `SUPPORTS_STREAM_OPTIONS` (line 105):

```rust
    /// Whether this provider requires an API key for requests.
    /// When `false`, requests may be sent without an `Authorization` header.
    const REQUIRES_AUTH: bool = true;
```

**Step 2: Verify it compiles**

Run: `cd /home/fey/projects/llm/llm-emesal && cargo check`
Expected: compiles cleanly (default value = backwards compatible)

**Step 3: Commit**

```bash
cd /home/fey/projects/llm/llm-emesal
git add src/providers/openai_compatible.rs
git commit -m "feat: add REQUIRES_AUTH to OpenAIProviderConfig trait

Default true preserves existing behaviour. Providers that support
keyless access (e.g. OpenRouter for free models) can override to false."
```

---

### Task 2: llm crate — conditional auth guards and bearer_auth in `OpenAICompatibleProvider`

**Files:**
- Modify: `/home/fey/projects/llm/llm-emesal/src/providers/openai_compatible.rs`
  - Lines 574-577 (`chat` auth guard)
  - Line 628 (`chat` bearer_auth)
  - Lines 705-708 (`chat_with_history` auth guard)
  - Line 750 (`chat_with_history` bearer_auth)
  - Lines 799-802 (`chat_streaming` auth guard)
  - Line 853 (`chat_streaming` bearer_auth)

**Step 1: Make auth guards conditional on `T::REQUIRES_AUTH`**

At each of the three `api_key.is_empty()` guards (lines 574, 705, 799), change from:

```rust
        if self.config.api_key.is_empty() {
```

to:

```rust
        if T::REQUIRES_AUTH && self.config.api_key.is_empty() {
```

**Step 2: Make bearer_auth conditional on non-empty key**

At each of the three `.bearer_auth(...)` sites (lines 628, 750, 853), change from:

```rust
            .bearer_auth(&self.config.api_key)
```

to:

```rust
            .bearer_auth_if(!self.config.api_key.is_empty(), &self.config.api_key)
```

Wait — `reqwest::RequestBuilder` doesn't have `bearer_auth_if`. Instead, conditionally add the header. Replace each `.bearer_auth(&self.config.api_key)` call pattern. For example, the block around line 625-629:

```rust
        let mut request = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&body);
```

becomes:

```rust
        let mut request = self
            .client
            .post(url)
            .json(&body);
        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }
```

Apply this same pattern at all three sites (lines ~625-629, ~747-751, ~850-854). The `.json(&body)` is chained before the conditional auth.

**Step 3: Verify it compiles**

Run: `cd /home/fey/projects/llm/llm-emesal && cargo check`
Expected: compiles cleanly

**Step 4: Commit**

```bash
cd /home/fey/projects/llm/llm-emesal
git add src/providers/openai_compatible.rs
git commit -m "feat: conditional auth guards and bearer_auth in OpenAICompatibleProvider

Auth guard now checks T::REQUIRES_AUTH before rejecting empty keys.
Bearer auth header only sent when api_key is non-empty. This enables
keyless access for providers that support it (e.g. OpenRouter free tier)."
```

---

### Task 3: llm crate — set `REQUIRES_AUTH = false` for OpenRouter + fix `list_models`

**Files:**
- Modify: `/home/fey/projects/llm/llm-emesal/src/backends/openrouter.rs`
  - Line 22-29 (OpenRouterConfig impl)
  - Lines 122-133 (list_models auth guard + bearer_auth)

**Step 1: Override `REQUIRES_AUTH` in OpenRouterConfig**

In the `impl OpenAIProviderConfig for OpenRouterConfig` block (line 22), add:

```rust
    const REQUIRES_AUTH: bool = false;
```

**Step 2: Fix `list_models`**

Remove the empty-key guard (lines 122-126) and make bearer_auth conditional:

```rust
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = format!("{}models", OpenRouterConfig::DEFAULT_BASE_URL);

        let mut request = self.client.get(&url);
        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }

        let resp = request.send().await?.error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::OpenRouter,
        };
        Ok(Box::new(result))
    }
```

**Step 3: Verify it compiles**

Run: `cd /home/fey/projects/llm/llm-emesal && cargo check`

**Step 4: Commit**

```bash
cd /home/fey/projects/llm/llm-emesal
git add src/backends/openrouter.rs
git commit -m "feat: keyless OpenRouter — REQUIRES_AUTH = false, conditional list_models auth

OpenRouter supports unauthenticated access for free-tier models.
The /api/v1/models endpoint is also public."
```

---

### Task 4: llm crate — tests for REQUIRES_AUTH

**Files:**
- Modify: `/home/fey/projects/llm/llm-emesal/src/backends/openrouter.rs` (add test module)

**Step 1: Write tests**

Add at the bottom of `openrouter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::OpenAIProviderConfig;

    #[test]
    fn openrouter_does_not_require_auth() {
        assert!(!OpenRouterConfig::REQUIRES_AUTH);
    }

    #[test]
    fn openai_requires_auth() {
        use crate::backends::openai::OpenAIConfig;
        assert!(OpenAIConfig::REQUIRES_AUTH);
    }
}
```

**Step 2: Run tests**

Run: `cd /home/fey/projects/llm/llm-emesal && cargo test --lib -- openrouter::tests`
Expected: 2 tests pass

**Step 3: Commit**

```bash
cd /home/fey/projects/llm/llm-emesal
git add src/backends/openrouter.rs
git commit -m "test: verify REQUIRES_AUTH for OpenRouter and OpenAI"
```

---

### Task 5: ratatoskr — make `LlmChatProvider.api_key` optional

**Files:**
- Modify: `src/providers/llm_chat.rs`
  - Line 42 (field type)
  - Lines 55-64 (`new()`)
  - Lines 66-85 (`with_http_client()`)
  - Line 120 (`.api_key()` call in builder)
  - Line 360 (`.bearer_auth()` in metadata fetch)

**Step 1: Change the field type**

Line 42, change:

```rust
    api_key: String,
```

to:

```rust
    api_key: Option<String>,
```

**Step 2: Update `new()` and `with_http_client()` signatures**

`new()` (line 63):

```rust
    pub fn new(backend: LLMBackend, api_key: Option<impl Into<String>>, name: impl Into<String>) -> Self {
        Self::with_http_client(backend, api_key, name, reqwest::Client::new())
    }
```

`with_http_client()` (lines 71-84):

```rust
    pub fn with_http_client(
        backend: LLMBackend,
        api_key: Option<impl Into<String>>,
        name: impl Into<String>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            backend,
            api_key: api_key.map(|k| k.into()),
            name: name.into(),
            ollama_url: None,
            timeout_secs: 120,
            http_client,
            models_base_url: None,
        }
    }
```

**Step 3: Conditional `.api_key()` on LLM builder**

At line 120 (inside the method that builds the llm provider), change:

```rust
            .api_key(&self.api_key)
```

to:

```rust
```

And add after the builder chain, before the system prompt block:

```rust
        if let Some(ref key) = self.api_key {
            builder = builder.api_key(key);
        }
```

**Step 4: Conditional `bearer_auth` in metadata fetch**

At line 360, change:

```rust
            .bearer_auth(&self.api_key)
```

to conditional form — find the request builder chain around lines 356-362 and change:

```rust
        let mut request = self
            .http_client
            .get(url);
        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;
```

**Step 5: Update doc comment on `new()`**

Update the `/// * `api_key` - API key for the backend` line to:

```rust
    /// * `api_key` - API key for the backend (`None` for keyless access)
```

**Step 6: Verify it compiles**

Run: `cd /home/fey/projects/ratatoskr/rat-dev && cargo check`
Expected: compilation errors in builder.rs and test files (callers still pass `String`)

**Step 7: Commit (WIP — callers not yet updated)**

Don't commit yet — continue to Task 6.

---

### Task 6: ratatoskr — update builder for optional openrouter key

**Files:**
- Modify: `src/gateway/builder.rs`
  - Line 33-34 (add `openrouter_enabled` field)
  - Line 67 (default for new field)
  - Lines 97-101 (`openrouter()` method)
  - Line 295 (`has_chat_provider()`)
  - Lines 422-425 (build: openrouter registration block)

**Step 1: Add `openrouter_enabled` field**

In `RatatoskrBuilder` struct (after line 34), add:

```rust
    openrouter_enabled: bool,
```

In `new()` (after line 67), add:

```rust
            openrouter_enabled: false,
```

**Step 2: Update `openrouter()` method**

Change from:

```rust
    /// Configure OpenRouter provider (routes to many models).
    pub fn openrouter(mut self, api_key: impl Into<String>) -> Self {
        self.openrouter_key = Some(api_key.into());
        self
    }
```

to:

```rust
    /// Configure OpenRouter provider (routes to many models).
    ///
    /// Pass `Some(key)` for authenticated access or `None` for keyless
    /// free-tier access.
    pub fn openrouter(mut self, api_key: Option<impl Into<String>>) -> Self {
        self.openrouter_key = api_key.map(|k| k.into());
        self.openrouter_enabled = true;
        self
    }
```

**Step 3: Update `has_chat_provider()`**

Change `self.openrouter_key.is_some()` to `self.openrouter_enabled`:

```rust
    fn has_chat_provider(&self) -> bool {
        self.openrouter_enabled
            || self.anthropic_key.is_some()
            || self.openai_key.is_some()
            || self.google_key.is_some()
            || self.ollama_url.is_some()
    }
```

**Step 4: Update build() openrouter block**

Change from:

```rust
        // OpenRouter (routes to many models, good default)
        if let Some(ref key) = self.openrouter_key {
            let provider = make_provider(LLMBackend::OpenRouter, key.clone(), "openrouter");
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }
```

to:

```rust
        // OpenRouter (routes to many models, good default)
        if self.openrouter_enabled {
            let provider = Arc::new(
                LlmChatProvider::with_http_client(
                    LLMBackend::OpenRouter,
                    self.openrouter_key.clone(),
                    "openrouter",
                    http_client.clone(),
                )
                .timeout_secs(timeout_secs),
            );
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }
```

Note: we can't use the `make_provider` closure here because it expects `String`, and we now pass `Option<String>`. The other providers still use `make_provider` unchanged.

**Step 5: Update all callers that pass a bare string**

Every existing `.openrouter("key")` call becomes `.openrouter(Some("key"))`. Search all files in `src/` and `tests/`:

- `src/lib.rs:15` (doc comment) — `.openrouter(Some("sk-or-your-key"))`
- `src/gateway/builder.rs:220` (doc comment) — `.openrouter(Some("key"))`
- `src/gateway/builder.rs:270` (doc comment) — `.openrouter(Some("key"))`
- `src/providers/routing.rs:16` (doc comment) — `.openrouter(Some(key))`
- `src/bin/rat_registry.rs:100` — `.openrouter(Some(key))`
- `src/bin/ratd.rs:100` — `.openrouter(Some(key))`
- `tests/gateway_test.rs` — all `.openrouter("...")` calls become `.openrouter(Some("..."))`
- `tests/discovery_test.rs` — same pattern
- `tests/response_cache_test.rs` — same pattern

Use a project-wide search for `.openrouter(` and update all occurrences.

**Step 6: Verify it compiles**

Run: `cargo check`
Expected: compiles cleanly

**Step 7: Commit**

```bash
git add src/ tests/
git commit -m "feat: keyless openrouter support in builder (#24)

.openrouter(None) registers keyless OpenRouter for free-tier models.
.openrouter(Some(key)) preserves existing authenticated behaviour.

LlmChatProvider.api_key is now Option<String>; auth headers and llm
builder .api_key() calls are conditional on key presence."
```

---

### Task 7: ratatoskr — update ratd config wiring for keyless openrouter

**Files:**
- Modify: `src/bin/ratd.rs:97-102`

**Step 1: Change the openrouter config block**

From:

```rust
    // Configure API providers — only register when section is present AND key is available
    if config.providers.openrouter.is_some() {
        if let Some(key) = secrets.api_key("openrouter") {
            builder = builder.openrouter(key);
        }
    }
```

to:

```rust
    // OpenRouter: register when section is present; key is optional (keyless = free tier)
    if config.providers.openrouter.is_some() {
        builder = builder.openrouter(secrets.api_key("openrouter"));
    }
```

`secrets.api_key()` already returns `Option<String>`, which maps directly to the new signature.

**Step 2: Verify it compiles**

Run: `cargo check --features server`

**Step 3: Commit**

```bash
git add src/bin/ratd.rs
git commit -m "feat: ratd supports keyless openrouter via config section without api_key"
```

---

### Task 8: ratatoskr — tests for keyless builder

**Files:**
- Modify: `tests/gateway_test.rs`

**Step 1: Write the failing tests**

Add to `tests/gateway_test.rs`:

```rust
#[test]
fn test_builder_keyless_openrouter() {
    // keyless openrouter should build successfully
    let gateway = Ratatoskr::builder().openrouter(None::<String>).build();
    assert!(gateway.is_ok(), "keyless openrouter should build");
}

#[test]
fn test_builder_keyless_openrouter_has_chat() {
    let gateway = Ratatoskr::builder()
        .openrouter(None::<String>)
        .build()
        .unwrap();
    let caps = gateway.capabilities();
    assert!(caps.chat, "keyless openrouter should provide chat capability");
}
```

**Step 2: Run tests**

Run: `cargo nextest run --all-targets --locked -E 'test(keyless)'`
Expected: both pass

**Step 3: Commit**

```bash
git add tests/gateway_test.rs
git commit -m "test: keyless openrouter builder tests (#24)"
```

---

### Task 9: ratatoskr — wiremock test: keyless chat sends no auth header

**Files:**
- Modify: `tests/fetch_metadata_test.rs` (add keyless metadata fetch test alongside existing tests)

**Step 1: Write the test**

Add to `tests/fetch_metadata_test.rs`:

```rust
/// Build a keyless registry with a single OpenRouter provider pointed at wiremock.
fn keyless_registry_with_mock(mock_url: &str) -> ProviderRegistry {
    let provider = Arc::new(
        LlmChatProvider::new(LLMBackend::OpenRouter, None::<String>, "openrouter")
            .models_base_url(mock_url),
    );
    let mut registry = ProviderRegistry::new();
    registry.add_chat(provider);
    registry
}

#[tokio::test]
async fn fetch_metadata_keyless_sends_no_auth_header() {
    let server = MockServer::start().await;

    // Match GET /api/v1/models WITHOUT an Authorization header
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(wiremock::matchers::header_exists("Authorization").not())
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .expect(1)
        .mount(&server)
        .await;

    let registry = keyless_registry_with_mock(&server.uri());
    let metadata = registry
        .fetch_chat_metadata("test-vendor/test-model")
        .await
        .expect("keyless fetch should succeed");

    assert_eq!(metadata.info.id, "test-vendor/test-model");
}

#[tokio::test]
async fn fetch_metadata_keyed_sends_auth_header() {
    let server = MockServer::start().await;

    // Match GET /api/v1/models WITH the correct Authorization header
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(wiremock::matchers::header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_models_json()))
        .expect(1)
        .mount(&server)
        .await;

    let registry = registry_with_mock(&server.uri());
    let metadata = registry
        .fetch_chat_metadata("test-vendor/test-model")
        .await
        .expect("keyed fetch should succeed");

    assert_eq!(metadata.info.id, "test-vendor/test-model");
}
```

Note: check whether wiremock's `header_exists().not()` is available. If not, use a custom matcher or verify by checking that a mock *without* auth matching still gets hit. Adjust accordingly.

**Step 2: Run tests**

Run: `cargo nextest run --all-targets --locked -E 'test(fetch_metadata_keyless) | test(fetch_metadata_keyed_sends)'`
Expected: both pass

**Step 3: Commit**

```bash
git add tests/fetch_metadata_test.rs
git commit -m "test: wiremock tests for keyless vs keyed metadata fetch (#24)"
```

---

### Task 10: final verification

**Step 1: Run full test suite**

Run: `just pre-push`
Expected: format clean, clippy clean, all tests pass

**Step 2: Run llm crate tests**

Run: `cd /home/fey/projects/llm/llm-emesal && cargo test --lib`
Expected: all tests pass

**Step 3: Update AGENTS.md if needed**

The doc comment for the builder pattern in `AGENTS.md` shows `.openrouter(api_key)` — update to show both forms:

```rust
    .openrouter(Some(api_key))         // authenticated
    // .openrouter(None::<String>)     // keyless free-tier
```
