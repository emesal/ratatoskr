# Preset URI Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Model strings with the `ratatoskr:` prefix are transparently resolved to concrete model IDs via the preset table, enabling downstream tools to use preset references without special handling.

**Architecture:** Resolution lives in `EmbeddedGateway`, which already has access to the `ModelRegistry` (presets) and `ProviderRegistry` (API dispatch). A private `resolve_model` method intercepts preset URIs before delegation. Format: `ratatoskr:<tier>/<capability>` — both parts required.

**Tech Stack:** Rust, thiserror for error variant, integration tests in `tests/`.

---

## Progress

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| 1 | done | 012b21c | error variant |
| 2 | done | 7aa92d6 | resolve_model + unit tests |
| 3 | done | 51a1bab | wire into chat/chat_stream |
| 4 | done | 574d3f9 | wire into generate/generate_stream |
| 5 | done | e139d60 | wire into model_metadata/fetch_model_metadata |
| 6 | done | f249e96 | integration tests |
| 7 | done | — | design doc cleanup, commit |

---

### Task 1: Add `PresetNotFound` error variant

**Files:**
- Modify: `src/error.rs:4-87` (add variant to `RatatoskrError`)

**Step 1: Add the error variant**

In `src/error.rs`, add after the `ModelNotFound` variant (~line 28):

```rust
    #[error("preset not found: tier '{tier}', capability '{capability}'")]
    PresetNotFound { tier: String, capability: String },
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: PASS

**Step 3: Commit**

```bash
git add src/error.rs
git commit -m "feat: add PresetNotFound error variant (#23)"
```

---

### Task 2: Add `resolve_model` to `EmbeddedGateway`

**Files:**
- Modify: `src/gateway/embedded.rs` (add private method + unit tests)

**Step 1: Write failing tests**

Add a `#[cfg(test)]` module at the bottom of `src/gateway/embedded.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal gateway with the embedded seed registry (has presets).
    fn test_gateway() -> EmbeddedGateway {
        // Need at least one chat provider; use openrouter with a fake key
        Ratatoskr::builder()
            .openrouter("fake-key")
            .build()
            .unwrap()
    }

    #[test]
    fn test_resolve_preset_uri() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:free/text-generation").unwrap();
        assert_eq!(resolved, "google/gemini-2.0-flash-001");
    }

    #[test]
    fn test_resolve_preset_uri_agentic() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:free/agentic").unwrap();
        assert_eq!(resolved, "google/gemini-2.0-flash-001");
    }

    #[test]
    fn test_resolve_preset_uri_premium() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("ratatoskr:premium/agentic").unwrap();
        assert_eq!(resolved, "anthropic/claude-sonnet-4");
    }

    #[test]
    fn test_resolve_plain_model_passthrough() {
        let gw = test_gateway();
        let resolved = gw.resolve_model("anthropic/claude-sonnet-4").unwrap();
        assert_eq!(resolved, "anthropic/claude-sonnet-4");
    }

    #[test]
    fn test_resolve_preset_unknown_tier() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:nonexistent/agentic").unwrap_err();
        assert!(matches!(err, RatatoskrError::PresetNotFound { .. }));
    }

    #[test]
    fn test_resolve_preset_unknown_capability() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:free/nonexistent").unwrap_err();
        assert!(matches!(err, RatatoskrError::PresetNotFound { .. }));
    }

    #[test]
    fn test_resolve_preset_missing_capability() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:free").unwrap_err();
        assert!(matches!(err, RatatoskrError::InvalidInput(_)));
    }

    #[test]
    fn test_resolve_preset_empty_after_prefix() {
        let gw = test_gateway();
        let err = gw.resolve_model("ratatoskr:").unwrap_err();
        assert!(matches!(err, RatatoskrError::InvalidInput(_)));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p ratatoskr embedded::tests -- --nocapture`
Expected: compilation error (`resolve_model` doesn't exist)

**Step 3: Implement `resolve_model`**

Add to `src/gateway/embedded.rs`, inside the `impl EmbeddedGateway` block (not the trait impl):

```rust
    /// Prefix for preset model URIs.
    const PRESET_PREFIX: &str = "ratatoskr:";

    /// Resolve a model string, expanding `ratatoskr:<tier>/<capability>` preset
    /// URIs to concrete model IDs. Non-preset strings pass through unchanged.
    fn resolve_model(&self, model: &str) -> Result<String> {
        let Some(rest) = model.strip_prefix(Self::PRESET_PREFIX) else {
            return Ok(model.to_string());
        };

        let Some((tier, capability)) = rest.split_once('/') else {
            return Err(RatatoskrError::InvalidInput(format!(
                "preset URI must be `ratatoskr:<tier>/<capability>`, got `{model}`"
            )));
        };

        if tier.is_empty() || capability.is_empty() {
            return Err(RatatoskrError::InvalidInput(format!(
                "preset URI must be `ratatoskr:<tier>/<capability>`, got `{model}`"
            )));
        }

        self.model_registry
            .preset(tier, capability)
            .map(String::from)
            .ok_or_else(|| RatatoskrError::PresetNotFound {
                tier: tier.to_string(),
                capability: capability.to_string(),
            })
    }
```

Note: `impl EmbeddedGateway` already exists with `fn new(...)`. If there's no standalone `impl EmbeddedGateway` block (only `impl ModelGateway for EmbeddedGateway`), create one above the trait impl block.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p ratatoskr embedded::tests -- --nocapture`
Expected: all pass

**Step 5: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: resolve_model for ratatoskr: preset URIs (#23)

Parses ratatoskr:<tier>/<capability>, looks up preset in model
registry, returns concrete model ID. Non-preset strings pass through."
```

---

### Task 3: Wire resolution into `chat` and `chat_stream`

**Files:**
- Modify: `src/gateway/embedded.rs` (update `chat` and `chat_stream` in trait impl)

**Step 1: Update `chat`**

Replace the current `chat` implementation:

```rust
    #[instrument(name = "gateway.chat", skip(self, messages, tools, options), fields(model = %options.model))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let model = self.resolve_model(&options.model)?;
        if model == options.model {
            self.registry.chat(messages, tools, options).await
        } else {
            let resolved = ChatOptions { model, ..options.clone() };
            self.registry.chat(messages, tools, &resolved).await
        }
    }
```

**Step 2: Update `chat_stream`**

Same pattern:

```rust
    #[instrument(name = "gateway.chat_stream", skip(self, messages, tools, options), fields(model = %options.model))]
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let model = self.resolve_model(&options.model)?;
        if model == options.model {
            self.registry.chat_stream(messages, tools, options).await
        } else {
            let resolved = ChatOptions { model, ..options.clone() };
            self.registry.chat_stream(messages, tools, &resolved).await
        }
    }
```

**Step 3: Verify it compiles**

Run: `cargo build`
Expected: PASS

**Step 4: Run all tests**

Run: `cargo test`
Expected: all pass

**Step 5: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: resolve preset URIs in chat/chat_stream (#23)"
```

---

### Task 4: Wire resolution into `generate` and `generate_stream`

**Files:**
- Modify: `src/gateway/embedded.rs` (update `generate` and `generate_stream`)

**Step 1: Update `generate`**

```rust
    #[instrument(name = "gateway.generate", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        let model = self.resolve_model(&options.model)?;
        if model == options.model {
            self.registry.generate(prompt, options).await
        } else {
            let resolved = GenerateOptions { model, ..options.clone() };
            self.registry.generate(prompt, &resolved).await
        }
    }
```

**Step 2: Update `generate_stream`**

```rust
    #[instrument(name = "gateway.generate_stream", skip(self, prompt, options), fields(model = %options.model))]
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        let model = self.resolve_model(&options.model)?;
        if model == options.model {
            self.registry.generate_stream(prompt, options).await
        } else {
            let resolved = GenerateOptions { model, ..options.clone() };
            self.registry.generate_stream(prompt, &resolved).await
        }
    }
```

**Step 3: Run all tests**

Run: `cargo test`
Expected: all pass

**Step 4: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: resolve preset URIs in generate/generate_stream (#23)"
```

---

### Task 5: Wire resolution into `model_metadata` and `fetch_model_metadata`

**Files:**
- Modify: `src/gateway/embedded.rs` (update metadata methods)

**Step 1: Update `model_metadata`**

```rust
    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        let resolved = self.resolve_model(model).ok()?;
        // Registry (curated) takes priority over cache (ephemeral)
        self.model_registry
            .get(&resolved)
            .cloned()
            .or_else(|| self.model_cache.get(&resolved))
    }
```

**Step 2: Update `fetch_model_metadata`**

```rust
    #[instrument(name = "gateway.fetch_model_metadata", skip(self))]
    async fn fetch_model_metadata(&self, model: &str) -> Result<ModelMetadata> {
        let resolved = self.resolve_model(model)?;
        let metadata = self.registry.fetch_chat_metadata(&resolved).await?;
        self.model_cache.insert(metadata.clone());
        Ok(metadata)
    }
```

**Step 3: Update `resolve_preset`**

The existing `resolve_preset` method should also resolve through preset URIs for consistency:

```rust
    fn resolve_preset(&self, tier: &str, capability: &str) -> Option<String> {
        self.model_registry
            .preset(tier, capability)
            .map(String::from)
    }
```

(This one is unchanged — it already works correctly since it takes tier/capability separately.)

**Step 4: Run all tests**

Run: `cargo test`
Expected: all pass

**Step 5: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: resolve preset URIs in model_metadata/fetch_model_metadata (#23)"
```

---

### Task 6: Integration tests

**Files:**
- Modify: `tests/gateway_test.rs` (add preset resolution tests)

**Step 1: Add integration tests**

Append to `tests/gateway_test.rs`:

```rust
// ===== Preset URI resolution =====

#[test]
fn test_preset_uri_resolves_in_model_metadata() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    // preset URI should resolve to the same metadata as the concrete model
    let via_preset = gateway.model_metadata("ratatoskr:free/text-generation");
    let via_direct = gateway.model_metadata("google/gemini-2.0-flash-001");

    assert!(via_preset.is_some(), "preset should resolve to metadata");
    assert_eq!(
        via_preset.unwrap().info.id,
        via_direct.unwrap().info.id,
        "preset and direct should yield same model"
    );
}

#[test]
fn test_preset_uri_resolve_preset_still_works() {
    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();

    // the resolve_preset trait method should still work independently
    let model = gateway.resolve_preset("free", "agentic");
    assert!(model.is_some());
    assert_eq!(model.unwrap(), "google/gemini-2.0-flash-001");
}

#[test]
fn test_preset_uri_unknown_tier_returns_error() {
    use ratatoskr::RatatoskrError;

    let gateway = Ratatoskr::builder().openrouter("fake-key").build().unwrap();
    let result = gateway.model_metadata("ratatoskr:nonexistent/agentic");
    assert!(result.is_none(), "unknown preset tier should return None from model_metadata");
}
```

**Step 2: Run integration tests**

Run: `cargo test --test gateway_test -- --nocapture`
Expected: all pass

**Step 3: Commit**

```bash
git add tests/gateway_test.rs
git commit -m "test: integration tests for preset URI resolution (#23)"
```

---

### Task 7: Final cleanup

**Files:**
- Modify: `docs/plans/2026-02-13-preset-uri-resolution-design.md` (mark complete)

**Step 1: Run full test suite**

Run: `cargo test`
Expected: all pass

**Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: no warnings

**Step 3: Update design doc progress**

Mark the design doc as implemented (add a "Status: implemented" line at the top).

**Step 4: Commit**

```bash
git add docs/plans/2026-02-13-preset-uri-resolution-design.md
git commit -m "docs: mark preset URI resolution design as implemented (#23)"
```
