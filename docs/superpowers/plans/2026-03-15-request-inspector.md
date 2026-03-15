# Request Inspector Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional builder-level callback that exposes wire-format request JSON before it's sent to LLM providers, with zero cost when not set.

**Architecture:** Type alias `RequestInspector = Arc<dyn Fn(&str) + Send + Sync>` threaded from `RatatoskrBuilder` → `LlmChatProvider` → llm `LLMBuilder` → each backend's provider struct. The llm crate's existing `log::log_enabled!(Trace)` serialization gates are extended to also fire the callback when present.

**Tech Stack:** Rust, llm crate (fork at `~/forks/llm`), ratatoskr, wiremock (tests)

**Spec:** `docs/superpowers/specs/2026-03-15-request-inspector-design.md`

**Closes:** #27

---

## Chunk 1: llm crate — type alias, builder state, builder method

### Task 1: Add `RequestInspector` type alias to the llm crate

**Files:**
- Modify: `~/forks/llm/llm-main/src/lib.rs`

- [ ] **Step 1: Add the type alias**

Add after line 29 (`use serde::{Deserialize, Serialize};`):

```rust
use std::sync::Arc;

/// Optional callback that receives serialized request JSON before it's sent
/// to the provider. Zero cost when `None`. Set via `LLMBuilder::request_inspector()`.
pub type RequestInspector = Arc<dyn Fn(&str) + Send + Sync>;
```

- [ ] **Step 2: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm`
Expected: compiles with no errors

- [ ] **Step 3: Commit**

```bash
cd ~/forks/llm && git add llm-main/src/lib.rs && git commit -m "feat: add RequestInspector type alias"
```

### Task 2: Add `request_inspector` to `BuilderState`

**Files:**
- Modify: `~/forks/llm/llm-main/src/builder/state.rs`

- [ ] **Step 1: Add the field**

Add to the `BuilderState` struct (after the last field, line 57):

```rust
pub(crate) request_inspector: Option<crate::RequestInspector>,
```

- [ ] **Step 2: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm`
Expected: compiles (Default derive still works since `Option<T>` is `Default`)

- [ ] **Step 3: Commit**

```bash
cd ~/forks/llm && git add llm-main/src/builder/state.rs && git commit -m "feat: add request_inspector field to BuilderState"
```

### Task 3: Add `.request_inspector()` to `LLMBuilder`

**Files:**
- Modify: `~/forks/llm/llm-main/src/builder/llm_builder.rs`

- [ ] **Step 1: Add the builder method**

Add after the `top_k` method (after line 113):

```rust
    /// Sets an optional callback that receives the serialized request JSON
    /// before it's sent to the provider. Useful for debugging wire format.
    pub fn request_inspector(mut self, inspector: crate::RequestInspector) -> Self {
        self.state.request_inspector = Some(inspector);
        self
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
cd ~/forks/llm && git add llm-main/src/builder/llm_builder.rs && git commit -m "feat: add request_inspector() method to LLMBuilder"
```

---

## Chunk 2: llm crate — thread inspector into all backends

The inspector needs to be threaded from `BuilderState` through each `build_*` function into the backend struct. Each backend struct gets an `Option<RequestInspector>` field, and each `new()` / `with_client()` constructor gains the parameter.

**Pattern:** Every backend follows the same mechanical transformation:

1. Add `request_inspector: Option<crate::RequestInspector>` field to the backend struct
2. Add the parameter to `new()` and `with_client()` (if applicable)
3. Wire it in the builder's `build_*` function: `state.request_inspector.take()` (first backend) or `state.request_inspector.clone()` (subsequent backends — but in practice only one backend is used per builder, so `.take()` is fine everywhere)

**Important:** Each `build_*` function is called from `backends::build_backend()` via a `match`. Only one arm fires per `build()` call, so `state.request_inspector.take()` is safe in every arm.

### Task 4: Thread inspector into `OpenAICompatibleProviderConfig`

This covers OpenRouter, Groq, Mistral, Cohere, HuggingFace (the OpenAI-compatible backends), **and** `OpenAI` (which wraps `OpenAICompatibleProvider<OpenAIConfig>` internally).

**Files:**
- Modify: `~/forks/llm/llm-main/src/providers/openai_compatible.rs`
- Modify: `~/forks/llm/llm-main/src/backends/openai.rs` (wraps `OpenAICompatibleProvider`)
- Modify: builder backend files for `build_openrouter`, `build_groq`, `build_mistral`, `build_cohere`, `build_huggingface`, `build_openai`

- [ ] **Step 1: Add field to `OpenAICompatibleProviderConfig`**

Add after the `extra_body` field (line 61 in `openai_compatible.rs`):

```rust
    /// Optional callback receiving serialized request JSON before send.
    pub request_inspector: Option<crate::RequestInspector>,
```

- [ ] **Step 2: Update `OpenAICompatibleProvider::new()` and `with_client()` constructors**

Both constructors in `openai_compatible.rs` build `OpenAICompatibleProviderConfig` directly. Add `request_inspector: Option<crate::RequestInspector>` as a parameter to both, and wire it into the config struct literal. Search for all `OpenAICompatibleProviderConfig { ... }` construction sites within this file.

- [ ] **Step 3: Update all builder `build_*` functions**

Each `build_*` function that creates an `OpenAICompatibleProvider` needs to pass `state.request_inspector.take()`. This includes:
- `build_openrouter`, `build_groq`, `build_mistral`, `build_cohere`, `build_huggingface` (in the builder backends module)
- `build_openai` — this calls `OpenAI::new()` which calls `OpenAICompatibleProvider::new()`, so `state.request_inspector.take()` must flow through `OpenAI::new()` → inner provider constructor

- [ ] **Step 4: Update `OpenAI::new()` and `OpenAI::with_client()` in `backends/openai.rs`**

`OpenAI` wraps `OpenAICompatibleProvider<OpenAIConfig>` as `self.provider`. Add `request_inspector: Option<crate::RequestInspector>` parameter to `OpenAI::new()` and `OpenAI::with_client()`, and pass it through to `OpenAICompatibleProvider::new()`.

- [ ] **Step 5: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm --all-features`
Expected: compiles

- [ ] **Step 6: Commit**

```bash
cd ~/forks/llm && git add -A && git commit -m "feat: thread request_inspector into OpenAI-compatible providers"
```

### Task 5: Thread inspector into standalone backends

Each truly standalone backend (ones that do NOT wrap `OpenAICompatibleProvider`) needs the same treatment. This is mechanical — same pattern for each.

**Note:** `OpenAI` (Responses API) is handled in Task 4 since it wraps `OpenAICompatibleProvider<OpenAIConfig>`. `ElevenLabs` is a TTS-only backend with no chat serialization — skip it. `AwsBedrock` shares the `azure_openai.rs` backend struct (constructed via `azure::build_bedrock`), so adding the field to `AzureOpenAI` covers both paths.

**Files (all under `~/forks/llm/llm-main/src/`):**
- `backends/anthropic.rs` — struct at line ~70, `new()` at line ~522, `with_client()` nearby
- `backends/google.rs` — struct at line ~100, `new()` at line ~499
- `backends/ollama.rs` — struct at line ~63, `new()` at line ~327
- `backends/deepseek.rs` — struct at line ~47, `new()` at line ~107
- `backends/phind.rs` — struct at line ~51, `new()` at line ~84
- `backends/xai.rs` — struct at line ~252 (approx), `new()` at same line
- `backends/azure_openai.rs` — struct at line ~72 (approx), `new()` at line ~364

And corresponding builder functions in `~/forks/llm/llm-main/src/builder/build/backends/`:
- `anthropic.rs`, `google.rs`, `ollama.rs`, `deepseek.rs`, `phind.rs`, `xai.rs`, `azure.rs` (covers both AzureOpenAI and AwsBedrock)

- [ ] **Step 1: For each backend struct, add the field**

```rust
    /// Optional callback receiving serialized request JSON before send.
    request_inspector: Option<crate::RequestInspector>,
```

- [ ] **Step 2: For each `new()` and `with_client()`, add the parameter and wire it**

Add `request_inspector: Option<crate::RequestInspector>` as the last parameter. Store it in `Self { ..., request_inspector }`.

- [ ] **Step 3: For each `build_*` function, pass `state.request_inspector.take()`**

Example for `build_anthropic`:

```rust
let provider = crate::backends::anthropic::Anthropic::new(
    api_key,
    state.model.take(),
    // ... existing params ...
    state.request_inspector.take(),
);
```

For `azure.rs`, ensure BOTH `build_azure_openai` and `build_bedrock` pass the inspector.

- [ ] **Step 4: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm --all-features`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
cd ~/forks/llm && git add -A && git commit -m "feat: thread request_inspector into all standalone backends"
```

---

## Chunk 3: llm crate — invoke the callback at serialization points

### Task 6: Add callback invocation at all trace logging serialization points

Every existing `log::log_enabled!(log::Level::Trace)` block that serializes the request body needs to be extended to also call the inspector.

**Transformation pattern** — from:

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
        if let Some(ref cb) = self.request_inspector {
            (cb)(&json);
        }
        log::trace!("...", json);
    }
}
```

For `OpenAICompatibleProvider<T>`, use `self.config.request_inspector` instead of `self.request_inspector` (since the field lives on the shared `Arc<OpenAICompatibleProviderConfig>`).

**Files and locations (all under `~/forks/llm/llm-main/src/`):**

| File | Function | Approx line | Self access |
|------|----------|-------------|-------------|
| `providers/openai_compatible.rs` | `chat_with_tools` | 635 | `self.config.request_inspector` |
| `providers/openai_compatible.rs` | `chat_stream_struct` | 756 | `self.config.request_inspector` |
| `providers/openai_compatible.rs` | `chat_stream_with_tools` | 861 | `self.config.request_inspector` |
| `backends/anthropic.rs` | `chat_with_tools` | 732 | `self.request_inspector` |
| `backends/anthropic.rs` | `chat_stream_with_tools` | 932 | `self.request_inspector` |
| `backends/google.rs` | `chat` | 736 | `self.request_inspector` |
| `backends/google.rs` | `chat_with_tools` | 910 | `self.request_inspector` |
| `backends/ollama.rs` | `chat_with_tools` | 504 | `self.request_inspector` |
| `backends/openai.rs` | `log_request_payload` (helper) | 594 | `self.provider.config.request_inspector` (see note below) |
| `backends/deepseek.rs` | `chat` | 227 | `self.request_inspector` |
| `backends/phind.rs` | `chat` | 293 | `self.request_inspector` |
| `backends/xai.rs` | `chat` | 468 | `self.request_inspector` |
| `backends/azure_openai.rs` | `chat_with_tools` | 617 | `self.request_inspector` |

**Note for `backends/openai.rs`:** The `log_request_payload` method on `OpenAI` uses an inverted early-return guard pattern, not the wrapping `if` pattern used everywhere else. The transformation is different:

From:
```rust
fn log_request_payload<T: Serialize>(&self, label: &str, body: &T) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Ok(json) = serde_json::to_string(body) {
        log::trace!("{label}: {json}");
    }
}
```

To:
```rust
fn log_request_payload<T: Serialize>(&self, label: &str, body: &T) {
    if self.provider.config.request_inspector.is_none() && !log::log_enabled!(log::Level::Trace) {
        return;
    }
    if let Ok(json) = serde_json::to_string(body) {
        if let Some(ref cb) = self.provider.config.request_inspector {
            (cb)(&json);
        }
        log::trace!("{label}: {json}");
    }
}
```

The inspector is accessed via `self.provider.config.request_inspector` because `OpenAI` wraps `OpenAICompatibleProvider<OpenAIConfig>` as `self.provider`, which stores config in `Arc<OpenAICompatibleProviderConfig>`.

- [ ] **Step 1: Apply the transformation to all serialization points in `openai_compatible.rs`** (3 points)

- [ ] **Step 2: Apply the transformation to all standalone backends** (~10 points)

- [ ] **Step 3: Verify it compiles**

Run: `cd ~/forks/llm && cargo check -p llm --all-features`
Expected: compiles

- [ ] **Step 4: Run existing llm tests**

Run: `cd ~/forks/llm && cargo test -p llm`
Expected: all existing tests pass (no behavioural change when inspector is `None`)

- [ ] **Step 5: Commit**

```bash
cd ~/forks/llm && git add -A && git commit -m "feat: invoke request_inspector at all serialization points"
```

---

## Chunk 4: ratatoskr — type alias, builder, plumbing

### Task 7: Add `RequestInspector` type alias to ratatoskr

**Files:**
- Modify: `src/types/mod.rs` — add the type alias
- Modify: `src/lib.rs` — re-export it

- [ ] **Step 1: Add the type alias to `src/types/mod.rs`**

Add at the end of the file:

```rust
/// Optional callback that receives serialized request JSON before it's sent
/// to the LLM provider. Set via [`RatatoskrBuilder::request_inspector()`].
///
/// Zero overhead when not configured — no serialization occurs.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use ratatoskr::RequestInspector;
///
/// let inspector: RequestInspector = Arc::new(|body: &str| {
///     eprintln!("wire: {body}");
/// });
/// ```
pub type RequestInspector = std::sync::Arc<dyn Fn(&str) + Send + Sync>;
```

- [ ] **Step 2: Re-export from `src/lib.rs`**

Add `RequestInspector` to the types re-export block (line 132):

```rust
pub use types::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason,
    GenerateEvent, GenerateOptions, GenerateResponse, Message, MessageContent, ModelCapability,
    ModelInfo, ModelMetadata, ModelStatus, NliLabel, NliResult, ParameterAvailability,
    ParameterName, ParameterRange, ParameterValidationPolicy, PricingInfo, ReasoningConfig,
    ReasoningEffort, RequestInspector, ResponseFormat, Role, StanceLabel, StanceResult, Token,
    ToolCall, ToolChoice, ToolDefinition, Usage,
};
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add src/types/mod.rs src/lib.rs && git commit -m "feat: add RequestInspector type alias"
```

### Task 8: Add `request_inspector` to `RatatoskrBuilder`

**Files:**
- Modify: `src/gateway/builder.rs`

- [ ] **Step 1: Add field to `RatatoskrBuilder` struct**

Add after the `registry_refresh_disabled` field (line 50):

```rust
    request_inspector: Option<crate::RequestInspector>,
```

- [ ] **Step 2: Initialize in `new()`**

Add to the `Self { ... }` block (after `registry_refresh_disabled: false,` around line 86):

```rust
            request_inspector: None,
```

- [ ] **Step 3: Add builder method**

Add after the existing builder methods (e.g. after `disable_registry_refresh`):

```rust
    /// Sets a callback that receives the serialized request JSON before it's
    /// sent to the LLM provider. Useful for debugging wire format.
    ///
    /// Zero overhead when not set — no serialization occurs.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use ratatoskr::Ratatoskr;
    ///
    /// let gateway = Ratatoskr::builder()
    ///     .openrouter(Some("sk-or-key"))
    ///     .request_inspector(Arc::new(|body| eprintln!("wire: {body}")))
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn request_inspector(mut self, inspector: crate::RequestInspector) -> Self {
        self.request_inspector = Some(inspector);
        self
    }
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add src/gateway/builder.rs && git commit -m "feat: add request_inspector to RatatoskrBuilder"
```

### Task 9: Add `request_inspector` to `LlmChatProvider` and thread into llm builder

**Files:**
- Modify: `src/providers/llm_chat.rs`

- [ ] **Step 1: Add field to `LlmChatProvider` struct**

Add after `models_base_url` field (line 55):

```rust
    /// Optional callback receiving serialized request JSON before send.
    request_inspector: Option<crate::RequestInspector>,
```

- [ ] **Step 2: Initialize in `with_http_client()`**

Add to the `Self { ... }` block (line 110-119):

```rust
            request_inspector: None,
```

- [ ] **Step 3: Add setter method**

Add after the existing setter methods (e.g. after `models_base_url`):

```rust
    /// Set the request inspector callback.
    pub fn request_inspector(mut self, inspector: crate::RequestInspector) -> Self {
        self.request_inspector = Some(inspector);
        self
    }
```

- [ ] **Step 4: Thread into `build_provider()`**

In `build_provider()`, after the builder is fully configured (around line 291, before `builder.build()`), add:

```rust
        // Thread request inspector to the llm crate
        if let Some(ref inspector) = self.request_inspector {
            builder = builder.request_inspector(inspector.clone());
        }
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 6: Commit**

```bash
git add src/providers/llm_chat.rs && git commit -m "feat: thread request_inspector through LlmChatProvider to llm builder"
```

### Task 10: Thread inspector from `RatatoskrBuilder.build()` to providers

**Files:**
- Modify: `src/gateway/builder.rs`

- [ ] **Step 1: Update the `make_provider` closure and all provider construction sites**

In `build()`, update the `make_provider` closure (around line 452) to accept and set the inspector:

```rust
        let inspector = self.request_inspector.clone();
        let make_provider = |backend, key: String, name: &str| -> Arc<LlmChatProvider> {
            let mut provider =
                LlmChatProvider::with_http_client(backend, Some(key), name, http_client.clone())
                    .timeout_secs(timeout_secs);
            if let Some(ref cb) = inspector {
                provider = provider.request_inspector(cb.clone());
            }
            Arc::new(provider)
        };
```

Also update the OpenRouter provider block (line 461-469) which doesn't use `make_provider`:

```rust
        if self.openrouter_enabled {
            let mut provider = LlmChatProvider::with_http_client(
                LLMBackend::OpenRouter,
                self.openrouter_key.clone(),
                "openrouter",
                http_client.clone(),
            )
            .timeout_secs(timeout_secs);
            if let Some(ref cb) = inspector {
                provider = provider.request_inspector(cb.clone());
            }
            let provider = Arc::new(provider);
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }
```

And the Ollama block (line 497-508):

```rust
        if let Some(ref url) = self.ollama_url {
            let mut provider = LlmChatProvider::with_http_client(
                LLMBackend::Ollama,
                Some("ollama"),
                "ollama",
                http_client.clone(),
            )
            .timeout_secs(timeout_secs)
            .ollama_url(url.clone());
            if let Some(ref cb) = inspector {
                provider = provider.request_inspector(cb.clone());
            }
            let provider = Arc::new(provider);
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }
```

And the Stub block (line 514-526):

```rust
        if let Some(ref url) = self.stub_url {
            let mut provider = LlmChatProvider::with_http_client(
                LLMBackend::OpenRouter,
                Some("stub-key"),
                "stub",
                http_client.clone(),
            )
            .timeout_secs(timeout_secs)
            .base_url(url.clone());
            if let Some(ref cb) = inspector {
                provider = provider.request_inspector(cb.clone());
            }
            let provider = Arc::new(provider);
            registry.add_chat(provider.clone());
            registry.add_generate(provider);
        }
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add src/gateway/builder.rs && git commit -m "feat: thread request_inspector from builder to all LlmChatProvider instances"
```

---

## Chunk 5: Tests

### Task 11: Write ratatoskr unit test — builder accepts inspector

**Files:**
- Modify: `tests/gateway_test.rs`

- [ ] **Step 1: Write the test**

Add at the end of `tests/gateway_test.rs`:

```rust
#[test]
fn test_builder_with_request_inspector() {
    use std::sync::Arc;

    let inspector = Arc::new(|_body: &str| {});
    let gateway = Ratatoskr::builder()
        .openrouter(Some("test-key"))
        .request_inspector(inspector)
        .build();

    assert!(gateway.is_ok());
}
```

- [ ] **Step 2: Run it**

Run: `cargo test --test gateway_test test_builder_with_request_inspector`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/gateway_test.rs && git commit -m "test: builder accepts request_inspector"
```

### Task 12: Write ratatoskr integration test — inspector fires on chat

**Files:**
- Create: `tests/request_inspector_test.rs`

- [ ] **Step 1: Write the test**

```rust
//! Integration test: request inspector callback fires with serialized JSON.

use std::sync::{Arc, Mutex};

use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal OpenAI-compatible chat response for the stub backend.
fn chat_response_json() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    })
}

#[tokio::test]
async fn inspector_fires_on_chat() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_json()))
        .mount(&server)
        .await;

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    let inspector = Arc::new(move |body: &str| {
        captured_clone.lock().unwrap().push(body.to_string());
    });

    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .request_inspector(inspector)
        .build()
        .unwrap();

    let _response = gateway
        .chat(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await
        .unwrap();

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "inspector should fire exactly once");

    // Verify the captured JSON is valid and contains expected fields
    let parsed: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(parsed["model"], "test-model");
    assert!(parsed["messages"].is_array());
}

#[tokio::test]
async fn inspector_not_set_no_overhead() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_json()))
        .mount(&server)
        .await;

    // No inspector set — should work fine
    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .build()
        .unwrap();

    let response = gateway
        .chat(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await;

    assert!(response.is_ok());
}

#[tokio::test]
async fn inspector_fires_on_chat_stream() {
    use futures_util::StreamExt;

    let server = MockServer::start().await;

    // SSE streaming response
    let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    let inspector = Arc::new(move |body: &str| {
        captured_clone.lock().unwrap().push(body.to_string());
    });

    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .request_inspector(inspector)
        .build()
        .unwrap();

    let mut stream = gateway
        .chat_stream(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await
        .unwrap();

    // Drain the stream
    while let Some(_event) = stream.next().await {}

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "inspector should fire once for stream");
    let parsed: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert!(parsed["stream"].as_bool().unwrap_or(false), "stream should be true");
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test --test request_inspector_test`
Expected: all 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/request_inspector_test.rs && git commit -m "test: request inspector integration tests (chat, stream, not-set)"
```

---

## Chunk 6: Lint, docs, final verification

### Task 13: Update AGENTS.md and run pre-push checks

**Files:**
- Modify: `AGENTS.md` — add `RequestInspector` to the Key Types section

- [ ] **Step 1: Add to Key Types in AGENTS.md**

Add a bullet in the Key Types section:

```markdown
- `RequestInspector` — `Arc<dyn Fn(&str) + Send + Sync>`; optional builder-level callback receiving serialized request JSON before send. Set via `RatatoskrBuilder::request_inspector()`.
```

- [ ] **Step 2: Run lint**

Run: `just lint`
Expected: no errors (fix any clippy/fmt issues)

- [ ] **Step 3: Run full test suite**

Run: `just test`
Expected: all tests pass

- [ ] **Step 4: Build docs**

Run: `cargo doc --no-deps`
Expected: docs build with no warnings

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md && git commit -m "docs: add RequestInspector to AGENTS.md key types"
```

- [ ] **Step 6: Also lint and test in the llm fork**

Run: `cd ~/forks/llm && cargo fmt --all && cargo clippy -p llm --all-features -- -D warnings && cargo test -p llm`
Expected: all pass

- [ ] **Step 7: Remind user about `just pre-push` and `just merge-to-dev`**

---

## Notes for implementer

- The llm fork is at `~/forks/llm`. Ratatoskr depends on it via a path dependency. Changes to both repos are needed.
- `cargo target` is at `~/.cache/cargo-target` (set in the project's cargo config).
- When modifying backend `new()` signatures, the `request_inspector` parameter should always be the **last** parameter to minimize churn in existing call sites.
- The `OpenAICompatibleProvider<T>` accesses the inspector via `self.config.request_inspector` (through the `Arc<OpenAICompatibleProviderConfig>`), while standalone backends use `self.request_inspector` directly.
- In the `build_backend()` match dispatch, only one arm fires per call, so `state.request_inspector.take()` is safe in every arm.
- Clippy will flag `3.14` in unit tests — this is a known project quirk, ignore it.
- After all tasks: collect any notes needing to be added to AGENTS.md.
