# v1.0.0 Code Review — Consolidated Findings

> **For Claude:** this is a self-contained review document. each task is independent unless noted. work through tasks in batch order (A → B → C → D → E). tick `[ ]` → `[x]` as you go. run `just pre-push` after each batch to verify nothing breaks.

**Date:** 2026-02-11
**Scope:** full codebase review across five parallel review agents covering: core types & traits, providers & gateway, gRPC server/client, registry & model layer, tests & project config.

**Overall assessment:** architecture is clean, patterns are consistent, docs are thorough. most critical items are API surface concerns that become permanent post-1.0.

---

## Batch A — Blockers (fix before release)

These are API surface issues that become permanent after 1.0, plus broken tooling.

### A1. Memory leak in `from_status` `[  ]`

**File:** `src/client/service_client.rs` (~line 62)
**Severity:** CRITICAL

`Box::leak` is used to convert `String` → `&'static str` for the `NotImplemented` variant. this leaks memory on every `Unimplemented` gRPC status. in a long-running client hitting a misconfigured server, this is unbounded growth.

**Fix:** change `RatatoskrError::NotImplemented` to accept `String` or `Cow<'static, str>` instead of `&'static str`. update all construction sites. the enum is `#[non_exhaustive]` after A7, so this is the last chance to change it cheaply.

**Grep for impact:** `NotImplemented` in `src/error.rs`, `src/client/service_client.rs`, `src/server/service.rs`, `src/traits.rs`

---

### A2. `rat health` ignores the Health RPC `[  ]`

**File:** `src/bin/rat.rs` (~line 125)
**Severity:** CRITICAL

the `health` subcommand calls `list_models()` and unconditionally prints "status: healthy". the proto defines a `Health` RPC returning version, git SHA, and healthy flag — none of which are used. `ServiceClient` doesn't expose the health RPC at all (it's not part of `ModelGateway`).

**Fix:**
1. add a `health()` method to `ServiceClient` (outside the trait impl) that calls the `Health` RPC
2. update `rat health` to call it and display version, git SHA, healthy status
3. keep the `list_models()` call as supplementary info if desired

---

### A3. `pub mod response` leaks internal module path `[  ]`

**File:** `src/types/mod.rs` (line 10)
**Severity:** CRITICAL

every other submodule in `types/` is `mod` (private) with re-exports. `response` is `pub mod`, which means consumers can reach types via both `ratatoskr::ChatEvent` and `ratatoskr::types::response::ChatEvent`. post-1.0, removing the `pub` is breaking.

**Fix:** change `pub mod response` → `mod response`. all types are already re-exported. verify no external code depends on the `types::response::` path (grep the workspace).

---

### A4. rename `future.rs` — misleading module name `[  ]`

**File:** `src/types/future.rs`
**Severity:** CRITICAL

module is named `future` but contains production NLP types (`Embedding`, `NliResult`, `NliLabel`, `ClassifyResult`). the doc says "stub types for future capabilities (Phase 2+)" — but Phase 2 is complete. the name collides conceptually with `std::future::Future`.

**Fix:** rename to `src/types/inference.rs` (or `nlp.rs`). update the module doc to accurately describe the contents. update `mod` declaration in `types/mod.rs`.

---

### A5. `ChatOptions::default()` foot-gun `[  ]`

**File:** `src/types/options.rs`
**Severity:** CRITICAL

`ChatOptions` derives `Default`, producing an empty `model: String("")`. this silently succeeds at construction and fails deep in provider dispatch. `GenerateOptions` correctly requires a model via `new(model)`.

**Fix:**
1. remove `Default` derive from `ChatOptions`
2. add `ChatOptions::new(model: impl Into<String>)` constructor (matching `GenerateOptions::new()`)
3. update all call sites that use `ChatOptions::default()` — grep for `ChatOptions::default()` and `ChatOptions { .. }` with no model field
4. update builder pattern methods if any rely on `Default`

---

### A6. justfile `install` target references nonexistent path `[  ]`

**File:** `justfile` (line 250)
**Severity:** CRITICAL

```just
install:
  cargo install --path crates/chibi-cli
```

there is no `crates/` directory. this is a leftover from a different project.

**Fix:** either update to `cargo install --path . --features server,client` (to install `ratd` + `rat`), or remove the target entirely.

---

## Batch B — High-Value API Hardening

one-line-per-site changes with massive long-term payoff. do these together since they touch the same files.

### B1. add `#[non_exhaustive]` to all public enums `[  ]`

**Severity:** MAJOR

without `#[non_exhaustive]`, adding a variant to any public enum post-1.0 is a breaking change for downstream `match` expressions. this is the single highest-leverage change for forward-compatibility.

**Enums to annotate:**
- `RatatoskrError` — `src/error.rs`
- `Role` — `src/types/message.rs`
- `MessageContent` — `src/types/message.rs`
- `ChatEvent` — `src/types/response.rs`
- `GenerateEvent` — `src/types/generate.rs`
- `FinishReason` — `src/types/response.rs`
- `ModelCapability` — `src/types/model.rs`
- `ModelStatus` — `src/types/model.rs`
- `ParameterName` — `src/types/parameter.rs`
- `ParameterAvailability` — `src/types/parameter.rs`
- `ParameterValidationPolicy` — `src/types/validation.rs`
- `StanceLabel` — `src/types/stance.rs`
- `NliLabel` — `src/types/future.rs` (or `inference.rs` after A4)
- `ToolChoice` — `src/types/tool.rs`
- `ReasoningEffort` — `src/types/options.rs`
- `ResponseFormat` — `src/types/options.rs`

**Note:** after adding `#[non_exhaustive]`, internal `match` arms within the crate need `_ => unreachable!()` or similar. external consumers already need wildcard arms. check all `match` expressions on these enums within the crate.

---

### B2. add `PartialEq` derives across core types `[  ]`

**Severity:** MAJOR

most core types are missing `PartialEq`, making test assertions painful. all fields on these types support `PartialEq`.

**Types to add `PartialEq` to:**
- `Message`, `MessageContent` — `src/types/message.rs`
- `ToolCall`, `ToolDefinition`, `ToolChoice` — `src/types/tool.rs`
- `ChatOptions`, `GenerateOptions`, `ReasoningConfig`, `ResponseFormat` — `src/types/options.rs`, `src/types/generate.rs`
- `ChatEvent` — `src/types/response.rs`
- `GenerateEvent` — `src/types/generate.rs`
- `ChatResponse`, `GenerateResponse`, `Usage` — `src/types/response.rs`, `src/types/generate.rs`
- `Embedding`, `NliResult`, `ClassifyResult` — `src/types/future.rs`
- `Capabilities` — `src/types/capabilities.rs`
- `ModelInfo` — `src/types/model.rs`
- `StanceResult` — `src/types/stance.rs`

**Note:** `ChatOptions` and `GenerateOptions` contain `Option<serde_json::Value>` which does implement `PartialEq`, so the derive should work. add `Eq` too where all fields are `Eq`-compatible (i.e. no floats).

---

### B3. add `Serialize`/`Deserialize` to `ChatEvent` `[  ]`

**Severity:** MAJOR

`ChatEvent` derives only `Debug, Clone`. `GenerateEvent` derives `Debug, Clone, Serialize, Deserialize` with serde tags. these are both streaming event types serving the same role. the asymmetry prevents serializing `ChatEvent` for logging, debugging, or wire format.

**Fix:** add `Serialize, Deserialize` with `#[serde(tag = "type")]` (or similar) to `ChatEvent`, matching `GenerateEvent`'s pattern. update any serde-related tests.

---

### B4. unify capability representations `[  ]`

**Severity:** MAJOR

two parallel representations of the same concept:
- `Capabilities` struct (bool fields): `chat`, `chat_streaming`, `generate`, `tool_use`, `embeddings`, `nli`, `classification`, `stance`, `token_counting`, `local_inference`
- `ModelCapability` enum: `Chat`, `Generate`, `Embed`, `Nli`, `Classify`, `Stance`

naming mismatches (`embeddings` vs `Embed`, `classification` vs `Classify`) and no conversion between them. the gateway-only fields (`chat_streaming`, `tool_use`, `token_counting`, `local_inference`) don't exist in the enum.

**Suggested approach: unify into `ModelCapability` enum + `HashSet` wrapper.**

rationale: two representations of the same concept is a classic drift bug source. every new capability requires updating both types and any bridge logic — with nothing in the compiler enforcing it. a single enum eliminates this class of bug entirely.

the gateway-specific fields (`ChatStreaming`, `ToolUse`, `TokenCounting`, `LocalInference`) are legitimately model capabilities too — a model either supports tool use or it doesn't. the "gateway vs model" distinction is really just aggregation level, not a different concept.

**Implementation sketch:**
1. add variants to `ModelCapability`: `ChatStreaming`, `ToolUse`, `TokenCounting`, `LocalInference`
2. replace `Capabilities` struct with `Capabilities(HashSet<ModelCapability>)` newtype
3. add convenience: `caps.has(ModelCapability::Chat)`, `Capabilities::chat_only()`, etc.
4. update `ModelGateway::capabilities()` return and all call sites
5. update `EmbeddedGateway::capabilities()` construction
6. update tests (`capabilities_test.rs`, `gateway_test.rs`, `traits_test.rs`)
7. update proto `Capabilities` message and conversions if applicable

**Ergonomics cost:** `caps.chat` becomes `caps.has(Chat)` — minor, and can add `caps.can_chat()` shortcuts if desired.

**Note:** this is a suggestion, not final. re-evaluate when actually working on this task — the right call depends on how the codebase looks after batches A and earlier B tasks land.

---

## Batch C — Correctness Fixes

these address logic bugs, race conditions, and error handling gaps.

### C1. `from_status` loses error context `[  ]`

**File:** `src/client/service_client.rs` (~lines 55-68)
**Severity:** MAJOR

the server's `to_status` encodes rich info into gRPC status messages, but `from_status` discards it:
- `ModelNotAvailable` — message lost entirely
- `RateLimited` — always `retry_after: None` even though server encodes duration in message
- `PermissionDenied` (`ContentFiltered`) and `OutOfRange` (`ContextLengthExceeded`) — fall through to generic `Http` catch-all

**Fix:**
1. map `PermissionDenied` → `ContentFiltered` and `OutOfRange` → `ContextLengthExceeded`
2. attempt to parse `retry_after` duration from the message string for `RateLimited`
3. preserve message string in `ModelNotAvailable` (may need to change variant to carry a `String`)

---

### C2. `LimitsConfig` parsed but never enforced `[  ]`

**File:** `src/server/config.rs` (LimitsConfig), `src/bin/ratd.rs` (build_gateway)
**Severity:** MAJOR

`max_concurrent_requests` and `request_timeout_secs` are parsed from TOML config but never wired into tonic/tower layers. operators will think they are enforced.

**Fix:** either wire them in via `tonic::transport::Server::concurrency_limit_per_connection()` + tower timeout layer, or remove the config fields and document as future work.

---

### C3. no tracing subscriber init in `ratd` `[  ]`

**File:** `src/bin/ratd.rs`
**Severity:** MAJOR

`tracing::info!` is called but no subscriber is initialized. all spans and events are silently dropped.

**Fix:** add `tracing_subscriber::fmt::init()` (or `EnvFilter`-based init for `RUST_LOG` support) at the start of `main()`. `tracing-subscriber` is likely already in the dependency tree via `tracing`.

---

### C4. `apply_routing()` missing `stance` capability `[  ]`

**File:** `src/providers/registry.rs`, `src/providers/routing.rs`
**Severity:** MAJOR

`RoutingConfig` has fields for chat, generate, embed, nli, classify — but not stance. `apply_routing()` similarly omits stance. this means stance provider ordering can never be configured.

**Fix:** add `pub stance: Option<String>` to `RoutingConfig` and a corresponding `if let Some(ref name) = config.stance { ... }` branch in `apply_routing()`. update serde, tests, and TOML config docs.

---

### C5. `ProviderLatency::record()` race on first observation `[  ]`

**File:** `src/providers/routing.rs`
**Severity:** MAJOR

two concurrent threads can both see `count == 0` and overwrite the initial EWMA value. the `count.fetch_add(1)` happens after the CAS, not atomically with it.

**Fix (pick one):**
- **(a)** accept the benign race, add a doc comment explaining it's observability-only data and the race has bounded impact
- **(b)** use a `Mutex` (latency recording is not a hot path)
- **(c)** pack count into the atomic state and CAS both together

option (a) is pragmatic for v1.0. **(b)** is cleanest.

---

### C6. `ModelCache` is unbounded `[  ]`

**File:** `src/cache/mod.rs`
**Severity:** MAJOR

`ModelCache` is `RwLock<HashMap>` with no eviction. in a long-running `ratd`, every unique model ID fetched stays in memory forever. `moka` is already a dependency.

**Fix:** replace `HashMap` with a `moka::sync::Cache` with a reasonable max-entries default (e.g. 1000), or add a configurable cap like `DiscoveryConfig`. add `len()` and `clear()` methods for observability.

---

### C7. tokenizer alias cycle → stack overflow `[  ]`

**File:** `src/tokenizer/mod.rs` (~lines 147-170)
**Severity:** MAJOR

`resolve_source` follows aliases recursively with no cycle detection. circular aliases cause a stack overflow.

**Fix:** add a depth counter or visited set:
```rust
fn resolve_source_inner(&self, model: &str, depth: u8) -> Result<TokenizerSource> {
    if depth > 10 {
        return Err(RatatoskrError::Configuration(
            format!("alias cycle detected for model: {}", model)
        ));
    }
    // ...existing logic, calling resolve_source_inner(target, depth + 1)
}
```

---

### C8. TOCTOU race in `ModelManager` budget check `[  ]`

**File:** `src/model/manager.rs` (~lines 104-121)
**Severity:** MAJOR

`can_load()` is called before acquiring the write lock. between the check and the actual load, another thread can exhaust the budget.

**Fix:** move the budget check inside the write lock, after the double-check for an existing model:
```rust
let mut models = self.embedding_models.write()...;
if let Some(provider) = models.get(&key) {
    return Ok(Arc::clone(provider));
}
if !self.can_load(model_name) {
    return Err(RatatoskrError::ModelNotAvailable);
}
// proceed to load
```

note: `can_load` reads `loaded_sizes` with a separate lock — restructure to avoid deadlock (e.g. inline the budget check using data already held).

---

### C9. irrefutable pattern in `to_llm_messages` `[  ]`

**File:** `src/convert/mod.rs` (~lines 19, 24, 51, 56)
**Severity:** MAJOR

```rust
let MessageContent::Text(text) = &msg.content;
```

this is irrefutable today but will silently break if `MessageContent` gains variants (e.g. `Image`, `MultiPart`). since `MessageContent` is public and will be `#[non_exhaustive]` after B1, this is a ticking time bomb.

**Fix:** use explicit `match`:
```rust
let text = match &msg.content {
    MessageContent::Text(t) => t.clone(),
    other => return Err(RatatoskrError::Unsupported(
        format!("message content type not supported for llm conversion: {:?}", other)
    )),
};
```

---

## Batch D — Housekeeping

dependency hygiene, test coverage gaps, stale data.

### D1. pin `llm` dependency to a specific rev `[  ]`

**File:** `Cargo.toml` (line 29)
**Severity:** MAJOR

```toml
llm = { git = "https://github.com/emesal/llm.git", branch = "emesal/ratatoskr" }
```

a git branch can move. for reproducible builds, pin to a specific commit:
```toml
llm = { git = "https://github.com/emesal/llm.git", rev = "<commit-sha>" }
```

or better: get changes merged upstream / publish the fork.

---

### D2. add integration tests for `ParameterDiscoveryCache` `[  ]`

**Severity:** MAJOR

no integration tests for builder integration, runtime recording, cache consultation during validation, or metric emission. inline unit tests in `src/cache/discovery.rs` cover basics only.

**Create:** `tests/discovery_test.rs` covering:
- builder `.discovery(DiscoveryConfig::new()...)` wiring
- `.disable_parameter_discovery()` opt-out
- runtime discovery recording when `UnsupportedParameter` errors occur
- cache consultation during validation (parameter rejected on second attempt)

---

### D3. add integration tests for `RemoteRegistry` `[  ]`

**Severity:** MAJOR

no integration tests for cache roundtrip, versioned format parsing, legacy fallback, or HTTP fetch.

**Create:** `tests/remote_registry_test.rs` covering:
- `load_cached` / `save_cache` roundtrip (use `tempfile`)
- versioned format `{ "version": 1, "models": [...] }` parsing
- legacy bare-array fallback
- `fetch_remote` via wiremock
- version rejection (version > MAX_SUPPORTED_VERSION)

---

### D4. add tests for tokenizer, convert, and device modules `[  ]`

**Severity:** MAJOR

all three modules have zero test coverage.

**Tokenizer** (`src/tokenizer/mod.rs`):
- `resolve_source`: exact match, prefix match, longest-prefix-wins, alias resolution, unknown model error
- alias cycle detection (after C7 fix)

**Convert** (`src/convert/mod.rs`):
- round-trip test for each message role
- tool call conversion
- usage conversion
- multi-message conversation with mixed roles

**Device** (`src/model/device.rs`):
- `Default` is `Cpu`, `name()` returns expected strings

---

### D5. update `seed.json` with current models `[  ]`

**File:** `src/registry/seed.json`
**Severity:** MAJOR

stale for feb 2026. missing newer models (claude opus 4, claude haiku 4, gpt-4o-mini, gemini 2.0 pro, gemini 2.5 flash, etc.). also inconsistent: only `anthropic/claude-sonnet-4` declares `stop` and `reasoning` params.

**Fix:**
1. add missing current-generation models
2. ensure all chat models consistently declare their supported parameters
3. consider using versioned format `{ "version": 1, "models": [...] }` instead of bare array

---

## Batch E — Polish

nice-to-haves for a professional release. each is small and independent.

### E1. graceful shutdown in `ratd` `[  ]`

**File:** `src/bin/ratd.rs`

use `.serve_with_shutdown()` with a SIGTERM/SIGINT handler for clean connection draining. the systemd unit in `contrib/` sends SIGTERM.

---

### E2. `ServiceClient::capabilities()` queries server `[  ]`

**File:** `src/client/service_client.rs` (~lines 106-116)

currently hardcodes `local_inference: false`. should query the server (or cache from first call) so clients see accurate capability info.

---

### E3. validate JSON in `ToolDefinition` conversion `[  ]`

**File:** `src/server/convert.rs` (~line 52)

`serde_json::from_str(&p.parameters_json).unwrap_or_default()` silently defaults on malformed JSON. at minimum log a warning, or return an error.

---

### E4. doc path discrepancy for secrets file `[  ]`

**File:** `src/server/mod.rs`, `AGENTS.md`

`server/mod.rs` and/or `AGENTS.md` may reference `~/.config/ratatoskr/secrets.toml` but actual code loads from `~/.ratatoskr/secrets.toml`. verify and align all docs.

---

### E5. filter non-content events in `generate_stream` `[  ]`

**File:** `src/providers/llm_chat.rs`

non-content chat events (reasoning, tool calls, usage) are mapped to `GenerateEvent::Text(String::new())`, emitting spurious empty-text events. use `filter_map` instead.

---

### E6. document `retry_after` as forward-looking `[  ]`

**File:** `src/error.rs`

`RateLimited { retry_after }` is always `None` from the `LLMError` conversion. add a doc comment on the variant noting this is forward-looking for when the llm crate exposes `Retry-After` headers.

---

### E7. document telemetry labels `[  ]`

**File:** `src/telemetry.rs`

`PARAMETER_DISCOVERIES_TOTAL` is missing label docs. add `/// Labels: \`provider\`, \`model\`, \`parameter\`.` to match the pattern of other metric constants.

---

### E8. `save_cache` tmp file collision risk `[  ]`

**File:** `src/registry/remote.rs` (~line 154)

tmp path is `path.with_extension("json.tmp")` — deterministic, so concurrent processes collide. use `tempfile::NamedTempFile::new_in(parent)` + `persist()`, or append PID.

---

### E9. `fetch_remote` needs a timeout `[  ]`

**File:** `src/registry/remote.rs` (~line 176)

`reqwest::get(url)` uses no timeout. a hanging fetch blocks `rat update-registry` indefinitely. add a 30s timeout via `reqwest::Client::builder().timeout(...)`.

---

### E10. seed.json should use versioned format `[  ]`

**File:** `src/registry/seed.json`

currently uses legacy bare-array `[...]` format. should use canonical `{ "version": 1, "models": [...] }` since it's the reference data.

---

### E11. `RatatoskrError` can't be `Clone` `[  ]`

**File:** `src/error.rs`

the `Json(#[from] serde_json::Error)` variant blocks `Clone`. consider wrapping as `Json(String)` (storing the display string) to enable `Clone` on the error type.

---

### E12. document `StanceResult::from_scores` tie-breaking `[  ]`

**File:** `src/types/stance.rs` (~line 38)

doc says "label determined from highest score" but doesn't mention tie-breaking order (favor > against > neutral). add to doc comment.

---

### E13. `Usage` fields `u32` → `u64` `[  ]`

**File:** `src/types/response.rs` (~line 51)

`u32` is fine today but changing post-1.0 is breaking. `u64` is future-proof with zero cost.

---

### E14. `set_parameters()` duplication `[  ]`

**Files:** `src/types/options.rs`, `src/types/generate.rs`

both `ChatOptions::set_parameters()` and `GenerateOptions::set_parameters()` are long if-chains that must stay in sync. consider a declarative macro to generate both from a single definition.

---

### E15. clean up `#[allow(dead_code)]` in production code `[  ]`

**Files:** `src/gateway/embedded.rs:54`, `src/providers/onnx_nli.rs:96`, `src/providers/openrouter_models.rs:35,59`

either use these fields or remove them. dead code annotations shouldn't ship in v1.0.

---

### E16. resolve sole TODO in codebase `[  ]`

**File:** `src/bin/ratd.rs` (line 120)

```rust
// TODO: make this configurable per-model in config.toml
```

either implement it or convert to a tracked issue and remove the TODO.

---

### E17. remove duplicate test `[  ]`

**Files:** `tests/capabilities_test.rs`, `tests/gateway_test.rs`

`test_capabilities_chat_only` exists identically in both files. remove one.

---

### E18. justfile `update-deps` safety `[  ]`

**File:** `justfile` (line 322)

`git commit -a` stages all tracked modifications (could include unrelated work) and pushes directly to dev. consider creating a chore branch or at least removing `-a`.

---

### E19. pre-release dependency audit `[  ]`

**File:** `Cargo.toml`

- `ort = "2.0.0-rc"` — upgrade to stable if available, or document as known limitation
- `vergen-gitcl = "10.0.0-beta.5"` — same

---

### E20. strengthen `response_test.rs` assertions `[  ]`

**File:** `tests/response_test.rs`

tests are mostly `is_empty()` / `matches!` smoke tests. add serde roundtrip assertions for `ChatResponse` and `ChatEvent` (after B3 adds serde derives) to catch deserialization regressions.

---

## What Was Done Well

reviewers consistently praised:

- **architecture**: clean separation of concerns, single-pattern consistency (decorator, fallback chains, per-capability traits)
- **builder API**: ergonomic, sensible defaults, well-composed
- **documentation**: thorough doc comments on nearly every public symbol
- **error handling**: `is_transient()` classification, structured retry support
- **test quality**: meaningful assertions, good edge case coverage, well-designed mock providers
- **proto design**: clean naming, proper `oneof`, `UNSPECIFIED = 0` zero values
- **centralized conversions**: single-source-of-truth in `server::convert`
