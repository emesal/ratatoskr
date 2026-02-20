# Preset Parameters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Presets gain optional generation parameters (temperature, top_p, etc.) that act as caller-overridable defaults, so `ratatoskr:free/coder` resolves to the right model *and* the right tuning.

**Architecture:** `PresetEntry` enum replaces bare `String` in all preset maps. serde untagged gives backwards compatibility. `ResolvedModel` carries optional `PresetParameters` which are applied as defaults (fill `None`, never overwrite `Some`) in `apply_resolved_chat()` / `apply_resolved_generate()`. New `ResolvePreset` gRPC RPC for introspection.

**Tech Stack:** Rust, serde (untagged enum), tonic/prost (proto), clap (CLI flags)

**Design doc:** `docs/plans/2026-02-20-preset-parameters-design.md`

---

### Task 1: PresetEntry and PresetParameters types

**Files:**
- Create: `src/registry/preset.rs`
- Modify: `src/registry/mod.rs` (add `pub mod preset;` + re-export)
- Modify: `src/lib.rs` (re-export new types)
- Test: inline unit tests in `src/registry/preset.rs`

**Step 1: Write the failing test**

In `src/registry/preset.rs`, create the module with tests at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_string_deserializes() {
        let json = r#""some/model""#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        assert!(entry.parameters().is_none());
    }

    #[test]
    fn with_params_deserializes() {
        let json = r#"{"model": "some/model", "parameters": {"temperature": 0.3, "top_p": 0.95}}"#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        let params = entry.parameters().unwrap();
        assert_eq!(params.temperature, Some(0.3));
        assert_eq!(params.top_p, Some(0.95));
        assert!(params.max_tokens.is_none());
    }

    #[test]
    fn with_params_no_parameters_key() {
        let json = r#"{"model": "some/model"}"#;
        let entry: PresetEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.model(), "some/model");
        // WithParams variant but empty parameters
        assert!(entry.parameters().map_or(true, |p| p.is_empty()));
    }

    #[test]
    fn bare_string_round_trips() {
        let entry = PresetEntry::Bare("x/model".to_owned());
        let json = serde_json::to_string(&entry).unwrap();
        let back: PresetEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model(), "x/model");
    }

    #[test]
    fn with_params_round_trips() {
        let entry = PresetEntry::WithParams {
            model: "x/model".to_owned(),
            parameters: PresetParameters {
                temperature: Some(0.5),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: PresetEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model(), "x/model");
        assert_eq!(back.parameters().unwrap().temperature, Some(0.5));
    }

    #[test]
    fn preset_parameters_is_empty() {
        assert!(PresetParameters::default().is_empty());
        assert!(!PresetParameters {
            temperature: Some(0.5),
            ..Default::default()
        }
        .is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib preset::tests -q`
Expected: FAIL — module doesn't exist yet

**Step 3: Write the implementation**

Create `src/registry/preset.rs`:

```rust
//! Preset entry types: model ID + optional default generation parameters.

use serde::{Deserialize, Serialize};

use crate::types::{ReasoningConfig, ResponseFormat, ToolChoice};

/// A single preset: model ID with optional default generation parameters.
///
/// Deserialises from either a bare string (`"model-id"`) or an object
/// (`{ "model": "...", "parameters": { ... } }`).  The bare string form
/// provides backwards compatibility with the pre-parameters format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PresetEntry {
    /// Full preset with model and optional parameters.
    WithParams {
        model: String,
        #[serde(default, skip_serializing_if = "PresetParameters::is_empty")]
        parameters: PresetParameters,
    },
    /// Legacy bare string (just a model ID).
    Bare(String),
}

impl PresetEntry {
    /// The model identifier this preset resolves to.
    pub fn model(&self) -> &str {
        match self {
            Self::WithParams { model, .. } => model,
            Self::Bare(id) => id,
        }
    }

    /// Optional preset parameters, `None` for bare entries or empty params.
    pub fn parameters(&self) -> Option<&PresetParameters> {
        match self {
            Self::WithParams { parameters, .. } if !parameters.is_empty() => Some(parameters),
            _ => None,
        }
    }
}

/// Default generation parameters attached to a preset.
///
/// All fields are optional — only set fields act as defaults.  When applied,
/// they fill `None` slots in the caller's `ChatOptions` / `GenerateOptions`
/// but never overwrite explicitly-set values.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PresetParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_provider_options: Option<serde_json::Value>,
}

impl PresetParameters {
    /// True if no parameters are set.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.max_tokens.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
            && self.seed.is_none()
            && self.stop.is_none()
            && self.reasoning.is_none()
            && self.tool_choice.is_none()
            && self.parallel_tool_calls.is_none()
            && self.response_format.is_none()
            && self.cache_prompt.is_none()
            && self.raw_provider_options.is_none()
    }
}
```

Add `pub mod preset;` to `src/registry/mod.rs` and re-export:
```rust
pub use preset::{PresetEntry, PresetParameters};
```

Add to `src/lib.rs` re-exports (in the `pub use types::` block or a new line for registry):
```rust
pub use registry::{PresetEntry, PresetParameters};
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib preset::tests -q`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/registry/preset.rs src/registry/mod.rs src/lib.rs
git commit -m "feat: add PresetEntry and PresetParameters types"
```

---

### Task 2: apply_defaults methods on PresetParameters

**Files:**
- Modify: `src/registry/preset.rs` (add methods + tests)

**Step 1: Write the failing tests**

Add to the `tests` module in `src/registry/preset.rs`:

```rust
    use crate::types::{ChatOptions, GenerateOptions};

    #[test]
    fn apply_defaults_fills_none_chat() {
        let params = PresetParameters {
            temperature: Some(0.3),
            top_p: Some(0.95),
            max_tokens: Some(4096),
            ..Default::default()
        };
        let mut opts = ChatOptions {
            model: "x".to_owned(),
            ..Default::default()
        };
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts.temperature, Some(0.3));
        assert_eq!(opts.top_p, Some(0.95));
        assert_eq!(opts.max_tokens, Some(4096));
    }

    #[test]
    fn apply_defaults_preserves_caller_chat() {
        let params = PresetParameters {
            temperature: Some(0.3),
            top_p: Some(0.95),
            ..Default::default()
        };
        let mut opts = ChatOptions {
            model: "x".to_owned(),
            temperature: Some(0.7), // caller set this
            ..Default::default()
        };
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts.temperature, Some(0.7)); // caller wins
        assert_eq!(opts.top_p, Some(0.95)); // preset fills
    }

    #[test]
    fn apply_defaults_noop_when_empty() {
        let params = PresetParameters::default();
        let mut opts = ChatOptions {
            model: "x".to_owned(),
            temperature: Some(0.5),
            ..Default::default()
        };
        let before = opts.clone();
        params.apply_defaults_to_chat(&mut opts);
        assert_eq!(opts, before);
    }

    #[test]
    fn apply_defaults_fills_none_generate() {
        let params = PresetParameters {
            temperature: Some(0.8),
            top_k: Some(50),
            ..Default::default()
        };
        let mut opts = GenerateOptions {
            model: "x".to_owned(),
            ..Default::default()
        };
        params.apply_defaults_to_generate(&mut opts);
        assert_eq!(opts.temperature, Some(0.8));
        assert_eq!(opts.top_k, Some(50));
    }

    #[test]
    fn apply_defaults_preserves_caller_generate() {
        let params = PresetParameters {
            temperature: Some(0.8),
            ..Default::default()
        };
        let mut opts = GenerateOptions {
            model: "x".to_owned(),
            temperature: Some(0.1),
            ..Default::default()
        };
        params.apply_defaults_to_generate(&mut opts);
        assert_eq!(opts.temperature, Some(0.1));
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib preset::tests -q`
Expected: FAIL — methods don't exist

**Step 3: Write the implementation**

Add methods to `impl PresetParameters` in `src/registry/preset.rs`:

```rust
impl PresetParameters {
    /// Apply these parameters as defaults to `ChatOptions`.
    ///
    /// Fills `None` fields from preset values; never overwrites `Some`.
    pub fn apply_defaults_to_chat(&self, opts: &mut ChatOptions) {
        macro_rules! fill {
            ($field:ident) => {
                if opts.$field.is_none() {
                    opts.$field = self.$field.clone();
                }
            };
        }
        fill!(temperature);
        fill!(top_p);
        fill!(top_k);
        fill!(max_tokens);
        fill!(frequency_penalty);
        fill!(presence_penalty);
        fill!(seed);
        fill!(stop);
        fill!(reasoning);
        fill!(tool_choice);
        fill!(parallel_tool_calls);
        fill!(response_format);
        fill!(cache_prompt);
        fill!(raw_provider_options);
    }

    /// Apply these parameters as defaults to `GenerateOptions`.
    ///
    /// Fills `None` fields from preset values; never overwrites `Some`.
    /// Fields not present in `GenerateOptions` (e.g. `tool_choice`) are
    /// silently ignored.
    pub fn apply_defaults_to_generate(&self, opts: &mut GenerateOptions) {
        macro_rules! fill {
            ($field:ident) => {
                if opts.$field.is_none() {
                    opts.$field = self.$field.clone();
                }
            };
        }
        fill!(temperature);
        fill!(top_p);
        fill!(top_k);
        fill!(max_tokens);
        fill!(frequency_penalty);
        fill!(presence_penalty);
        fill!(seed);
        fill!(reasoning);
        // stop: GenerateOptions uses `stop_sequences`, not `stop`
        if opts.stop_sequences.is_none() {
            opts.stop_sequences = self.stop.clone();
        }
    }
}
```

Note: `GenerateOptions` has `stop_sequences` while `ChatOptions` has `stop` — the apply method maps between them.

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib preset::tests -q`
Expected: PASS (all 11 tests)

**Step 5: Commit**

```bash
git add src/registry/preset.rs
git commit -m "feat: PresetParameters::apply_defaults_to_chat/generate"
```

---

### Task 3: Migrate ModelRegistry to PresetEntry

**Files:**
- Modify: `src/registry/mod.rs` (change `presets` type + methods)
- Test: `tests/registry_test.rs` (update existing + add new)

**Step 1: Update ModelRegistry struct and methods**

In `src/registry/mod.rs`:

- Change field: `presets: BTreeMap<String, BTreeMap<String, String>>` → `BTreeMap<String, BTreeMap<String, PresetEntry>>`
- `preset()` → returns `Option<&PresetEntry>` instead of `Option<&str>`
- `set_preset()` → takes `PresetEntry` instead of `&str`
- `merge_presets()` → accepts new map type
- `validate_presets()` → calls `entry.model()` to get the model ID

The `preset()` method body changes from:
```rust
self.presets.get(tier).and_then(|cap_map| cap_map.get(capability)).map(String::as_str)
```
to:
```rust
self.presets.get(tier).and_then(|cap_map| cap_map.get(capability))
```

`set_preset()` changes from `model_id: &str` to `entry: PresetEntry`.

`validate_presets()` changes inner usage from `model_id` to `entry.model()`.

**Step 2: Update tests in `tests/registry_test.rs`**

All existing preset tests use `set_preset("tier", "cap", "model")` — update call sites to `set_preset("tier", "cap", PresetEntry::Bare("model".to_owned()))`.

All assertions using `registry.preset("tier", "cap")` that compare to `Some("model")` need to compare to `Some(&PresetEntry::Bare("model".to_owned()))` or use `.map(|e| e.model())` to compare the model string.

Choosing the `.model()` accessor approach is cleaner — change assertions from:
```rust
assert_eq!(registry.preset("free", "text-generation"), Some("free/model"));
```
to:
```rust
assert_eq!(registry.preset("free", "text-generation").map(|e| e.model()), Some("free/model"));
```

Add a new test for parameterised presets:
```rust
#[test]
fn preset_with_parameters() {
    let mut registry = ModelRegistry::new();
    registry.set_preset(
        "budget",
        "agentic",
        PresetEntry::WithParams {
            model: "xiaomi/mimo-v2-flash".to_owned(),
            parameters: PresetParameters {
                temperature: Some(0.3),
                top_p: Some(0.95),
                ..Default::default()
            },
        },
    );
    let entry = registry.preset("budget", "agentic").unwrap();
    assert_eq!(entry.model(), "xiaomi/mimo-v2-flash");
    let params = entry.parameters().unwrap();
    assert_eq!(params.temperature, Some(0.3));
    assert_eq!(params.top_p, Some(0.95));
}
```

**Step 3: Run tests**

Run: `cargo test registry_test -q`
Expected: PASS

**Step 4: Commit**

```bash
git add src/registry/mod.rs tests/registry_test.rs
git commit -m "feat: migrate ModelRegistry presets to PresetEntry"
```

---

### Task 4: Migrate remote registry types to PresetEntry

**Files:**
- Modify: `src/registry/remote.rs` (change `RemoteRegistry`, `RegistryPayload` types)
- Test: `tests/remote_registry_test.rs` (update existing + add new)

**Step 1: Update types**

In `src/registry/remote.rs`:

- `RemoteRegistry.presets` → `BTreeMap<String, BTreeMap<String, PresetEntry>>`
- `RegistryPayload.presets` → `BTreeMap<String, BTreeMap<String, PresetEntry>>`
- `save_cache()` already uses `serde_json::to_string_pretty` — works automatically.
- `parse_payload()` — serde untagged on `PresetEntry` handles both bare strings and objects.

**Step 2: Update tests in `tests/remote_registry_test.rs`**

All tests constructing `RegistryPayload` with `presets: BTreeMap::new()` stay the same (empty maps are type-compatible).

Tests that insert preset strings like:
```rust
free_map.insert("text-generation".to_owned(), "some/free-model".to_owned());
```
become:
```rust
free_map.insert("text-generation".to_owned(), PresetEntry::Bare("some/free-model".to_owned()));
```

Assertions accessing preset values need similar `.model()` updates.

Add a test for round-tripping parameterised presets through save/load.

**Step 3: Run tests**

Run: `cargo test remote_registry_test -q`
Expected: PASS

**Step 4: Run full test suite**

Run: `cargo test -q`
Expected: PASS — no regressions

**Step 5: Commit**

```bash
git add src/registry/remote.rs tests/remote_registry_test.rs
git commit -m "feat: migrate remote registry types to PresetEntry"
```

---

### Task 5: Update ResolvedModel and gateway resolution

**Files:**
- Modify: `src/gateway/embedded.rs` (ResolvedModel + resolve_model + apply_resolved_*)
- Tests: inline tests in `src/gateway/embedded.rs`

**Step 1: Extend ResolvedModel**

Add field:
```rust
pub(crate) struct ResolvedModel {
    pub provider: Option<String>,
    pub model: String,
    pub preset_parameters: Option<PresetParameters>,
}
```

**Step 2: Update resolve_model()**

In the preset branch, after looking up the `PresetEntry`:
```rust
let entry = self.model_registry.preset(tier, capability)
    .ok_or_else(|| ...)?;
Ok(ResolvedModel {
    provider: None,
    model: entry.model().to_owned(),
    preset_parameters: entry.parameters().cloned(),
})
```

Non-preset branches set `preset_parameters: None`.

**Step 3: Update apply_resolved_chat() and apply_resolved_generate()**

```rust
fn apply_resolved_chat(&self, options: &ChatOptions, resolved: &ResolvedModel) -> Option<ChatOptions> {
    let needs_model_swap = resolved.model != options.model;
    let has_params = resolved.preset_parameters.is_some();

    if !needs_model_swap && !has_params {
        return None;
    }

    let mut opts = options.clone();
    opts.model = resolved.model.clone();

    if let Some(params) = &resolved.preset_parameters {
        params.apply_defaults_to_chat(&mut opts);
    }

    Some(opts)
}
```

Same pattern for `apply_resolved_generate()`.

**Step 4: Fix existing tests**

All `ResolvedModel` construction in tests needs `preset_parameters: None` added.

Add new tests:
```rust
#[test]
fn test_resolve_preset_carries_parameters() {
    // build a gateway whose registry has a parameterised preset
    // assert resolved.preset_parameters is Some with expected values
}

#[test]
fn test_resolve_plain_model_no_parameters() {
    let gw = test_gateway();
    let resolved = gw.resolve_model("anthropic/claude-sonnet-4").unwrap();
    assert!(resolved.preset_parameters.is_none());
}
```

**Step 5: Run tests**

Run: `cargo test --lib gateway -q && cargo test gateway_test -q`
Expected: PASS

**Step 6: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: carry preset parameters through model resolution"
```

---

### Task 6: Update ModelGateway trait resolve_preset

**Files:**
- Modify: `src/traits.rs` (change return type)
- Create or modify: `src/types/mod.rs` or appropriate file (add `PresetResolution` type)
- Modify: `src/gateway/embedded.rs` (update `resolve_preset` impl)
- Test: `tests/gateway_test.rs` (update existing)

**Step 1: Add PresetResolution type**

Either in `src/registry/preset.rs` or `src/types/`:

```rust
/// Result of resolving a preset: the concrete model ID and any default parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PresetResolution {
    /// The resolved model identifier.
    pub model: String,
    /// Optional default parameters for this preset.
    pub parameters: Option<PresetParameters>,
}
```

**Step 2: Update trait**

In `src/traits.rs`:
```rust
fn resolve_preset(&self, _tier: &str, _capability: &str) -> Option<PresetResolution> {
    None
}
```

**Step 3: Update EmbeddedGateway impl**

```rust
fn resolve_preset(&self, tier: &str, capability: &str) -> Option<PresetResolution> {
    self.model_registry.preset(tier, capability).map(|entry| PresetResolution {
        model: entry.model().to_owned(),
        parameters: entry.parameters().cloned(),
    })
}
```

**Step 4: Update tests**

In `tests/gateway_test.rs`, `test_preset_uri_resolve_preset_still_works` needs to match the new return type.

**Step 5: Run tests**

Run: `cargo test gateway_test -q`
Expected: PASS

**Step 6: Commit**

```bash
git add src/traits.rs src/registry/preset.rs src/gateway/embedded.rs src/lib.rs tests/gateway_test.rs
git commit -m "feat: resolve_preset returns PresetResolution with parameters"
```

---

### Task 7: Proto + gRPC ResolvePreset RPC

**Files:**
- Modify: `proto/ratatoskr.proto` (add messages + RPC)
- Modify: `src/server/convert.rs` (add conversions)
- Modify: `src/server/service.rs` (add handler)
- Modify: `src/client/service_client.rs` (implement resolve_preset)
- Test: `tests/service_test.rs` (if exists, add test)

**Step 1: Proto changes**

Add to `ratatoskr.proto` service block:
```protobuf
rpc ResolvePreset(ResolvePresetRequest) returns (ResolvePresetResponse);
```

Add message types:
```protobuf
message ResolvePresetRequest {
  string tier = 1;
  string capability = 2;
}

message ProtoPresetParameters {
  optional float temperature = 1;
  optional float top_p = 2;
  optional uint32 top_k = 3;
  optional uint64 max_tokens = 4;
  optional float frequency_penalty = 5;
  optional float presence_penalty = 6;
  optional uint64 seed = 7;
  repeated string stop = 8;
}

message ResolvePresetResponse {
  bool found = 1;
  optional string model_id = 2;
  optional ProtoPresetParameters parameters = 3;
}
```

**Step 2: Add conversions in `src/server/convert.rs`**

`From<PresetParameters> for proto::ProtoPresetParameters` and reverse.
`From<PresetResolution> for proto::ResolvePresetResponse` and reverse.

**Step 3: Add handler in `src/server/service.rs`**

```rust
async fn resolve_preset(&self, request: Request<ResolvePresetRequest>) -> Result<Response<ResolvePresetResponse>, Status> {
    let req = request.into_inner();
    match self.gateway.resolve_preset(&req.tier, &req.capability) {
        Some(resolution) => Ok(Response::new(resolution.into())),
        None => Ok(Response::new(ResolvePresetResponse { found: false, model_id: None, parameters: None })),
    }
}
```

**Step 4: Implement on ServiceClient**

```rust
fn resolve_preset(&self, tier: &str, capability: &str) -> Option<PresetResolution> {
    // This is a sync trait method; use block_on or cache.
    // Match existing patterns in ServiceClient for sync methods.
}
```

Check how other sync methods are handled in `ServiceClient` — may need `block_in_place` or similar.

**Step 5: Run tests**

Run: `cargo test --features server,client -q`
Expected: PASS

**Step 6: Commit**

```bash
git add proto/ratatoskr.proto src/server/convert.rs src/server/service.rs src/client/service_client.rs
git commit -m "feat: ResolvePreset gRPC RPC"
```

---

### Task 8: Update rat_registry CLI

**Files:**
- Modify: `src/bin/rat_registry.rs`

**Step 1: Update PresetCommand::Set**

Add optional clap flags:
```rust
Set {
    tier: String,
    slot: String,
    model_id: String,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    max_tokens: Option<usize>,
    #[arg(long)]
    frequency_penalty: Option<f32>,
    #[arg(long)]
    presence_penalty: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
},
```

**Step 2: Update preset_set()**

Build `PresetParameters` from the flags. If all are `None`, use `PresetEntry::Bare`. Otherwise use `PresetEntry::WithParams`.

Update the insert call:
```rust
.insert(slot.to_string(), entry);
```

**Step 3: Update preset_list()**

The display currently prints the model ID from the string value. Update to call `.model()` on the `PresetEntry`. Optionally show a marker (e.g. `*`) for presets that have parameters.

**Step 4: Update helper functions**

- `known_slots()` — iterates `slots.keys()`, no change needed (map key is still `String`)
- `preset_references()` — changes `id == model_id` to `entry.model() == model_id`

**Step 5: Run full test suite**

Run: `cargo test -q`
Expected: PASS

**Step 6: Commit**

```bash
git add src/bin/rat_registry.rs
git commit -m "feat: rat registry preset set gains parameter flags"
```

---

### Task 9: Update registry.json and seed.json

**Files:**
- Modify: `../registry/registry.json` (add parameters to relevant presets)

**Step 1: Update the upstream registry**

Add parameters to presets that benefit from them. Example for budget/agentic:
```json
"agentic": {
  "model": "xiaomi/mimo-v2-flash",
  "parameters": { "temperature": 0.3, "top_p": 0.95 }
}
```

The specific parameter values per preset should be determined based on model
recommendations. Start with the known ones (MiMo-v2-Flash) and leave others
as bare strings until we have recommendations.

**Step 2: Regenerate seed.json**

Follow whatever the existing seed generation process is (likely a copy or
script). Verify the new seed.json parses correctly.

**Step 3: Run full test suite**

Run: `cargo test -q`
Expected: PASS (seed tests should still work with updated format)

**Step 4: Commit**

Commit in the registry repo, then update the seed in this repo.

---

### Task 10: Final verification

**Step 1: Lint**

Run: `just lint`
Expected: PASS

**Step 2: Full test suite**

Run: `just test`
Expected: PASS

**Step 3: Docs build**

Run: `cargo doc --no-deps`
Expected: PASS — new types are documented

**Step 4: Commit any fixes**

If anything needed fixing, commit.

**Step 5: Squash or tidy commits if desired**

Run: `just pre-push`
