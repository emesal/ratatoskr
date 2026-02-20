# Preset Parameters Design

**date**: 2026-02-20
**branch**: `feature/preset-parameters-2602`
**status**: approved

## Problem

Presets currently map `tier/capability` to a model ID, but some models need
specific generation parameters for optimal results depending on the use-case.
For example, Xiaomi recommends `temperature=0.3` for agentic tasks with
MiMo-v2-Flash but `temperature=0.8` for writing — the same model, different
purposes.  Consumers using `ratatoskr:free/coder` should get sensible defaults
without needing to know the underlying model's tuning recommendations.

## Design

### Approach

Inline parameters in the preset value (approach A).  Change the preset value
from a plain string to a `PresetEntry` enum that accepts either a bare model ID
(backwards-compatible) or a struct with `model` + `parameters`.

Rejected alternatives:
- **Separate `preset_parameters` map** — violates single-source-of-truth.
- **Parameters on `ModelMetadata`** — per-model, not per-purpose.

### Data Model

```rust
/// a single preset: model ID + optional default generation parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PresetEntry {
    /// full preset with model and optional parameters
    WithParams {
        model: String,
        #[serde(default, skip_serializing_if = "PresetParameters::is_empty")]
        parameters: PresetParameters,
    },
    /// legacy bare string (just a model ID)
    Bare(String),
}
```

`PresetParameters` mirrors all `Option<T>` fields from `ChatOptions` and
`GenerateOptions`: temperature, top_p, top_k, max_tokens, frequency_penalty,
presence_penalty, seed, stop, reasoning, tool_choice, parallel_tool_calls,
response_format, cache_prompt, raw_provider_options.  All fields are
`Option<T>` — absent in JSON means "no default".

Helpers: `PresetEntry::model() -> &str`,
`PresetEntry::parameters() -> Option<&PresetParameters>`,
`PresetParameters::is_empty() -> bool`.

### Override Semantics

Preset parameters act as **defaults**.  They fill `None` fields in the caller's
`ChatOptions` / `GenerateOptions` but never overwrite explicitly-set values.

One parameter set per preset — no per-operation-type distinction.

### JSON Format

```json
{
  "presets": {
    "budget": {
      "agentic": {
        "model": "xiaomi/mimo-v2-flash",
        "parameters": { "temperature": 0.3, "top_p": 0.95 }
      },
      "text-generation": {
        "model": "mistralai/mistral-small-creative",
        "parameters": { "temperature": 0.8 }
      },
      "embedding": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

Bare strings still parse via `#[serde(untagged)]` on `PresetEntry`.

### Resolution Flow

`ResolvedModel` gains a `preset_parameters: Option<PresetParameters>` field.

`resolve_model()` extracts both model ID and parameters from the `PresetEntry`.
Non-preset model strings resolve with `preset_parameters: None` (zero-cost
path).

`apply_resolved_chat()` / `apply_resolved_generate()` call
`PresetParameters::apply_defaults_to_chat(&mut ChatOptions)` /
`apply_defaults_to_generate(&mut GenerateOptions)` — fill `None` fields from
preset values, never overwrite `Some`.

### Registry & Storage

- `ModelRegistry::presets` type changes from
  `BTreeMap<String, BTreeMap<String, String>>` to
  `BTreeMap<String, BTreeMap<String, PresetEntry>>`.
- `preset()` returns `Option<&PresetEntry>`.
- `set_preset()` takes a `PresetEntry`.
- `merge_presets()` — incoming entry fully replaces existing for same
  tier+capability (no deep-merge of individual parameter fields).
- `validate_presets()` calls `entry.model()` to extract the model ID.
- `RegistryPayload` / `VersionedPayload` get the same type change.
- `seed.json` is generated from `../registry/registry.json` — update that
  file and seed follows.

### CLI (`rat registry preset set`)

Extended with optional parameter flags:

```
rat registry preset set budget agentic xiaomi/mimo-v2-flash \
    --temperature 0.3 --top-p 0.95
```

No flags = bare model entry (no parameters).  Existing behaviour preserved.

### Proto / gRPC

New `ResolvePreset` RPC:

```protobuf
message ResolvePresetRequest {
  string tier = 1;
  string capability = 2;
}

message PresetParameters {
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
  string model_id = 1;
  PresetParameters parameters = 2;
}
```

Complex fields (reasoning, tool_choice, response_format, raw_provider_options)
deferred from proto — added when needed.  The Rust `PresetParameters` struct
has all fields; the proto exposes the commonly-used numeric/simple ones.

`ModelGateway::resolve_preset()` return type changes to
`Option<PresetResolution>` where `PresetResolution { model: String,
parameters: Option<PresetParameters> }`.  `ServiceClient` implements via the
new RPC.

### Testing

**Unit tests:**
- `PresetEntry` serde: bare string, struct form, untagged dispatch
- `PresetParameters::apply_defaults_to_chat()`: None filled, Some preserved,
  empty params no-op
- `PresetParameters::apply_defaults_to_generate()`: same
- `PresetParameters::is_empty()`

**Registry tests:**
- `set_preset` / `preset()` with `PresetEntry::WithParams`
- `merge_presets` with mixed bare + parameterised entries
- `validate_presets` extracts model ID from either variant
- legacy seed.json bare strings still load

**Gateway tests:**
- `resolve_model("ratatoskr:budget/agentic")` → `preset_parameters: Some(...)`
- `apply_resolved_chat()` fills defaults, preserves caller values
- non-preset models → `preset_parameters: None`

**Remote registry tests:**
- save/load round-trip with parameterised presets
- legacy cached JSON still parses
- versioned format without presets still works

**Proto / service tests:**
- `ResolvePreset` RPC returns model + parameters
- `ServiceClient::resolve_preset()` matches `EmbeddedGateway`

No new live tests needed.
