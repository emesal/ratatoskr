# Phase 6: Model Intelligence & Parameter Surface — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** give consumers (orlog, chibi) full visibility into model capabilities and control over the complete parameter surface, with a unified model registry that merges embedded seed data with live provider metadata.

**Architecture:** a three-layer model registry (embedded JSON seed → live provider queries → future remote updates) backs a new `model_metadata()` trait method. parameter introspection uses a hybrid `ParameterName` enum (well-known params + `Custom(String)` escape hatch). providers declare their supported parameters, enabling opt-in validation. `GenerateOptions` is brought to parity with `ChatOptions`. all changes flow through proto and conversions.

**Tech Stack:** rust, tonic/prost (proto), serde_json (registry seed), existing test patterns (wiremock, integration tests)

---

## Design Decisions (Resolved)

| question | decision | rationale |
|----------|----------|-----------|
| 6b/6c coupling | unified from the start | single source of truth; registry holds `ModelMetadata`, `model_metadata()` queries it |
| parameter namespace | hybrid enum + `Custom(String)` | type-safe for well-known params, extensible for provider-specific ones |
| embedded registry format | JSON via `include_str!` | machine-written machine-read; same format as future remote updates (#6) |
| registry refresh | startup only (live); periodic deferred to #6 | keep scope tight; infrastructure supports future refresh |

## Task Overview

```
task 1:  ParameterName enum + ParameterAvailability + ParameterRange types     ✓
task 2:  ModelMetadata + PricingInfo types                                      ✓
task 3:  add top_k to ChatOptions (+ proto + convert)                          ✓
task 4:  bring GenerateOptions to parity (+ proto + convert)                   ✓
task 5:  raw_provider_options in proto ChatOptions (escape hatch)
task 6:  provider parameter declaration (supported_parameters on traits)
task 7:  model registry core (ModelRegistry struct, merge logic)
task 8:  embedded JSON seed file + registry loading
task 9:  model_metadata() on ModelGateway trait
task 10: EmbeddedGateway + ServiceClient integration
task 11: proto ModelMetadata RPC + messages
task 12: parameter validation in ProviderRegistry
task 13: UnsupportedParameter error variant
task 14: update AGENTS.md and docs
```

Dependencies: task 1 → 2 → 7 → 8 → 9 → 10. task 3, 4, 5 are independent. task 6 → 12. task 11 depends on 2 and 9. task 13 depends on 12.

### Implementation Notes

- **Task 1 deviation:** `ParameterRange::default()` builder renamed to `default_value()` to avoid shadowing `Default::default()`.
- **Task 1 deviation:** `ParameterName` uses custom `Serialize`/`Deserialize` impls (flat string) instead of `#[serde(rename_all)]` — required for use as `HashMap` keys in JSON.
- **Tasks 1-4 verified:** `just pre-push` passes (105 tests, 0 clippy warnings).

---

## Task 1: Parameter Types

**Files:**
- Create: `src/types/parameter.rs`
- Modify: `src/types/mod.rs`
- Modify: `src/lib.rs`
- Test: `tests/parameter_test.rs`

### Step 1: Write failing tests

```rust
// tests/parameter_test.rs
use ratatoskr::{ParameterAvailability, ParameterName, ParameterRange};

#[test]
fn well_known_parameter_names() {
    // all well-known params exist and display correctly
    assert_eq!(ParameterName::Temperature.as_str(), "temperature");
    assert_eq!(ParameterName::TopP.as_str(), "top_p");
    assert_eq!(ParameterName::TopK.as_str(), "top_k");
    assert_eq!(ParameterName::MaxTokens.as_str(), "max_tokens");
    assert_eq!(ParameterName::FrequencyPenalty.as_str(), "frequency_penalty");
    assert_eq!(ParameterName::PresencePenalty.as_str(), "presence_penalty");
    assert_eq!(ParameterName::Seed.as_str(), "seed");
    assert_eq!(ParameterName::Stop.as_str(), "stop");
    assert_eq!(ParameterName::Reasoning.as_str(), "reasoning");
    assert_eq!(ParameterName::CachePrompt.as_str(), "cache_prompt");
    assert_eq!(ParameterName::ResponseFormat.as_str(), "response_format");
    assert_eq!(ParameterName::ToolChoice.as_str(), "tool_choice");
}

#[test]
fn custom_parameter_name() {
    let custom = ParameterName::Custom("logit_bias".into());
    assert_eq!(custom.as_str(), "logit_bias");
}

#[test]
fn parameter_name_from_str() {
    assert_eq!("temperature".parse::<ParameterName>().unwrap(), ParameterName::Temperature);
    assert_eq!("top_k".parse::<ParameterName>().unwrap(), ParameterName::TopK);
    assert_eq!("logit_bias".parse::<ParameterName>().unwrap(), ParameterName::Custom("logit_bias".into()));
}

#[test]
fn parameter_name_serde_roundtrip() {
    let names = vec![ParameterName::Temperature, ParameterName::Custom("logit_bias".into())];
    let json = serde_json::to_string(&names).unwrap();
    let parsed: Vec<ParameterName> = serde_json::from_str(&json).unwrap();
    assert_eq!(names, parsed);
}

#[test]
fn parameter_range_defaults() {
    let range = ParameterRange::default();
    assert!(range.min.is_none());
    assert!(range.max.is_none());
    assert!(range.default.is_none());
}

#[test]
fn parameter_range_builder() {
    let range = ParameterRange::new().min(0.0).max(2.0).default(1.0);
    assert_eq!(range.min, Some(0.0));
    assert_eq!(range.max, Some(2.0));
    assert_eq!(range.default, Some(1.0));
}

#[test]
fn parameter_availability_variants() {
    let mutable = ParameterAvailability::Mutable {
        range: ParameterRange::new().min(0.0).max(2.0),
    };
    let read_only = ParameterAvailability::ReadOnly {
        value: serde_json::json!(0.7),
    };
    let opaque = ParameterAvailability::Opaque;
    let unsupported = ParameterAvailability::Unsupported;

    assert!(matches!(mutable, ParameterAvailability::Mutable { .. }));
    assert!(matches!(read_only, ParameterAvailability::ReadOnly { .. }));
    assert!(matches!(opaque, ParameterAvailability::Opaque));
    assert!(matches!(unsupported, ParameterAvailability::Unsupported));
}

#[test]
fn parameter_availability_serde_roundtrip() {
    let avail = ParameterAvailability::Mutable {
        range: ParameterRange::new().min(0.0).max(2.0).default(1.0),
    };
    let json = serde_json::to_string(&avail).unwrap();
    let parsed: ParameterAvailability = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, ParameterAvailability::Mutable { .. }));
}

#[test]
fn parameter_availability_is_supported() {
    assert!(ParameterAvailability::Mutable { range: ParameterRange::default() }.is_supported());
    assert!(ParameterAvailability::ReadOnly { value: serde_json::json!(1) }.is_supported());
    assert!(ParameterAvailability::Opaque.is_supported());
    assert!(!ParameterAvailability::Unsupported.is_supported());
}
```

### Step 2: Run tests to verify they fail

Run: `cargo test --test parameter_test`
Expected: compilation error — types don't exist yet

### Step 3: Implement parameter types

```rust
// src/types/parameter.rs
//! Parameter metadata types for model introspection.
//!
//! These types describe what parameters a model supports and their constraints.
//! Used by the model registry and `model_metadata()` for consumer introspection.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Well-known parameter names with a `Custom` escape hatch.
///
/// The well-known variants cover the standard parameter surface shared across
/// providers (OpenAI conventions). `Custom(String)` handles provider-specific
/// parameters without requiring a ratatoskr release.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterName {
    Temperature,
    TopP,
    TopK,
    MaxTokens,
    FrequencyPenalty,
    PresencePenalty,
    Seed,
    Stop,
    Reasoning,
    CachePrompt,
    ResponseFormat,
    ToolChoice,
    /// Provider-specific parameter not in the well-known set.
    Custom(String),
}

impl ParameterName {
    /// Canonical string representation matching `ChatOptions`/`GenerateOptions` field names.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Temperature => "temperature",
            Self::TopP => "top_p",
            Self::TopK => "top_k",
            Self::MaxTokens => "max_tokens",
            Self::FrequencyPenalty => "frequency_penalty",
            Self::PresencePenalty => "presence_penalty",
            Self::Seed => "seed",
            Self::Stop => "stop",
            Self::Reasoning => "reasoning",
            Self::CachePrompt => "cache_prompt",
            Self::ResponseFormat => "response_format",
            Self::ToolChoice => "tool_choice",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for ParameterName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ParameterName {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "temperature" => Self::Temperature,
            "top_p" => Self::TopP,
            "top_k" => Self::TopK,
            "max_tokens" => Self::MaxTokens,
            "frequency_penalty" => Self::FrequencyPenalty,
            "presence_penalty" => Self::PresencePenalty,
            "seed" => Self::Seed,
            "stop" => Self::Stop,
            "reasoning" => Self::Reasoning,
            "cache_prompt" => Self::CachePrompt,
            "response_format" => Self::ResponseFormat,
            "tool_choice" => Self::ToolChoice,
            other => Self::Custom(other.to_string()),
        })
    }
}

/// Numeric range constraints for a parameter.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParameterRange {
    /// Minimum allowed value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum allowed value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Default value if not specified by the consumer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
}

impl ParameterRange {
    /// Create an empty range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum value.
    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value.
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set default value.
    pub fn default(mut self, default: f64) -> Self {
        self.default = Some(default);
        self
    }
}

/// How a parameter is exposed for a given model.
///
/// This tells consumers whether they can set a parameter, what range it accepts,
/// or whether it's fixed/unsupported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
pub enum ParameterAvailability {
    /// Consumer can set this freely within range.
    Mutable {
        #[serde(default)]
        range: ParameterRange,
    },
    /// Value is fixed by the provider/model.
    ReadOnly { value: serde_json::Value },
    /// Parameter exists but constraints are unknown.
    Opaque,
    /// Parameter is not supported by this model.
    Unsupported,
}

impl ParameterAvailability {
    /// Whether this parameter is supported (anything other than `Unsupported`).
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}
```

### Step 4: Wire up module exports

Add to `src/types/mod.rs`:
```rust
mod parameter;
pub use parameter::{ParameterAvailability, ParameterName, ParameterRange};
```

Add to `src/lib.rs` re-exports:
```rust
pub use types::{
    // ... existing ...
    ParameterAvailability, ParameterName, ParameterRange,
};
```

### Step 5: Run tests

Run: `cargo test --test parameter_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS, no clippy warnings

### Step 7: Commit

```bash
git add src/types/parameter.rs src/types/mod.rs src/lib.rs tests/parameter_test.rs
git commit -m "feat: add ParameterName, ParameterRange, ParameterAvailability types

Hybrid enum with well-known params + Custom(String) escape hatch.
Serde-compatible for JSON registry and future proto representation."
```

---

## Task 2: ModelMetadata and PricingInfo Types

**Files:**
- Modify: `src/types/model.rs`
- Modify: `src/types/mod.rs`
- Modify: `src/lib.rs`
- Test: `tests/model_metadata_test.rs`

### Step 1: Write failing tests

```rust
// tests/model_metadata_test.rs
use std::collections::HashMap;
use ratatoskr::{
    ModelCapability, ModelInfo, ModelMetadata, ParameterAvailability, ParameterName,
    ParameterRange, PricingInfo,
};

#[test]
fn model_metadata_from_info() {
    let info = ModelInfo::new("claude-sonnet-4", "openrouter")
        .with_capability(ModelCapability::Chat)
        .with_context_window(200_000);

    let metadata = ModelMetadata::from_info(info.clone());
    assert_eq!(metadata.info.id, "claude-sonnet-4");
    assert!(metadata.parameters.is_empty());
    assert!(metadata.pricing.is_none());
    assert!(metadata.max_output_tokens.is_none());
}

#[test]
fn model_metadata_builder() {
    let metadata = ModelMetadata::from_info(ModelInfo::new("gpt-4o", "openrouter"))
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(2.0).default(1.0),
            },
        )
        .with_parameter(
            ParameterName::TopK,
            ParameterAvailability::Unsupported,
        )
        .with_max_output_tokens(16384)
        .with_pricing(PricingInfo {
            prompt_cost_per_mtok: Some(2.50),
            completion_cost_per_mtok: Some(10.0),
        });

    assert_eq!(metadata.parameters.len(), 2);
    assert!(metadata.parameters[&ParameterName::Temperature].is_supported());
    assert!(!metadata.parameters[&ParameterName::TopK].is_supported());
    assert_eq!(metadata.max_output_tokens, Some(16384));
    assert_eq!(metadata.pricing.as_ref().unwrap().prompt_cost_per_mtok, Some(2.50));
}

#[test]
fn model_metadata_serde_roundtrip() {
    let mut params = HashMap::new();
    params.insert(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(2.0),
        },
    );
    params.insert(
        ParameterName::Custom("logit_bias".into()),
        ParameterAvailability::Opaque,
    );

    let metadata = ModelMetadata {
        info: ModelInfo::new("test", "test").with_capability(ModelCapability::Chat),
        parameters: params,
        pricing: Some(PricingInfo {
            prompt_cost_per_mtok: Some(1.0),
            completion_cost_per_mtok: Some(2.0),
        }),
        max_output_tokens: Some(4096),
    };

    let json = serde_json::to_string(&metadata).unwrap();
    let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.info.id, "test");
    assert_eq!(parsed.parameters.len(), 2);
    assert!(parsed.parameters.contains_key(&ParameterName::Temperature));
    assert!(parsed.parameters.contains_key(&ParameterName::Custom("logit_bias".into())));
    assert_eq!(parsed.max_output_tokens, Some(4096));
}

#[test]
fn pricing_info_default() {
    let pricing = PricingInfo::default();
    assert!(pricing.prompt_cost_per_mtok.is_none());
    assert!(pricing.completion_cost_per_mtok.is_none());
}

#[test]
fn model_metadata_merge_parameters() {
    let base = ModelMetadata::from_info(ModelInfo::new("test", "test"))
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(1.0),
            },
        )
        .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported);

    // override merges on top — more specific data wins
    let override_params = {
        let mut m = HashMap::new();
        m.insert(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().min(0.0).max(2.0).default(1.0),
            },
        );
        m.insert(ParameterName::Seed, ParameterAvailability::Opaque);
        m
    };

    let merged = base.merge_parameters(override_params);
    // temperature updated to new range
    assert!(matches!(
        merged.parameters[&ParameterName::Temperature],
        ParameterAvailability::Mutable { ref range } if range.max == Some(2.0)
    ));
    // top_k preserved from base
    assert!(matches!(
        merged.parameters[&ParameterName::TopK],
        ParameterAvailability::Unsupported
    ));
    // seed added from override
    assert!(matches!(
        merged.parameters[&ParameterName::Seed],
        ParameterAvailability::Opaque
    ));
}
```

### Step 2: Run tests to verify failure

Run: `cargo test --test model_metadata_test`
Expected: compilation error — `ModelMetadata`, `PricingInfo` don't exist

### Step 3: Implement types

Add to `src/types/model.rs` (after existing types):

```rust
use std::collections::HashMap;
use super::parameter::{ParameterAvailability, ParameterName};

/// Pricing information for a model (cost per million tokens).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingInfo {
    /// Cost per million prompt/input tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cost_per_mtok: Option<f64>,
    /// Cost per million completion/output tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_cost_per_mtok: Option<f64>,
}

/// Extended model metadata including parameter availability and pricing.
///
/// This is the primary type returned by the model registry. It extends
/// [`ModelInfo`] with parameter constraints and cost information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Basic model information.
    pub info: ModelInfo,
    /// Per-parameter availability and constraints.
    #[serde(default)]
    pub parameters: HashMap<ParameterName, ParameterAvailability>,
    /// Pricing information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingInfo>,
    /// Maximum output tokens (distinct from context window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
}

impl ModelMetadata {
    /// Create metadata from a [`ModelInfo`] with empty parameter map.
    pub fn from_info(info: ModelInfo) -> Self {
        Self {
            info,
            parameters: HashMap::new(),
            pricing: None,
            max_output_tokens: None,
        }
    }

    /// Add a parameter declaration.
    pub fn with_parameter(mut self, name: ParameterName, availability: ParameterAvailability) -> Self {
        self.parameters.insert(name, availability);
        self
    }

    /// Set pricing information.
    pub fn with_pricing(mut self, pricing: PricingInfo) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Set maximum output tokens.
    pub fn with_max_output_tokens(mut self, max: usize) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// Merge parameter overrides into this metadata.
    ///
    /// Override values replace existing entries; base entries not present
    /// in the override are preserved. Used for the registry merge strategy
    /// (live data overrides embedded defaults).
    pub fn merge_parameters(
        mut self,
        overrides: HashMap<ParameterName, ParameterAvailability>,
    ) -> Self {
        for (name, avail) in overrides {
            self.parameters.insert(name, avail);
        }
        self
    }
}
```

### Step 4: Wire up exports

Add to `src/types/mod.rs`:
```rust
pub use model::{ModelMetadata, PricingInfo};
```

Add to `src/lib.rs` re-exports:
```rust
pub use types::{
    // ... existing ...
    ModelMetadata, PricingInfo,
};
```

### Step 5: Run tests

Run: `cargo test --test model_metadata_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/types/model.rs src/types/mod.rs src/lib.rs tests/model_metadata_test.rs
git commit -m "feat: add ModelMetadata, PricingInfo types

Extended model metadata with parameter availability map, pricing,
and max output tokens. Supports merge for registry layering."
```

---

## Task 3: Add top_k to ChatOptions

**Files:**
- Modify: `src/types/options.rs`
- Modify: `proto/ratatoskr.proto`
- Modify: `src/server/convert.rs`
- Modify: `tests/options_test.rs`
- Modify: `tests/convert_test.rs`

### Step 1: Write failing test

Add to `tests/options_test.rs`:
```rust
#[test]
fn test_chat_options_top_k() {
    let opts = ChatOptions::default().model("gpt-4").top_k(40);
    assert_eq!(opts.top_k, Some(40));
}

#[test]
fn test_chat_options_top_k_serde() {
    let opts = ChatOptions::default().model("test").top_k(50);
    let json = serde_json::to_string(&opts).unwrap();
    let parsed: ChatOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.top_k, Some(50));
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test options_test`
Expected: compilation error — `top_k` doesn't exist on `ChatOptions`

### Step 3: Add field and builder method to `src/types/options.rs`

Add field to `ChatOptions` struct (after `top_p`):
```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
```

Add builder method:
```rust
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
```

### Step 4: Update proto `ChatOptions` message

Add to `proto/ratatoskr.proto` `ChatOptions` message (use field number 13):
```proto
    optional uint32 top_k = 13;
```

### Step 5: Update conversions in `src/server/convert.rs`

In `From<proto::ChatOptions> for ChatOptions`:
```rust
    top_k: p.top_k.map(|k| k as usize),
```

In `From<ChatOptions> for proto::ChatOptions`:
```rust
    top_k: o.top_k.map(|k| k as u32),
```

### Step 6: Run tests

Run: `cargo test --test options_test && cargo test --test convert_test`
Expected: all PASS

### Step 7: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 8: Commit

```bash
git add src/types/options.rs proto/ratatoskr.proto src/server/convert.rs tests/options_test.rs
git commit -m "feat: add top_k to ChatOptions

New optional parameter for top-k sampling, plumbed through
proto and bidirectional conversions."
```

---

## Task 4: Bring GenerateOptions to Parity with ChatOptions

**Files:**
- Modify: `src/types/generate.rs`
- Modify: `proto/ratatoskr.proto`
- Modify: `src/server/convert.rs`
- Modify: `src/providers/llm_chat.rs` (pass new params through)
- Test: `tests/generate_test.rs`

### Step 1: Write failing tests

Add to `tests/generate_test.rs` (or create if needed — check existing content first):

```rust
use ratatoskr::{GenerateOptions, ReasoningConfig, ReasoningEffort};

#[test]
fn generate_options_new_fields() {
    let opts = GenerateOptions::new("llama3")
        .top_k(40)
        .frequency_penalty(0.5)
        .presence_penalty(0.3)
        .seed(42)
        .reasoning(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude_from_output: None,
        });

    assert_eq!(opts.top_k, Some(40));
    assert_eq!(opts.frequency_penalty, Some(0.5));
    assert_eq!(opts.presence_penalty, Some(0.3));
    assert_eq!(opts.seed, Some(42));
    assert!(opts.reasoning.is_some());
}

#[test]
fn generate_options_serde_with_new_fields() {
    let opts = GenerateOptions::new("test")
        .top_k(50)
        .frequency_penalty(0.1)
        .seed(123);

    let json = serde_json::to_string(&opts).unwrap();
    let parsed: GenerateOptions = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.top_k, Some(50));
    assert_eq!(parsed.frequency_penalty, Some(0.1));
    assert_eq!(parsed.seed, Some(123));
}

#[test]
fn generate_options_backwards_compatible() {
    // existing fields still work
    let opts = GenerateOptions::new("model")
        .max_tokens(100)
        .temperature(0.7)
        .top_p(0.9)
        .stop_sequence("END");

    assert_eq!(opts.max_tokens, Some(100));
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.top_p, Some(0.9));
    assert_eq!(opts.stop_sequences, vec!["END".to_string()]);

    // new fields default to None
    assert!(opts.top_k.is_none());
    assert!(opts.frequency_penalty.is_none());
    assert!(opts.presence_penalty.is_none());
    assert!(opts.seed.is_none());
    assert!(opts.reasoning.is_none());
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test generate_test`
Expected: compilation error — new fields don't exist

### Step 3: Add fields to `GenerateOptions` in `src/types/generate.rs`

Add fields after `stop_sequences`:
```rust
    /// Top-k sampling: only consider the k most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Penalise tokens based on frequency in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Penalise tokens based on whether they appear in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Seed for deterministic generation (where supported).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Reasoning configuration for extended thinking models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
```

Add `use crate::types::options::ReasoningConfig;` at the top of generate.rs.

Update `GenerateOptions::new()` to initialise new fields to `None`.

Add builder methods:
```rust
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn reasoning(mut self, config: ReasoningConfig) -> Self {
        self.reasoning = Some(config);
        self
    }
```

### Step 4: Update proto `GenerateOptions`

Add to `proto/ratatoskr.proto` `GenerateOptions` message:
```proto
    optional uint32 top_k = 6;
    optional float frequency_penalty = 7;
    optional float presence_penalty = 8;
    optional uint64 seed = 9;
    optional ReasoningConfig reasoning = 10;
```

### Step 5: Update conversions

In `From<proto::GenerateOptions> for GenerateOptions`:
```rust
    top_k: p.top_k.map(|k| k as usize),
    frequency_penalty: p.frequency_penalty,
    presence_penalty: p.presence_penalty,
    seed: p.seed,
    reasoning: p.reasoning.map(Into::into),
```

In `From<GenerateOptions> for proto::GenerateOptions`:
```rust
    top_k: o.top_k.map(|k| k as u32),
    frequency_penalty: o.frequency_penalty,
    presence_penalty: o.presence_penalty,
    seed: o.seed,
    reasoning: o.reasoning.map(|r| proto::ReasoningConfig {
        effort: r.effort.map(|e| match e {
            ReasoningEffort::Low => proto::ReasoningEffort::Low as i32,
            ReasoningEffort::Medium => proto::ReasoningEffort::Medium as i32,
            ReasoningEffort::High => proto::ReasoningEffort::High as i32,
        }),
        max_tokens: r.max_tokens.map(|t| t as u32),
        exclude_from_output: r.exclude_from_output,
    }),
```

### Step 6: Update `LlmChatProvider::generate()` in `src/providers/llm_chat.rs`

The `generate()` method currently only passes temperature and max_tokens to the chat options. Update it to also pass `top_p`, the new fields where the llm crate supports them. At minimum, ensure the new fields are present on the struct even if the llm crate ignores them:

```rust
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        let mut chat_options = ChatOptions::default().model(&options.model);
        if let Some(temp) = options.temperature {
            chat_options = chat_options.temperature(temp);
        }
        if let Some(max) = options.max_tokens {
            chat_options = chat_options.max_tokens(max);
        }
        if let Some(p) = options.top_p {
            chat_options = chat_options.top_p(p);
        }
        // top_k, frequency_penalty, presence_penalty, seed, reasoning
        // are not yet passed to llm crate — tracked for future provider updates
        // ... rest unchanged ...
```

### Step 7: Run tests

Run: `cargo test --test generate_test`
Expected: all PASS

### Step 8: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 9: Commit

```bash
git add src/types/generate.rs proto/ratatoskr.proto src/server/convert.rs src/providers/llm_chat.rs tests/generate_test.rs
git commit -m "feat: bring GenerateOptions to parity with ChatOptions

Add top_k, frequency_penalty, presence_penalty, seed, reasoning
to GenerateOptions. Proto and conversions updated."
```

---

## Task 5: raw_provider_options in Proto ChatOptions

**Files:**
- Modify: `proto/ratatoskr.proto`
- Modify: `src/server/convert.rs`

### Step 1: Add proto field

In `proto/ratatoskr.proto` `ChatOptions` message:
```proto
    // Escape hatch: provider-specific options as JSON string.
    optional string raw_provider_options = 14;
```

### Step 2: Update conversions

In `From<proto::ChatOptions> for ChatOptions`, change:
```rust
    raw_provider_options: None,
```
to:
```rust
    raw_provider_options: p.raw_provider_options
        .and_then(|s| serde_json::from_str(&s).ok()),
```

In `From<ChatOptions> for proto::ChatOptions`, add:
```rust
    raw_provider_options: o.raw_provider_options
        .and_then(|v| serde_json::to_string(&v).ok()),
```

### Step 3: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 4: Commit

```bash
git add proto/ratatoskr.proto src/server/convert.rs
git commit -m "feat: add raw_provider_options to proto ChatOptions

Escape hatch is now plumbed through gRPC — ServiceClient can
forward provider-specific JSON options."
```

---

## Task 6: Provider Parameter Declaration

**Files:**
- Modify: `src/providers/traits.rs`
- Modify: `src/providers/llm_chat.rs`
- Test: `tests/traits_test.rs`

### Step 1: Write failing test

Add to `tests/traits_test.rs`:

```rust
use ratatoskr::ParameterName;

// Note: this test verifies that the default implementation returns empty
// and that LlmChatProvider declares its supported params.
// Testing the actual trait method requires importing the provider traits.
```

For now, add an integration test that verifies LlmChatProvider declares params:

```rust
// tests/provider_params_test.rs
use ratatoskr::providers::traits::ChatProvider;
use ratatoskr::providers::LlmChatProvider;
use ratatoskr::ParameterName;
use llm::builder::LLMBackend;

#[test]
fn llm_chat_provider_declares_parameters() {
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "key", "openrouter");
    let params = ChatProvider::supported_chat_parameters(&provider);
    assert!(params.contains(&ParameterName::Temperature));
    assert!(params.contains(&ParameterName::MaxTokens));
    assert!(params.contains(&ParameterName::TopP));
    assert!(params.contains(&ParameterName::Reasoning));
    // top_k not yet supported by llm crate
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test provider_params_test`
Expected: compilation error — method doesn't exist

### Step 3: Add default method to provider traits

In `src/providers/traits.rs`, add to `ChatProvider`:
```rust
    /// Parameters this provider supports for chat requests.
    ///
    /// Providers that override this enable parameter validation in the registry.
    /// Default: empty (legacy behaviour — no validation).
    fn supported_chat_parameters(&self) -> Vec<ParameterName> {
        vec![]
    }
```

Add to `GenerateProvider`:
```rust
    /// Parameters this provider supports for generate requests.
    fn supported_generate_parameters(&self) -> Vec<ParameterName> {
        vec![]
    }
```

Add `use crate::ParameterName;` to the imports.

### Step 4: Implement for LlmChatProvider

In `src/providers/llm_chat.rs`, in the `ChatProvider` impl:
```rust
    fn supported_chat_parameters(&self) -> Vec<ParameterName> {
        vec![
            ParameterName::Temperature,
            ParameterName::MaxTokens,
            ParameterName::TopP,
            ParameterName::Reasoning,
            ParameterName::Stop,
        ]
    }
```

In the `GenerateProvider` impl:
```rust
    fn supported_generate_parameters(&self) -> Vec<ParameterName> {
        vec![
            ParameterName::Temperature,
            ParameterName::MaxTokens,
            ParameterName::TopP,
            ParameterName::Stop,
        ]
    }
```

### Step 5: Run tests

Run: `cargo test --test provider_params_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/providers/traits.rs src/providers/llm_chat.rs tests/provider_params_test.rs
git commit -m "feat: add supported_parameters() to provider traits

Opt-in parameter declaration enables validation. LlmChatProvider
declares the params the llm crate supports."
```

---

## Task 7: Model Registry Core

**Files:**
- Create: `src/registry/mod.rs`
- Modify: `src/lib.rs`
- Test: `tests/registry_test.rs`

### Step 1: Write failing tests

```rust
// tests/registry_test.rs
use std::collections::HashMap;
use ratatoskr::{
    ModelCapability, ModelInfo, ModelMetadata, ModelRegistry, ParameterAvailability,
    ParameterName, ParameterRange, PricingInfo,
};

#[test]
fn empty_registry() {
    let registry = ModelRegistry::new();
    assert!(registry.get("nonexistent").is_none());
    assert!(registry.list().is_empty());
}

#[test]
fn insert_and_get() {
    let mut registry = ModelRegistry::new();
    let metadata = ModelMetadata::from_info(
        ModelInfo::new("claude-sonnet-4", "openrouter")
            .with_capability(ModelCapability::Chat)
            .with_context_window(200_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(1.0),
        },
    );

    registry.insert(metadata.clone());

    let found = registry.get("claude-sonnet-4").unwrap();
    assert_eq!(found.info.id, "claude-sonnet-4");
    assert_eq!(found.info.context_window, Some(200_000));
}

#[test]
fn list_returns_all() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(ModelInfo::new("a", "p")));
    registry.insert(ModelMetadata::from_info(ModelInfo::new("b", "p")));
    registry.insert(ModelMetadata::from_info(ModelInfo::new("c", "p")));

    let list = registry.list();
    assert_eq!(list.len(), 3);
}

#[test]
fn merge_updates_existing() {
    let mut registry = ModelRegistry::new();

    // base entry (from embedded seed)
    let base = ModelMetadata::from_info(
        ModelInfo::new("gpt-4o", "openrouter").with_context_window(128_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(1.0),
        },
    )
    .with_max_output_tokens(4096);
    registry.insert(base);

    // live data (overrides)
    let live = ModelMetadata::from_info(
        ModelInfo::new("gpt-4o", "openrouter").with_context_window(128_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(2.0).default(1.0),
        },
    )
    .with_max_output_tokens(16384);

    registry.merge(live);

    let result = registry.get("gpt-4o").unwrap();
    // max_output_tokens updated
    assert_eq!(result.max_output_tokens, Some(16384));
    // temperature range updated
    assert!(matches!(
        &result.parameters[&ParameterName::Temperature],
        ParameterAvailability::Mutable { range } if range.max == Some(2.0)
    ));
}

#[test]
fn merge_preserves_non_overridden_params() {
    let mut registry = ModelRegistry::new();

    let base = ModelMetadata::from_info(ModelInfo::new("m", "p"))
        .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported)
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable { range: ParameterRange::default() },
        );
    registry.insert(base);

    // live data only knows about temperature
    let live = ModelMetadata::from_info(ModelInfo::new("m", "p"))
        .with_parameter(
            ParameterName::Temperature,
            ParameterAvailability::Mutable {
                range: ParameterRange::new().max(2.0),
            },
        );
    registry.merge(live);

    let result = registry.get("m").unwrap();
    // top_k preserved from base
    assert!(result.parameters.contains_key(&ParameterName::TopK));
    // temperature updated
    assert_eq!(result.parameters.len(), 2);
}

#[test]
fn merge_batch() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(ModelInfo::new("a", "p")));

    let batch = vec![
        ModelMetadata::from_info(ModelInfo::new("a", "p")).with_max_output_tokens(100),
        ModelMetadata::from_info(ModelInfo::new("b", "p")).with_max_output_tokens(200),
    ];

    registry.merge_batch(batch);

    assert_eq!(registry.get("a").unwrap().max_output_tokens, Some(100));
    assert_eq!(registry.get("b").unwrap().max_output_tokens, Some(200));
}

#[test]
fn filter_by_capability() {
    let mut registry = ModelRegistry::new();
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("chat-model", "p").with_capability(ModelCapability::Chat),
    ));
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("embed-model", "p").with_capability(ModelCapability::Embed),
    ));
    registry.insert(ModelMetadata::from_info(
        ModelInfo::new("multi-model", "p")
            .with_capability(ModelCapability::Chat)
            .with_capability(ModelCapability::Embed),
    ));

    let chat_models = registry.filter_by_capability(ModelCapability::Chat);
    assert_eq!(chat_models.len(), 2);
    assert!(chat_models.iter().any(|m| m.info.id == "chat-model"));
    assert!(chat_models.iter().any(|m| m.info.id == "multi-model"));
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test registry_test`
Expected: compilation error

### Step 3: Implement ModelRegistry

```rust
// src/registry/mod.rs
//! Model registry — centralised model metadata with layered merge.
//!
//! The registry holds [`ModelMetadata`] entries from multiple sources:
//! 1. **Embedded seed** — compiled-in JSON, always available
//! 2. **Live provider data** — runtime API queries
//! 3. **Remote curated data** — future, see issue #6
//!
//! Merge priority: later data overrides earlier (live > embedded).
//! Within a merge, per-parameter overrides are applied (not whole-entry replacement).

use std::collections::HashMap;

use crate::{ModelCapability, ModelMetadata, ParameterAvailability, ParameterName};

/// Centralised model metadata registry.
///
/// Thread-safe reads are ensured by storing data in a `HashMap` keyed by model ID.
/// For concurrent access (e.g., from `EmbeddedGateway`), wrap in `Arc<RwLock<>>`.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a model entry, replacing any existing entry with the same ID.
    pub fn insert(&mut self, metadata: ModelMetadata) {
        self.entries.insert(metadata.info.id.clone(), metadata);
    }

    /// Get metadata for a model by ID.
    pub fn get(&self, model: &str) -> Option<&ModelMetadata> {
        self.entries.get(model)
    }

    /// List all model metadata entries.
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.entries.values().collect()
    }

    /// Merge a single entry: update existing parameters or insert new.
    ///
    /// If the model already exists, parameters from `incoming` override the
    /// existing entry's parameters (per-key), and scalar fields (pricing,
    /// max_output_tokens, context_window) are replaced if present in `incoming`.
    /// If the model doesn't exist, it's inserted directly.
    pub fn merge(&mut self, incoming: ModelMetadata) {
        let id = incoming.info.id.clone();
        if let Some(existing) = self.entries.get_mut(&id) {
            // merge parameters (incoming overrides existing per-key)
            for (name, avail) in incoming.parameters {
                existing.parameters.insert(name, avail);
            }
            // scalar overrides
            if incoming.pricing.is_some() {
                existing.pricing = incoming.pricing;
            }
            if incoming.max_output_tokens.is_some() {
                existing.max_output_tokens = incoming.max_output_tokens;
            }
            if incoming.info.context_window.is_some() {
                existing.info.context_window = incoming.info.context_window;
            }
            // merge capabilities (union)
            for cap in incoming.info.capabilities {
                if !existing.info.capabilities.contains(&cap) {
                    existing.info.capabilities.push(cap);
                }
            }
        } else {
            self.entries.insert(id, incoming);
        }
    }

    /// Merge a batch of entries.
    pub fn merge_batch(&mut self, batch: Vec<ModelMetadata>) {
        for entry in batch {
            self.merge(entry);
        }
    }

    /// Filter entries by capability.
    pub fn filter_by_capability(&self, capability: ModelCapability) -> Vec<&ModelMetadata> {
        self.entries
            .values()
            .filter(|m| m.info.capabilities.contains(&capability))
            .collect()
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
```

### Step 4: Wire up module

Add to `src/lib.rs`:
```rust
pub mod registry;
pub use registry::ModelRegistry;
```

### Step 5: Run tests

Run: `cargo test --test registry_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/registry/mod.rs src/lib.rs tests/registry_test.rs
git commit -m "feat: add ModelRegistry with layered merge

Centralised model metadata store with per-parameter merge strategy.
Supports insert, merge, batch merge, and capability filtering."
```

---

## Task 8: Embedded JSON Seed File

**Files:**
- Create: `src/registry/seed.json`
- Modify: `src/registry/mod.rs`
- Test: `tests/registry_test.rs` (extend)

### Step 1: Write failing test

Add to `tests/registry_test.rs`:

```rust
#[test]
fn load_embedded_seed() {
    let registry = ModelRegistry::with_embedded_seed();
    // seed should contain at least a few well-known models
    assert!(!registry.is_empty());

    // check a known model exists
    let claude = registry.get("anthropic/claude-sonnet-4");
    assert!(claude.is_some(), "embedded seed should contain claude-sonnet-4");
    let claude = claude.unwrap();
    assert!(claude.info.capabilities.contains(&ModelCapability::Chat));
    assert!(claude.info.context_window.is_some());
}

#[test]
fn embedded_seed_has_parameter_metadata() {
    let registry = ModelRegistry::with_embedded_seed();
    let claude = registry.get("anthropic/claude-sonnet-4").unwrap();

    // should declare at least temperature
    assert!(claude.parameters.contains_key(&ParameterName::Temperature));
    assert!(claude.parameters[&ParameterName::Temperature].is_supported());
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test registry_test -- load_embedded`
Expected: compilation error — `with_embedded_seed` doesn't exist

### Step 3: Create seed file

Create `src/registry/seed.json` with a representative set of popular models. The format is an array of `ModelMetadata` entries. Start with a handful — this is the fallback, not the source of truth:

```json
[
  {
    "info": {
      "id": "anthropic/claude-sonnet-4",
      "provider": "openrouter",
      "capabilities": ["Chat"],
      "context_window": 200000,
      "dimensions": null
    },
    "parameters": {
      "temperature": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0, "default": 1.0 } },
      "top_p": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0, "default": 1.0 } },
      "top_k": { "availability": "mutable", "range": { "min": 1, "max": null, "default": null } },
      "max_tokens": { "availability": "mutable", "range": { "min": 1, "max": 16384, "default": null } },
      "stop": { "availability": "mutable", "range": {} },
      "reasoning": { "availability": "mutable", "range": {} }
    },
    "pricing": { "prompt_cost_per_mtok": 3.0, "completion_cost_per_mtok": 15.0 },
    "max_output_tokens": 16384
  },
  {
    "info": {
      "id": "anthropic/claude-haiku-3.5",
      "provider": "openrouter",
      "capabilities": ["Chat"],
      "context_window": 200000,
      "dimensions": null
    },
    "parameters": {
      "temperature": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0, "default": 1.0 } },
      "top_p": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0 } },
      "top_k": { "availability": "mutable", "range": { "min": 1 } },
      "max_tokens": { "availability": "mutable", "range": { "min": 1, "max": 8192 } }
    },
    "pricing": { "prompt_cost_per_mtok": 0.80, "completion_cost_per_mtok": 4.0 },
    "max_output_tokens": 8192
  },
  {
    "info": {
      "id": "openai/gpt-4o",
      "provider": "openrouter",
      "capabilities": ["Chat"],
      "context_window": 128000,
      "dimensions": null
    },
    "parameters": {
      "temperature": { "availability": "mutable", "range": { "min": 0.0, "max": 2.0, "default": 1.0 } },
      "top_p": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0 } },
      "frequency_penalty": { "availability": "mutable", "range": { "min": -2.0, "max": 2.0, "default": 0.0 } },
      "presence_penalty": { "availability": "mutable", "range": { "min": -2.0, "max": 2.0, "default": 0.0 } },
      "max_tokens": { "availability": "mutable", "range": { "min": 1, "max": 16384 } },
      "seed": { "availability": "mutable", "range": {} }
    },
    "pricing": { "prompt_cost_per_mtok": 2.50, "completion_cost_per_mtok": 10.0 },
    "max_output_tokens": 16384
  },
  {
    "info": {
      "id": "google/gemini-2.0-flash-001",
      "provider": "openrouter",
      "capabilities": ["Chat"],
      "context_window": 1048576,
      "dimensions": null
    },
    "parameters": {
      "temperature": { "availability": "mutable", "range": { "min": 0.0, "max": 2.0, "default": 1.0 } },
      "top_p": { "availability": "mutable", "range": { "min": 0.0, "max": 1.0 } },
      "top_k": { "availability": "mutable", "range": { "min": 1 } },
      "max_tokens": { "availability": "mutable", "range": { "min": 1, "max": 8192 } }
    },
    "pricing": { "prompt_cost_per_mtok": 0.10, "completion_cost_per_mtok": 0.40 },
    "max_output_tokens": 8192
  },
  {
    "info": {
      "id": "sentence-transformers/all-MiniLM-L6-v2",
      "provider": "huggingface",
      "capabilities": ["Embed"],
      "context_window": 512,
      "dimensions": 384
    },
    "parameters": {},
    "pricing": null,
    "max_output_tokens": null
  }
]
```

### Step 4: Implement `with_embedded_seed()`

Add to `src/registry/mod.rs`:

```rust
/// Raw JSON seed data compiled into the binary.
const EMBEDDED_SEED: &str = include_str!("seed.json");

impl ModelRegistry {
    // ... existing methods ...

    /// Create a registry pre-populated with the embedded seed data.
    ///
    /// The seed contains a curated set of well-known models with parameter
    /// metadata, pricing, and capabilities. It's always available as a
    /// fallback when live provider APIs are unreachable.
    pub fn with_embedded_seed() -> Self {
        let mut registry = Self::new();
        match serde_json::from_str::<Vec<ModelMetadata>>(EMBEDDED_SEED) {
            Ok(entries) => {
                for entry in entries {
                    registry.insert(entry);
                }
            }
            Err(e) => {
                // This should never happen — seed is compiled in and tested.
                // Log the error but don't panic; an empty registry is usable.
                eprintln!("warning: failed to parse embedded model seed: {e}");
            }
        }
        registry
    }
}
```

### Step 5: Run tests

Run: `cargo test --test registry_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/registry/seed.json src/registry/mod.rs tests/registry_test.rs
git commit -m "feat: add embedded JSON seed to model registry

Compiled-in fallback with curated model metadata for popular models.
JSON format matches future remote update channel (#6)."
```

---

## Task 9: model_metadata() on ModelGateway Trait

**Files:**
- Modify: `src/traits.rs`
- Test: `tests/traits_test.rs`

### Step 1: Write failing test

Add to `tests/traits_test.rs`:

```rust
use ratatoskr::{ModelMetadata, ParameterName};

// The default implementation returns None.
// Full integration is tested through EmbeddedGateway in task 10.
```

### Step 2: Add trait method

In `src/traits.rs`, add import for `ModelMetadata` and add to `ModelGateway` trait:

```rust
    // ===== Phase 6: Model intelligence =====

    /// Get extended metadata for a model, including parameter availability.
    ///
    /// Returns `None` if the model is not known to the registry.
    fn model_metadata(&self, _model: &str) -> Option<ModelMetadata> {
        None
    }
```

### Step 3: Run full suite

Run: `just pre-push`
Expected: all PASS (default implementation returns None, no implementors break)

### Step 4: Commit

```bash
git add src/traits.rs
git commit -m "feat: add model_metadata() to ModelGateway trait

Default returns None. EmbeddedGateway and ServiceClient will
override in the next task."
```

---

## Task 10: EmbeddedGateway + ServiceClient Integration

**Files:**
- Modify: `src/gateway/embedded.rs`
- Modify: `src/gateway/builder.rs`
- Modify: `src/client/service_client.rs`
- Modify: `src/server/service.rs`
- Test: `tests/gateway_test.rs`

### Step 1: Write failing test

Add to `tests/gateway_test.rs` (check existing content first):

```rust
use ratatoskr::{ModelGateway, ModelMetadata, ParameterName, Ratatoskr};

#[test]
fn embedded_gateway_model_metadata() {
    let gateway = Ratatoskr::builder()
        .openrouter("fake-key")
        .build()
        .unwrap();

    // the embedded seed should provide metadata
    let metadata = gateway.model_metadata("anthropic/claude-sonnet-4");
    assert!(metadata.is_some(), "should find claude-sonnet-4 in registry");

    let metadata = metadata.unwrap();
    assert!(metadata.parameters.contains_key(&ParameterName::Temperature));
}

#[test]
fn embedded_gateway_model_metadata_unknown() {
    let gateway = Ratatoskr::builder()
        .openrouter("fake-key")
        .build()
        .unwrap();

    assert!(gateway.model_metadata("totally-fake-model-xyz").is_none());
}
```

### Step 2: Run test to verify failure

Run: `cargo test --test gateway_test -- model_metadata`
Expected: returns `None` for everything (default implementation)

### Step 3: Add registry to EmbeddedGateway

In `src/gateway/embedded.rs`, add field:
```rust
use crate::registry::ModelRegistry;

pub struct EmbeddedGateway {
    registry: ProviderRegistry,
    model_registry: ModelRegistry,
    // ... existing fields ...
}
```

Update constructor to accept and store it. Update builder in `src/gateway/builder.rs` to create `ModelRegistry::with_embedded_seed()` and pass it.

Implement `model_metadata()`:
```rust
    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        self.model_registry.get(model).cloned()
    }
```

### Step 4: Update ServiceClient

In `src/client/service_client.rs`, implement `model_metadata()`:
```rust
    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        // For now, ServiceClient doesn't have a local registry.
        // The ModelMetadata RPC (task 11) will enable this.
        None
    }
```

(This is a temporary stub — task 11 adds the proper gRPC path.)

### Step 5: Run tests

Run: `cargo test --test gateway_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/gateway/embedded.rs src/gateway/builder.rs src/client/service_client.rs tests/gateway_test.rs
git commit -m "feat: integrate ModelRegistry into EmbeddedGateway

model_metadata() returns registry data from embedded seed.
ServiceClient stub (pending ModelMetadata RPC in next task)."
```

---

## Task 11: Proto ModelMetadata RPC + Messages

**Files:**
- Modify: `proto/ratatoskr.proto`
- Modify: `src/server/convert.rs`
- Modify: `src/server/service.rs`
- Modify: `src/client/service_client.rs`
- Test: `tests/service_test.rs` (extend)

### Step 1: Add proto messages and RPC

In `proto/ratatoskr.proto`:

Add new RPC to service:
```proto
    // Model metadata (Phase 6)
    rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse);
```

Add new messages in Model Management section:
```proto
message ModelMetadataRequest {
    string model = 1;
}

message ModelMetadataResponse {
    bool found = 1;
    optional ProtoModelMetadata metadata = 2;
}

message ProtoModelMetadata {
    ModelInfo info = 1;
    repeated ParameterEntry parameters = 2;
    optional ProtoPricingInfo pricing = 3;
    optional uint32 max_output_tokens = 4;
}

message ParameterEntry {
    string name = 1;
    ProtoParameterAvailability availability = 2;
}

message ProtoParameterAvailability {
    oneof kind {
        ProtoParameterRange mutable = 1;
        string read_only_json = 2;  // JSON-encoded value
        bool opaque = 3;
        bool unsupported = 4;
    }
}

message ProtoParameterRange {
    optional double min = 1;
    optional double max = 2;
    optional double default = 3;
}

message ProtoPricingInfo {
    optional double prompt_cost_per_mtok = 1;
    optional double completion_cost_per_mtok = 2;
}
```

### Step 2: Add conversions

In `src/server/convert.rs`, add bidirectional conversions for the new proto types.

Native → Proto (for server responses and client deserialization):
```rust
impl From<ModelMetadata> for proto::ProtoModelMetadata { ... }
impl From<ParameterRange> for proto::ProtoParameterRange { ... }
impl From<PricingInfo> for proto::ProtoPricingInfo { ... }
```

Proto → Native (for client responses and server deserialization):
```rust
impl From<proto::ProtoModelMetadata> for ModelMetadata { ... }
impl From<proto::ProtoParameterRange> for ParameterRange { ... }
impl From<proto::ProtoPricingInfo> for PricingInfo { ... }
```

The `ParameterEntry` and `ProtoParameterAvailability` conversions require special handling for the HashMap ↔ repeated message mapping and the oneof ↔ enum mapping.

### Step 3: Add service handler

In `src/server/service.rs`:
```rust
    async fn model_metadata(
        &self,
        request: Request<proto::ModelMetadataRequest>,
    ) -> GrpcResult<proto::ModelMetadataResponse> {
        let req = request.into_inner();
        match self.gateway.model_metadata(&req.model) {
            Some(metadata) => Ok(Response::new(proto::ModelMetadataResponse {
                found: true,
                metadata: Some(metadata.into()),
            })),
            None => Ok(Response::new(proto::ModelMetadataResponse {
                found: false,
                metadata: None,
            })),
        }
    }
```

### Step 4: Update ServiceClient

In `src/client/service_client.rs`, replace the stub:
```rust
    fn model_metadata(&self, model: &str) -> Option<ModelMetadata> {
        let rt = tokio::runtime::Handle::try_current().ok()?;
        let request = proto::ModelMetadataRequest {
            model: model.to_string(),
        };
        let mut client = self.inner.clone();
        let response = rt.block_on(async { client.model_metadata(request).await }).ok()?;
        let resp = response.into_inner();
        if resp.found {
            resp.metadata.map(Into::into)
        } else {
            None
        }
    }
```

### Step 5: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 6: Commit

```bash
git add proto/ratatoskr.proto src/server/convert.rs src/server/service.rs src/client/service_client.rs
git commit -m "feat: add ModelMetadata gRPC RPC

New proto messages for parameter availability and pricing.
ServiceClient now queries the server for model metadata."
```

---

## Task 12: Parameter Validation in ProviderRegistry

**Files:**
- Modify: `src/providers/registry.rs`
- Modify: `src/error.rs`
- Test: `tests/validation_test.rs`

### Step 1: Write failing test

```rust
// tests/validation_test.rs
use ratatoskr::{ChatOptions, ParameterValidationPolicy, RatatoskrError};

// Testing validation requires a mock provider — see step 3 for details.
// These tests verify the policy enum and error variant exist.

#[test]
fn validation_policy_variants() {
    let _ = ParameterValidationPolicy::Warn;
    let _ = ParameterValidationPolicy::Error;
    let _ = ParameterValidationPolicy::Ignore;
}
```

### Step 2: Add error variant

In `src/error.rs`:
```rust
    #[error("unsupported parameter '{param}' for model '{model}' (provider: {provider})")]
    UnsupportedParameter {
        param: String,
        model: String,
        provider: String,
    },
```

### Step 3: Add validation policy type

Create `src/types/validation.rs` or add to `src/providers/registry.rs`:

```rust
/// How to handle unsupported parameters in requests.
#[derive(Debug, Clone, Copy, Default)]
pub enum ParameterValidationPolicy {
    /// Log a warning, proceed without the parameter.
    #[default]
    Warn,
    /// Return an error.
    Error,
    /// Silently ignore (legacy behaviour).
    Ignore,
}
```

### Step 4: Add validation to registry dispatch

In `src/providers/registry.rs`, add a `validation_policy` field and check parameters against `supported_chat_parameters()` before dispatching chat/generate calls. Only validate when the provider declares parameters (non-empty return from `supported_chat_parameters()`).

### Step 5: Run tests

Run: `cargo test --test validation_test`
Expected: all PASS

### Step 6: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 7: Commit

```bash
git add src/error.rs src/providers/registry.rs tests/validation_test.rs
git commit -m "feat: add parameter validation in ProviderRegistry

Opt-in validation: providers that declare supported_parameters()
get checked. Policy: Warn (default), Error, or Ignore."
```

---

## Task 13: Update Error Re-exports

**Files:**
- Modify: `src/lib.rs`

Ensure `UnsupportedParameter` is accessible and `ParameterValidationPolicy` is re-exported.

### Step 1: Update lib.rs

Add to re-exports as needed.

### Step 2: Run full suite

Run: `just pre-push`
Expected: all PASS

### Step 3: Commit

```bash
git add src/lib.rs
git commit -m "chore: re-export new phase 6 types from crate root"
```

---

## Task 14: Update Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/ROADMAP.md` (mark phase 6 complete)

### Step 1: Update AGENTS.md

Add new types to "Key Types" section:
- `ModelMetadata` — extended model info with parameter availability, pricing
- `ModelRegistry` — centralised model metadata with layered merge
- `ParameterName` — hybrid enum for well-known + custom parameter names
- `ParameterAvailability` — mutable/read-only/opaque/unsupported per parameter
- `ParameterValidationPolicy` — warn/error/ignore for unsupported params

Update project structure to include `src/registry/`.

Update `ModelGateway` trait description to include `model_metadata()`.

### Step 2: Update ROADMAP.md

Mark Phase 6 as ✓ in the overview. Add a "See" link to this plan document.

### Step 3: Run full suite one last time

Run: `just pre-push`
Expected: all PASS

### Step 4: Commit

```bash
git add AGENTS.md docs/plans/ROADMAP.md
git commit -m "docs: update AGENTS.md and ROADMAP.md for phase 6"
```
