# Capabilities Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `Capabilities` bool-field struct with a `Capabilities(HashSet<ModelCapability>)` newtype, eliminating the dual-representation problem and its bridge boilerplate.

**Architecture:** `Capabilities` becomes a newtype over `HashSet<ModelCapability>`. All call sites switch from field access (`caps.chat`) to method calls (`caps.has(ModelCapability::Chat)`). The proto `CapabilitiesResponse` adopts the existing `ModelCapability` enum (extended with 4 missing variants) and drops the 10 bool fields.

**Tech Stack:** Rust, prost (protobuf), `std::collections::HashSet`

**Design doc:** `docs/plans/2026-02-20-capabilities-unification-design.md`

---

### Task 1: Extend proto `ModelCapability` enum and replace `CapabilitiesResponse`

**Files:**
- Modify: `proto/ratatoskr.proto`

The existing `ModelCapability` proto enum has 6 variants. Add the 4 missing ones and update `CapabilitiesResponse` to use `repeated ModelCapability` instead of 10 bool fields.

**Step 1: Edit the proto file**

In `proto/ratatoskr.proto`, find the `ModelCapability` enum (around line 381) and replace it:

```proto
enum ModelCapability {
    MODEL_CAPABILITY_UNSPECIFIED = 0;
    MODEL_CAPABILITY_CHAT = 1;
    MODEL_CAPABILITY_GENERATE = 2;
    MODEL_CAPABILITY_EMBED = 3;
    MODEL_CAPABILITY_NLI = 4;
    MODEL_CAPABILITY_CLASSIFY = 5;
    MODEL_CAPABILITY_STANCE = 6;
    MODEL_CAPABILITY_CHAT_STREAMING = 7;
    MODEL_CAPABILITY_TOOL_USE = 8;
    MODEL_CAPABILITY_TOKEN_COUNTING = 9;
    MODEL_CAPABILITY_LOCAL_INFERENCE = 10;
}
```

Then find `CapabilitiesResponse` (around line 505) and replace the entire message:

```proto
message CapabilitiesResponse {
    repeated ModelCapability capabilities = 1;
}
```

**Step 2: Verify proto compiles**

```bash
cargo build --features server,client 2>&1 | head -40
```

Expected: compile errors in `service.rs` and `service_client.rs` referencing the old bool fields — this is expected and will be fixed in later tasks.

**Step 3: Commit proto changes**

```bash
git add proto/ratatoskr.proto
git commit -m "proto: extend ModelCapability enum, replace CapabilitiesResponse bool fields with repeated enum"
```

---

### Task 2: Rewrite `src/types/capabilities.rs` as a newtype

**Files:**
- Modify: `src/types/capabilities.rs`

**Step 1: Replace the file contents**

```rust
//! Gateway capability reporting

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::model::ModelCapability;

/// What capabilities a gateway supports.
///
/// A typed set of [`ModelCapability`] values. Use [`Capabilities::has`] to query
/// individual capabilities and the factory methods for common configurations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities(HashSet<ModelCapability>);

impl Capabilities {
    /// Returns `true` if this set contains the given capability.
    pub fn has(&self, cap: ModelCapability) -> bool {
        self.0.contains(&cap)
    }

    /// Inserts a capability into the set.
    pub fn insert(&mut self, cap: ModelCapability) {
        self.0.insert(cap);
    }

    /// Returns an iterator over the capabilities in this set.
    pub fn iter(&self) -> impl Iterator<Item = &ModelCapability> {
        self.0.iter()
    }

    /// Merges two capability sets using union (OR) logic.
    pub fn merge(&self, other: &Self) -> Self {
        Self(self.0.union(&other.0).copied().collect())
    }

    /// Chat-capable gateway (chat, streaming, generate, tool use).
    pub fn chat_only() -> Self {
        Self::from_iter([
            ModelCapability::Chat,
            ModelCapability::ChatStreaming,
            ModelCapability::Generate,
            ModelCapability::ToolUse,
        ])
    }

    /// All capabilities enabled.
    pub fn full() -> Self {
        Self::from_iter([
            ModelCapability::Chat,
            ModelCapability::ChatStreaming,
            ModelCapability::Generate,
            ModelCapability::ToolUse,
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::Classify,
            ModelCapability::Stance,
            ModelCapability::TokenCounting,
            ModelCapability::LocalInference,
        ])
    }

    /// Local inference only (no API calls needed).
    pub fn local_only() -> Self {
        Self::from_iter([
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::TokenCounting,
            ModelCapability::LocalInference,
        ])
    }

    /// HuggingFace API capabilities (embeddings, NLI, classification).
    pub fn huggingface_only() -> Self {
        Self::from_iter([
            ModelCapability::Embed,
            ModelCapability::Nli,
            ModelCapability::Classify,
        ])
    }
}

impl FromIterator<ModelCapability> for Capabilities {
    fn from_iter<I: IntoIterator<Item = ModelCapability>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<HashSet<ModelCapability>> for Capabilities {
    fn from(set: HashSet<ModelCapability>) -> Self {
        Self(set)
    }
}

impl From<Capabilities> for HashSet<ModelCapability> {
    fn from(caps: Capabilities) -> Self {
        caps.0
    }
}
```

**Step 2: Build (expect failures in gateway, service, client)**

```bash
cargo build 2>&1 | grep "^error" | head -30
```

Note the files and line numbers with errors — they will all be fixed in the next tasks.

**Step 3: Commit**

```bash
git add src/types/capabilities.rs
git commit -m "feat: rewrite Capabilities as HashSet<ModelCapability> newtype"
```

---

### Task 3: Update `src/gateway/embedded.rs`

**Files:**
- Modify: `src/gateway/embedded.rs:227-261`

**Step 1: Find the current `capabilities()` method**

It's `impl ModelGateway for EmbeddedGateway`. The body builds a `Capabilities { chat: ..., ... }` struct literal.

**Step 2: Replace the method body**

```rust
fn capabilities(&self) -> Capabilities {
    use crate::ModelCapability::*;

    let mut caps = vec![
        (self.registry.has_chat(), Chat),
        (self.registry.has_chat(), ChatStreaming),
        (self.registry.has_generate(), Generate),
        (self.registry.has_chat(), ToolUse),
        (self.registry.has_embedding(), Embed),
        (self.registry.has_nli(), Nli),
        (self.registry.has_classify(), Classify),
        (self.registry.has_stance(), Stance),
    ];

    #[cfg(feature = "local-inference")]
    {
        let names = self.registry.provider_names();
        let has_local = names
            .embedding
            .iter()
            .any(|n| n.starts_with("local-") || n.contains("fastembed"))
            || names
                .nli
                .iter()
                .any(|n| n.starts_with("local-") || n.contains("onnx"));
        caps.push((true, TokenCounting));
        caps.push((has_local, LocalInference));
    }

    caps.into_iter()
        .filter(|(enabled, _)| *enabled)
        .map(|(_, cap)| cap)
        .collect()
}
```

**Step 3: Build to check this file is clean**

```bash
cargo build 2>&1 | grep "embedded" | head -20
```

Expected: no errors mentioning `embedded.rs`.

**Step 4: Commit**

```bash
git add src/gateway/embedded.rs
git commit -m "feat: update EmbeddedGateway::capabilities() to build Capabilities set"
```

---

### Task 4: Update `src/server/service.rs`

**Files:**
- Modify: `src/server/service.rs` (around line 384)

**Step 1: Find and replace `get_capabilities`**

The current body manually maps each bool field to a `proto::CapabilitiesResponse` field. Replace it:

```rust
async fn get_capabilities(
    &self,
    _request: Request<proto::CapabilitiesRequest>,
) -> GrpcResult<proto::CapabilitiesResponse> {
    let caps = self.gateway.capabilities();
    let capabilities = caps
        .iter()
        .filter_map(|cap| crate::server::convert::model_capability_to_proto(*cap))
        .map(|c| c as i32)
        .collect();
    Ok(Response::new(proto::CapabilitiesResponse { capabilities }))
}
```

Note: this calls a helper `model_capability_to_proto` that we'll add in the convert task. For now just write the call — it won't compile until Task 5.

**Step 2: Commit (even if not fully compiling yet — we're working task by task)**

```bash
git add src/server/service.rs
git commit -m "feat: update get_capabilities RPC to use repeated ModelCapability"
```

---

### Task 5: Update `src/server/convert.rs`

**Files:**
- Modify: `src/server/convert.rs`

**Context:** `convert.rs` already has `ModelInfo → proto::ModelInfo` conversion using a `match c { ... }` on `ModelCapability`. That match currently handles 6 variants and has a `_ => None` fallthrough for the gateway-level ones (ChatStreaming, ToolUse, TokenCounting, LocalInference). We need to:

1. Add the 4 missing variants to the existing `ModelInfo` conversion match.
2. Extract a reusable `model_capability_to_proto` helper.
3. Add the reverse `proto::ModelCapability → ModelCapability` helper.

**Step 1: Add a pub(crate) helper for the forward direction**

Find the `impl From<ModelInfo> for proto::ModelInfo` block. Before it, add:

```rust
/// Maps a [`ModelCapability`] to its proto equivalent. Returns `None` for unknown variants.
pub(crate) fn model_capability_to_proto(cap: ModelCapability) -> Option<proto::ModelCapability> {
    match cap {
        ModelCapability::Chat => Some(proto::ModelCapability::Chat),
        ModelCapability::ChatStreaming => Some(proto::ModelCapability::ChatStreaming),
        ModelCapability::Generate => Some(proto::ModelCapability::Generate),
        ModelCapability::ToolUse => Some(proto::ModelCapability::ToolUse),
        ModelCapability::Embed => Some(proto::ModelCapability::Embed),
        ModelCapability::Nli => Some(proto::ModelCapability::Nli),
        ModelCapability::Classify => Some(proto::ModelCapability::Classify),
        ModelCapability::Stance => Some(proto::ModelCapability::Stance),
        ModelCapability::TokenCounting => Some(proto::ModelCapability::TokenCounting),
        ModelCapability::LocalInference => Some(proto::ModelCapability::LocalInference),
        _ => None,
    }
}

/// Maps a proto [`proto::ModelCapability`] to its native equivalent. Returns `None` for unspecified.
pub(crate) fn proto_to_model_capability(cap: proto::ModelCapability) -> Option<ModelCapability> {
    match cap {
        proto::ModelCapability::Chat => Some(ModelCapability::Chat),
        proto::ModelCapability::ChatStreaming => Some(ModelCapability::ChatStreaming),
        proto::ModelCapability::Generate => Some(ModelCapability::Generate),
        proto::ModelCapability::ToolUse => Some(ModelCapability::ToolUse),
        proto::ModelCapability::Embed => Some(ModelCapability::Embed),
        proto::ModelCapability::Nli => Some(ModelCapability::Nli),
        proto::ModelCapability::Classify => Some(ModelCapability::Classify),
        proto::ModelCapability::Stance => Some(ModelCapability::Stance),
        proto::ModelCapability::TokenCounting => Some(ModelCapability::TokenCounting),
        proto::ModelCapability::LocalInference => Some(ModelCapability::LocalInference),
        proto::ModelCapability::Unspecified => None,
    }
}
```

**Step 2: Update `From<ModelInfo> for proto::ModelInfo`**

Replace the `filter_map` closure body in the capabilities field (currently has `_ => None` catchall):

```rust
capabilities: m
    .capabilities
    .into_iter()
    .filter_map(|c| model_capability_to_proto(c).map(|p| p as i32))
    .collect(),
```

**Step 3: Update the reverse conversion (proto `ModelInfo` → native `ModelInfo`)**

Find the block around line 626 that maps `proto.capabilities` back to `Vec<ModelCapability>`. Replace its `filter_map` closure to use the helper:

```rust
capabilities: p
    .capabilities
    .into_iter()
    .filter_map(|c| {
        proto::ModelCapability::try_from(c)
            .ok()
            .and_then(proto_to_model_capability)
    })
    .collect(),
```

**Step 4: Build to verify convert.rs is clean**

```bash
cargo build --features server,client 2>&1 | grep "convert\|service\.rs" | head -20
```

**Step 5: Commit**

```bash
git add src/server/convert.rs
git commit -m "feat: add model_capability_to_proto/proto_to_model_capability helpers, extend to all 10 variants"
```

---

### Task 6: Update `src/client/service_client.rs`

**Files:**
- Modify: `src/client/service_client.rs:164-193`

**Step 1: Replace `capabilities()` method body**

The current body manually maps bool fields back to a `Capabilities` struct. Replace:

```rust
fn capabilities(&self) -> Capabilities {
    let rt = match tokio::runtime::Handle::try_current() {
        Ok(rt) => rt,
        Err(_) => return Capabilities::default(),
    };

    let mut client = self.inner.clone();
    match block_in_place(|| {
        rt.block_on(async { client.get_capabilities(proto::CapabilitiesRequest {}).await })
    }) {
        Ok(response) => {
            let caps = response.into_inner();
            caps.capabilities
                .into_iter()
                .filter_map(|c| {
                    proto::ModelCapability::try_from(c)
                        .ok()
                        .and_then(crate::server::convert::proto_to_model_capability)
                })
                .collect()
        }
        Err(_) => Capabilities::default(),
    }
}
```

**Step 2: Build to verify**

```bash
cargo build --features client 2>&1 | grep "service_client" | head -20
```

Expected: no errors.

**Step 3: Commit**

```bash
git add src/client/service_client.rs
git commit -m "feat: update ServiceClient::capabilities() to decode repeated ModelCapability"
```

---

### Task 7: Rewrite `tests/capabilities_test.rs`

**Files:**
- Modify: `tests/capabilities_test.rs`

**Step 1: Replace file contents**

```rust
use ratatoskr::{Capabilities, Embedding, ModelCapability, NliLabel, NliResult};

#[test]
fn test_capabilities_default() {
    let caps = Capabilities::default();
    assert!(!caps.has(ModelCapability::Chat));
    assert!(!caps.has(ModelCapability::Embed));
}

#[test]
fn test_capabilities_chat_only() {
    let caps = Capabilities::chat_only();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::ChatStreaming));
    assert!(caps.has(ModelCapability::Generate));
    assert!(caps.has(ModelCapability::ToolUse));
    assert!(!caps.has(ModelCapability::Embed));
    assert!(!caps.has(ModelCapability::LocalInference));
}

#[test]
fn test_embedding_stub() {
    let emb = Embedding {
        values: vec![0.1, 0.2, 0.3],
        model: "text-embedding-3".into(),
        dimensions: 3,
    };
    assert_eq!(emb.dimensions, 3);
}

#[test]
fn test_nli_result_stub() {
    let nli = NliResult {
        entailment: 0.8,
        contradiction: 0.1,
        neutral: 0.1,
        label: NliLabel::Entailment,
    };
    assert!(matches!(nli.label, NliLabel::Entailment));
}

#[test]
fn test_capabilities_huggingface_only() {
    let caps = Capabilities::huggingface_only();
    assert!(!caps.has(ModelCapability::Chat), "huggingface_only should not have chat");
    assert!(
        !caps.has(ModelCapability::ChatStreaming),
        "huggingface_only should not have chat_streaming"
    );
    assert!(caps.has(ModelCapability::Embed), "huggingface_only should have embeddings");
    assert!(caps.has(ModelCapability::Nli), "huggingface_only should have nli");
    assert!(caps.has(ModelCapability::Classify), "huggingface_only should have classification");
    assert!(
        !caps.has(ModelCapability::TokenCounting),
        "huggingface_only should not have token_counting"
    );
}

#[test]
fn test_capabilities_merge() {
    let chat = Capabilities::chat_only();
    let hf = Capabilities::huggingface_only();
    let merged = chat.merge(&hf);

    assert!(merged.has(ModelCapability::Chat), "merged should have chat");
    assert!(merged.has(ModelCapability::ChatStreaming), "merged should have chat_streaming");
    assert!(merged.has(ModelCapability::Embed), "merged should have embeddings");
    assert!(merged.has(ModelCapability::Nli), "merged should have nli");
    assert!(merged.has(ModelCapability::Classify), "merged should have classification");
    assert!(
        !merged.has(ModelCapability::TokenCounting),
        "merged should not have token_counting (neither source has it)"
    );
}

#[test]
fn test_capabilities_merge_is_symmetric() {
    let a = Capabilities::chat_only();
    let b = Capabilities::huggingface_only();

    let ab = a.merge(&b);
    let ba = b.merge(&a);

    // Merge should be symmetric
    assert_eq!(ab, ba);
}

#[test]
fn test_capabilities_local_only() {
    let caps = Capabilities::local_only();
    assert!(!caps.has(ModelCapability::Chat), "local_only should not have chat");
    assert!(
        !caps.has(ModelCapability::ChatStreaming),
        "local_only should not have chat_streaming"
    );
    assert!(!caps.has(ModelCapability::Generate), "local_only should not have generate");
    assert!(!caps.has(ModelCapability::ToolUse), "local_only should not have tool_use");
    assert!(caps.has(ModelCapability::Embed), "local_only should have embeddings");
    assert!(caps.has(ModelCapability::Nli), "local_only should have nli");
    assert!(!caps.has(ModelCapability::Classify), "local_only should not have classification");
    assert!(caps.has(ModelCapability::TokenCounting), "local_only should have token_counting");
    assert!(
        caps.has(ModelCapability::LocalInference),
        "local_only should have local_inference"
    );
}

#[test]
fn test_capabilities_full() {
    let caps = Capabilities::full();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::ChatStreaming));
    assert!(caps.has(ModelCapability::Generate));
    assert!(caps.has(ModelCapability::ToolUse));
    assert!(caps.has(ModelCapability::Embed));
    assert!(caps.has(ModelCapability::Nli));
    assert!(caps.has(ModelCapability::Classify));
    assert!(caps.has(ModelCapability::TokenCounting));
    assert!(caps.has(ModelCapability::LocalInference));
}

#[test]
fn test_capabilities_insert() {
    let mut caps = Capabilities::default();
    assert!(!caps.has(ModelCapability::Chat));
    caps.insert(ModelCapability::Chat);
    assert!(caps.has(ModelCapability::Chat));
}

#[test]
fn test_capabilities_from_iter() {
    let caps: Capabilities = [ModelCapability::Chat, ModelCapability::Embed]
        .into_iter()
        .collect();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::Embed));
    assert!(!caps.has(ModelCapability::Nli));
}

#[test]
fn test_capabilities_roundtrip_hashset() {
    use std::collections::HashSet;
    let original = Capabilities::chat_only();
    let set: HashSet<ModelCapability> = original.clone().into();
    let roundtripped = Capabilities::from(set);
    assert_eq!(original, roundtripped);
}
```

**Step 2: Run tests**

```bash
cargo test --test capabilities_test 2>&1
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/capabilities_test.rs
git commit -m "test: update capabilities_test to use has() API, add insert/from_iter/roundtrip tests"
```

---

### Task 8: Fix `tests/traits_test.rs` and `tests/gateway_test.rs`

**Files:**
- Modify: `tests/traits_test.rs`
- Modify: `tests/gateway_test.rs`

**Step 1: Fix `traits_test.rs`**

Find `impl ModelGateway for MockGateway`. The `capabilities()` method returns `Capabilities::default()` — this is already valid (default is now an empty set). Only the import needs updating if the test uses field access elsewhere.

Search for `caps.` in the file:

```bash
grep -n "caps\." tests/traits_test.rs
```

Replace any `caps.field_name` with `caps.has(ModelCapability::FieldVariant)`. Add `ModelCapability` to the import if needed.

**Step 2: Fix `gateway_test.rs`**

Search for `caps.` in the file:

```bash
grep -n "caps\." tests/gateway_test.rs
```

Typical patterns to replace:
- `assert!(caps.embed)` → `assert!(caps.has(ModelCapability::Embed))`
- `assert!(caps.chat)` → `assert!(caps.has(ModelCapability::Chat))`
- `caps.chat,` (used as a value) → `caps.has(ModelCapability::Chat),`

Add `ModelCapability` to the import: `use ratatoskr::{ModelGateway, ModelCapability, ParameterName, Ratatoskr};`

**Step 3: Run full test suite**

```bash
cargo test --all-targets 2>&1 | tail -30
```

Expected: all tests pass (or only pre-existing ignored tests skipped).

**Step 4: Commit**

```bash
git add tests/traits_test.rs tests/gateway_test.rs
git commit -m "test: update traits_test and gateway_test to use Capabilities::has() API"
```

---

### Task 9: Full pre-push check

**Step 1: Run the full check**

```bash
just pre-push
```

Expected: format clean, no clippy warnings, all tests pass.

**Step 2: Fix any remaining issues**

If clippy flags anything (e.g. unused `_ => None` match arms that are now exhaustive, or `allow` attributes that can be removed), fix them.

**Step 3: Final commit if needed**

```bash
git add -p
git commit -m "chore: clippy fixes post-capabilities-unification"
```

---

### Task 10: Close issue

```bash
gh issue close 20 --comment "implemented: Capabilities is now a HashSet<ModelCapability> newtype. Proto CapabilitiesResponse uses repeated ModelCapability. All callsites use caps.has(). Refs merged in dev."
```
