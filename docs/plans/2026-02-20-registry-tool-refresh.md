# Registry Tool: model refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `rat-registry model refresh` — re-fetches metadata from providers for every model in the registry file and merges results in-place, warning on failures.

**Architecture:** Add a `Refresh` variant to `ModelCommand` and a `model_refresh` async function that iterates registry models, calls `gateway.fetch_model_metadata()`, applies `ModelRegistry::merge()` semantics, and writes the result atomically. Uses the existing `build_gateway()`, `load_registry()`, and `save_registry()` helpers with no new files needed.

**Tech Stack:** Rust, clap (existing), `ratatoskr::{ModelGateway, ModelRegistry}`, tokio (existing).

---

### Task 1: Add `Refresh` variant to `ModelCommand` enum

**Files:**
- Modify: `src/bin/rat_registry.rs:46-59` (the `ModelCommand` enum)

**Step 1: Write the failing test**

There's no direct unit test for the CLI enum itself — instead verify compilation:

```bash
cargo build --bin rat-registry --features registry-tool 2>&1 | grep error
```

Expected: compilation errors about missing `Refresh` match arm (or clean compile if adding first).

**Step 2: Add the variant**

In `ModelCommand`, add after `Remove`:

```rust
/// re-fetch metadata for all models from providers and merge in-place
Refresh,
```

**Step 3: Add the match arm in `main`**

In the `match args.command` block in `main()`, add:

```rust
Command::Model(ModelCommand::Refresh) => model_refresh(path).await,
```

**Step 4: Verify it compiles (stub function)**

Add a temporary stub so it compiles:

```rust
async fn model_refresh(_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    todo!()
}
```

Run:

```bash
cargo build --bin rat-registry --features registry-tool 2>&1 | grep error
```

Expected: clean (no errors).

**Step 5: Commit**

```bash
git add src/bin/rat_registry.rs
git commit -m "feat(registry-tool): stub model refresh command"
```

---

### Task 2: Implement `model_refresh` function

**Files:**
- Modify: `src/bin/rat_registry.rs` (replace stub with real implementation)

**Step 1: Write the test first**

There is no integration test file for `rat_registry` binary commands today. Create one:

```bash
# check if tests/rat_registry_test.rs exists
ls tests/rat_registry*
```

If it doesn't exist, create `tests/rat_registry_test.rs`:

```rust
//! Integration tests for `rat-registry model refresh` logic.
//!
//! Tests the merge semantics: successful fetches update the registry,
//! failures are collected and reported, presets are preserved.

use std::collections::{BTreeMap, HashMap};
use ratatoskr::registry::remote::RemoteRegistry;
use ratatoskr::{ModelCapability, ModelInfo, ModelMetadata};

fn make_metadata(id: &str, context_window: Option<u64>) -> ModelMetadata {
    ModelMetadata {
        info: ModelInfo {
            id: id.to_string(),
            provider: "openrouter".to_string(),
            capabilities: vec![ModelCapability::Chat],
            context_window,
            dimensions: None,
        },
        parameters: HashMap::new(),
        pricing: None,
        max_output_tokens: None,
    }
}

fn make_registry(models: Vec<ModelMetadata>) -> RemoteRegistry {
    RemoteRegistry {
        version: 1,
        models,
        presets: BTreeMap::new(),
    }
}

/// Simulates the merge logic that `model_refresh` applies.
/// Verifies that a successful fetch updates `context_window`
/// while preserving models that had no update.
#[test]
fn merge_updates_existing_model_fields() {
    use ratatoskr::ModelRegistry;

    let original = make_metadata("provider/model-a", Some(4096));
    let mut registry = ModelRegistry::default();
    registry.merge(original);

    let updated = make_metadata("provider/model-a", Some(128000));
    registry.merge(updated);

    let got = registry.get("provider/model-a").unwrap();
    assert_eq!(got.info.context_window, Some(128000));
}

/// Verifies merge preserves hand-curated fields absent from fetched metadata.
#[test]
fn merge_preserves_absent_optional_fields() {
    use ratatoskr::ModelRegistry;

    let mut original = make_metadata("provider/model-b", Some(8192));
    original.pricing = Some(ratatoskr::PricingInfo {
        prompt_cost_per_mtok: Some(3.0),
        completion_cost_per_mtok: Some(15.0),
    });

    let mut reg = ModelRegistry::default();
    reg.merge(original);

    // Fetched metadata has no pricing (provider didn't return it)
    let fetched = make_metadata("provider/model-b", Some(16384));
    // fetched.pricing is None
    reg.merge(fetched);

    let got = reg.get("provider/model-b").unwrap();
    // pricing preserved from original since fetched had None
    assert!(got.pricing.is_some());
    assert_eq!(got.info.context_window, Some(16384));
}
```

**Step 2: Run the tests to verify they pass** (these test `ModelRegistry::merge`, not our new code yet — they serve as contract tests for the merge semantics we rely on):

```bash
cargo nextest run --features registry-tool -E 'test(merge_)'
```

Expected: 2 PASS.

**Step 3: Implement `model_refresh`**

Replace the stub with:

```rust
async fn model_refresh(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = load_registry(path)?;

    if registry.models.is_empty() {
        println!("no models in registry — nothing to refresh.");
        return Ok(());
    }

    let total = registry.models.len();
    println!("refreshing {total} models...");

    let gateway = build_gateway()?;

    // Collect model IDs to iterate (avoid borrow conflict with registry).
    let model_ids: Vec<String> = registry.models.iter().map(|m| m.info.id.clone()).collect();

    let mut successes = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new(); // (id, error)

    // Load existing models into a ModelRegistry for merge semantics.
    let mut model_reg = ratatoskr::ModelRegistry::default();
    for m in registry.models.drain(..) {
        model_reg.merge(m);
    }

    for model_id in &model_ids {
        print!("  fetching {model_id}... ");
        match gateway.fetch_model_metadata(model_id).await {
            Ok(fetched) => {
                model_reg.merge(fetched);
                successes += 1;
                println!("ok");
            }
            Err(e) => {
                println!("WARN: {e}");
                failures.push((model_id.clone(), e.to_string()));
            }
        }
    }

    if successes == 0 {
        eprintln!("no models refreshed — registry unchanged.");
        if !failures.is_empty() {
            eprintln!("\nfailed ({}):", failures.len());
            for (id, err) in &failures {
                eprintln!("  {id}: {err}");
            }
        }
        return Err("all fetches failed".into());
    }

    // Extract merged models back into sorted vec, preserving presets.
    let mut merged_models: Vec<ModelMetadata> = model_reg.list().into_iter().cloned().collect();
    merged_models.sort_by(|a, b| a.info.id.cmp(&b.info.id));
    registry.models = merged_models;

    save_registry(path, &registry)?;

    println!("\nrefreshed {successes}/{total} models.");
    if !failures.is_empty() {
        eprintln!("failed ({}):", failures.len());
        for (id, err) in &failures {
            eprintln!("  {id}: {err}");
        }
    }

    Ok(())
}
```

Note: this requires `ratatoskr::ModelRegistry` and `ratatoskr::ModelMetadata` to be in scope. Check the existing imports at the top of the file — `ModelMetadata` is not currently imported. Add it:

```rust
use ratatoskr::registry::remote::RemoteRegistry;
use ratatoskr::{EmbeddedGateway, ModelGateway, ModelMetadata, ModelRegistry, PricingInfo, Ratatoskr};
```

**Step 4: Run tests**

```bash
cargo nextest run --features registry-tool -E 'test(merge_)'
```

Expected: 2 PASS.

**Step 5: Build to verify compilation**

```bash
cargo build --bin rat-registry --features registry-tool 2>&1 | grep -E "^error"
```

Expected: no errors.

**Step 6: Commit**

```bash
git add src/bin/rat_registry.rs tests/rat_registry_test.rs
git commit -m "feat(registry-tool): implement model refresh command"
```

---

### Task 3: Verify `ModelRegistry` is exported from crate root

**Files:**
- Read: `src/lib.rs`

**Step 1: Check exports**

```bash
grep -n "ModelRegistry" src/lib.rs
```

If `ModelRegistry` is not in `pub use` at the crate root, the `use ratatoskr::ModelRegistry` import in `rat_registry.rs` will fail. Check and fix if needed.

**Step 2: If missing, add the export**

In `src/lib.rs`, find the existing `pub use registry::...` line and add `ModelRegistry`:

```rust
pub use registry::{ModelRegistry, ...existing items...};
```

**Step 3: Verify compilation**

```bash
cargo build --bin rat-registry --features registry-tool 2>&1 | grep -E "^error"
```

Expected: clean.

**Step 4: Commit if changed**

```bash
git add src/lib.rs
git commit -m "chore: re-export ModelRegistry from crate root"
```

---

### Task 4: Full pre-push check

**Step 1: Run the full test suite**

```bash
just pre-push
```

Expected: all tests pass, no clippy warnings, no fmt issues.

**Step 2: If clippy complains about `model_refresh`**

Common clippy issues:
- `drain(..)` may suggest `.into_iter()` on `Vec` — use whichever compiles cleanly
- unused variable warnings if `total` is only used in one branch — use `_total` or restructure

Fix and re-run.

**Step 3: Commit any fixes**

```bash
git add -p
git commit -m "chore: fix clippy warnings in model refresh"
```

---

### Task 5: Update design doc status

**Files:**
- Modify: `docs/plans/2026-02-20-registry-tool-refresh-design.md`

**Step 1: Update status line**

Change:
```
**status**: design
```
to:
```
**status**: implemented
```

**Step 2: Commit**

```bash
git add docs/plans/2026-02-20-registry-tool-refresh-design.md
git commit -m "docs: mark registry refresh design as implemented"
```
