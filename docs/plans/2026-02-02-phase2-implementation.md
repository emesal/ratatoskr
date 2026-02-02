# Ratatoskr Phase 2 Implementation Plan: HuggingFace Provider

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HuggingFace Inference API support for embeddings, NLI, and zero-shot classification via a new provider system with capability-based routing.

**Architecture:** Introduce `src/providers/` module with `HuggingFaceClient` that handles direct HTTP calls to HuggingFace's Inference API. Add `CapabilityRouter` to `src/gateway/routing.rs` that determines which provider handles each capability. The builder configures routing implicitly when `.huggingface(key)` is called.

**Tech Stack:** Rust, reqwest (rustls-tls), serde_json, wiremock (dev)

---

## Progress Summary

### COMPLETED (Tasks 1-6)

The following has been implemented and committed:

1. **Task 1: Add reqwest dependency** - DONE
   - Added `huggingface` feature with reqwest (rustls-tls)
   - Added wiremock to dev-dependencies
   - Commit: `feat: add huggingface feature with reqwest dependency`

2. **Task 2: Create providers module structure** - DONE
   - Created `src/providers/mod.rs`
   - Updated `src/lib.rs` with conditional module
   - Commit: `feat: providers module structure (WIP, missing huggingface)`

3. **Tasks 3-5 COMBINED: Full HuggingFaceClient** - DONE
   - Created `src/providers/huggingface.rs` with ALL methods:
     - `embed()` - single text embeddings
     - `embed_batch()` - batch embeddings
     - `infer_nli()` - natural language inference
     - `classify()` - zero-shot classification
     - `handle_response_errors()` - 401, 404, 429, 503 handling
   - Created `tests/huggingface_test.rs` with parsing tests
   - Commit: `feat: HuggingFaceClient with embed, nli, and classify methods`

4. **Task 4: Create Capability Router** - DONE
   - Created `src/gateway/routing.rs` with `EmbedProvider`, `NliProvider`, `ClassifyProvider` enums
   - Created `CapabilityRouter` struct with `with_huggingface()` builder method
   - Created `tests/routing_test.rs` with 2 passing tests
   - Updated `src/gateway/mod.rs` to export routing module

5. **Task 5: Update Builder with HuggingFace Support** - DONE
   - Added `#[cfg(feature = "huggingface")]` gated `huggingface_key` field to `RatatoskrBuilder`
   - Added `huggingface()` method to builder
   - Added `has_chat_provider()` and `has_capability_provider()` helper methods
   - Updated `build()` to create `HuggingFaceClient` and `CapabilityRouter`
   - Updated `tests/gateway_test.rs` with 2 new huggingface tests

6. **Task 6: Update EmbeddedGateway with HuggingFace Integration** - DONE
   - Added `huggingface: Option<HuggingFaceClient>` and `router: CapabilityRouter` fields
   - Updated `new()` constructor to accept new parameters
   - Added `has_chat_provider()` helper method
   - Updated `capabilities()` to reflect HF capabilities via router
   - Implemented `embed()`, `embed_batch()`, `infer_nli()`, `classify_zero_shot()` methods that delegate to `HuggingFaceClient`

**Current state:**
- `cargo test --test routing_test` PASSES (2 tests)
- `cargo test --test gateway_test --features huggingface` PASSES (5 tests)
- `cargo check --features huggingface` PASSES
- `cargo check` (without feature) PASSES

---

## REMAINING TASKS (Start here in new session)

### Task 7: Wiremock Integration Tests

**Files:** Create `tests/huggingface_integration_test.rs`

Write wiremock integration tests for:
- `test_embed_success` - mock 200 response with embedding vector
- `test_embed_batch_success` - mock 200 response with batch embeddings  
- `test_classify_success` - mock 200 response with classification result
- `test_error_401_unauthorized` - mock 401, expect `AuthenticationFailed`
- `test_error_404_model_not_found` - mock 404, expect `ModelNotFound`
- `test_error_429_rate_limited` - mock 429 with retry-after header, expect `RateLimited`
- `test_error_503_model_loading` - mock 503, expect `Api { status: 503 }`

Use `HuggingFaceClient::with_base_url("test_key", mock_server.uri())` to point at wiremock.

**Verify:** `cargo test --test huggingface_integration_test --features huggingface`

---

### Task 8: Live Integration Tests

**Files:** Create `tests/huggingface_live_test.rs`

Write `#[ignore]` tests that hit the real HuggingFace API:
- `test_live_embed` - embed text with `sentence-transformers/all-MiniLM-L6-v2`
- `test_live_embed_batch` - batch embed 2 texts
- `test_live_classify` - classify sentiment with `facebook/bart-large-mnli`
- `test_live_nli` - NLI inference with premise/hypothesis

All tests use `std::env::var("HF_API_KEY")` and are `#[ignore]` by default.

**Verify:** `HF_API_KEY=hf_xxx cargo test --test huggingface_live_test --features huggingface -- --ignored`

---

### Task 9: Update Capabilities Helper Methods

**Files:** Modify `src/types/capabilities.rs`, `tests/capabilities_test.rs`

Add to `Capabilities`:
- `pub fn huggingface_only() -> Self` - embeddings, nli, classification = true
- `pub fn merge(&self, other: &Self) -> Self` - OR logic for all fields

**Verify:** `cargo test --test capabilities_test`

---

### Task 10: Update lib.rs Documentation

**Files:** Modify `src/lib.rs`

Update module doc example to show:
- `.huggingface("hf_your_key")` in builder
- Example `gateway.embed()` call

**Verify:** `cargo doc --no-deps --features huggingface`

---

### Task 11: Run Full Test Suite and Lint

**Files:** None (verification only)

**Verify:** `just pre-push` - must pass

---

### Task 12: Update CLAUDE.md Documentation

**Files:** Modify `CLAUDE.md`

Add Phase 2 status section documenting:
- New files created
- New capabilities (embed, embed_batch, infer_nli, classify_zero_shot)
- Live testing instructions

---

## Summary

**Completed:** Tasks 1-6 (dependencies, providers, routing, builder, gateway integration)

**Remaining:** Tasks 7-12 (wiremock tests, live tests, capabilities helpers, docs)
