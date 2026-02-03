# Ratatoskr Phase 3-4: Local Inference & Token Counting

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** 🟡 IN PROGRESS (11/15 tasks complete)

**Goal:** Add local inference capabilities (embeddings via fastembed-rs, NLI via ONNX) and token counting, enabling örlög to run high-volume operations without API costs.

**Architecture:** Hybrid approach — `fastembed-rs` for batteries-included embeddings, raw `ort` for NLI cross-encoder models. Token counting via HuggingFace `tokenizers` crate with extensible provider pattern.

**Tech Stack:** Rust, fastembed, ort, tokenizers, hf-hub

**Progress:**
- ✅ Tasks 1-6: Core infrastructure (dependencies, tokenizer, types, device, fastembed)
- ✅ Tasks 7-9: ONNX NLI, model manager, capabilities
- ✅ Tasks 10-11: Builder integration, gateway updates
- ⏳ Task 12: Generate impl (via llm crate)
- ⏳ Tasks 13-15: Live tests, lint, documentation

**Latest Commit:** `0a17cff` - feat(local-inference): add phase 3 foundation (tasks 1-6)

---

## 1. Architecture Overview

This phase adds local inference capabilities via two new provider types:

```
                          ModelGateway trait
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   EmbeddedGateway         (future)              (future)
         │                ServiceClient        RemoteOnnxProvider
         │
         ├── llm crate (chat providers)
         │     └── OpenRouter, Anthropic, OpenAI, Ollama, Google
         │
         ├── HuggingFaceClient (API-based)
         │     └── embed, NLI, classify via HTTP
         │
         ├── FastEmbedProvider (new)
         │     └── local embeddings via fastembed-rs
         │
         ├── OnnxNliProvider (new)
         │     └── local NLI via ort + cross-encoder models
         │
         └── TokenizerRegistry (new)
               └── token counting via tokenizers crate
```

The router gains new provider options:

```toml
[routing]
embed = "fastembed"      # local via fastembed-rs
nli = "onnx"             # local via ort
classify = "huggingface" # still API-based
chat = "openrouter"      # via llm crate
```

---

## 2. New Dependencies

```toml
[features]
# existing
huggingface = ["dep:reqwest"]

# new
local-inference = ["dep:fastembed", "dep:ort", "dep:tokenizers", "dep:hf-hub"]
cuda = ["ort/cuda"]  # optional GPU support

[dependencies]
# Local inference (optional)
fastembed = { version = "4", optional = true }
ort = { version = "2", optional = true, default-features = false }
tokenizers = { version = "0.20", optional = true, default-features = false, features = ["http"] }
hf-hub = { version = "0.3", optional = true }

# For blocking ops in async context
tokio = { version = "1", features = ["rt", "sync"] }
```

Key points:
- All local inference behind `local-inference` feature flag
- `cuda` is separate feature for GPU support
- `tokenizers` with `http` feature fetches from HuggingFace hub
- `hf-hub` handles model downloads with caching

---

## 3. Token Counting System

```rust
// src/tokenizer/mod.rs

/// Trait for tokenizer implementations, allowing future expansion
pub trait TokenizerProvider: Send + Sync {
    fn count_tokens(&self, text: &str) -> Result<usize>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
}

/// HuggingFace tokenizers implementation
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    /// Load from HuggingFace hub
    pub fn from_hub(repo_id: &str) -> Result<Self>;

    /// Load from local file
    pub fn from_file(path: &Path) -> Result<Self>;
}

/// Registry mapping model names to tokenizers
pub struct TokenizerRegistry {
    tokenizers: RwLock<HashMap<String, Arc<dyn TokenizerProvider>>>,
    model_mappings: HashMap<String, TokenizerSource>,
}

pub enum TokenizerSource {
    HuggingFace { repo_id: String },
    Local { path: PathBuf },
    Alias { target: String },  // "claude-sonnet-4" → "claude"
}

impl TokenizerRegistry {
    pub fn new() -> Self;

    /// Register a model → tokenizer mapping
    pub fn register(&mut self, model_pattern: &str, source: TokenizerSource);

    /// Count tokens, loading tokenizer lazily if needed
    pub fn count_tokens(&self, text: &str, model: &str) -> Result<usize>;
}
```

Default mappings (built-in):

| Model Pattern | Tokenizer Source |
|---------------|------------------|
| `claude*` | `Xenova/claude-tokenizer` |
| `gpt-4*`, `gpt-3.5*` | `Xenova/gpt-4` |
| `llama*`, `meta-llama/*` | `meta-llama/Llama-3.2-1B` |
| `mistral*` | `mistralai/Mistral-7B-v0.1` |

Users can override or add mappings via builder:

```rust
Ratatoskr::builder()
    .openrouter(key)
    .tokenizer_mapping("my-custom-model", TokenizerSource::Local {
        path: "/models/custom-tokenizer.json".into()
    })
    .build()
```

---

## 4. FastEmbed Provider (Local Embeddings)

```rust
// src/providers/fastembed.rs

pub struct FastEmbedProvider {
    model: fastembed::TextEmbedding,
    model_info: EmbeddingModelInfo,
}

pub struct EmbeddingModelInfo {
    pub name: String,
    pub dimensions: usize,
}

/// Supported embedding models (maps to fastembed's enum)
pub enum LocalEmbeddingModel {
    AllMiniLmL6V2,       // 384 dims, fast, good quality
    AllMiniLmL12V2,      // 384 dims, slightly better
    BgeSmallEn,          // 384 dims, strong retrieval
    BgeBaseEn,           // 768 dims, higher quality
    // ... expose what fastembed supports
}

impl FastEmbedProvider {
    pub fn new(model: LocalEmbeddingModel) -> Result<Self> {
        let options = fastembed::InitOptions::new(model.into())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir()?);

        let model = fastembed::TextEmbedding::try_new(options)?;
        Ok(Self { model, model_info: model.into() })
    }

    pub fn embed(&self, text: &str) -> Result<Embedding> {
        let vectors = self.model.embed(vec![text], None)?;
        Ok(Embedding {
            values: vectors.into_iter().next().unwrap(),
            model: self.model_info.name.clone(),
            dimensions: self.model_info.dimensions,
        })
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let vectors = self.model.embed(texts.to_vec(), None)?;
        // ... map to Embedding structs
    }
}
```

Note: `fastembed` is synchronous — wrap calls in `spawn_blocking` for async context.

---

## 5. ONNX NLI Provider (Local Cross-Encoder)

```rust
// src/providers/onnx_nli.rs

pub struct OnnxNliProvider {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    model_info: NliModelInfo,
    device: Device,
}

pub struct NliModelInfo {
    pub name: String,
    pub labels: Vec<String>,  // ["entailment", "neutral", "contradiction"]
}

pub enum Device {
    Cpu,
    Cuda { device_id: u32 },
}

/// Supported NLI models
pub enum LocalNliModel {
    /// cross-encoder/nli-deberta-v3-base — good balance of speed/accuracy
    NliDebertaV3Base,
    /// cross-encoder/nli-deberta-v3-small — faster, slightly less accurate
    NliDebertaV3Small,
    /// Custom model from path
    Custom { model_path: PathBuf, tokenizer_path: PathBuf },
}

impl OnnxNliProvider {
    pub fn new(model: LocalNliModel, device: Device) -> Result<Self> {
        let (model_path, tokenizer_path) = model.resolve_paths()?;

        let session = ort::Session::builder()?
            .with_execution_providers([execution_provider(&device)?])?
            .commit_from_file(&model_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)?;

        Ok(Self { session, tokenizer, model_info: model.into(), device })
    }

    pub fn infer_nli(&self, premise: &str, hypothesis: &str) -> Result<NliResult> {
        let encoding = self.tokenizer.encode((premise, hypothesis), true)?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ]?)?;

        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let probs = softmax(&logits);

        Ok(NliResult {
            entailment: probs[0],
            neutral: probs[1],
            contradiction: probs[2],
            label: NliLabel::from_max(&probs),
        })
    }

    pub fn infer_nli_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliResult>> {
        // Batch encode and run inference — more efficient than N calls
    }
}

fn execution_provider(device: &Device) -> Result<ort::ExecutionProvider> {
    match device {
        Device::Cpu => Ok(ort::CPUExecutionProvider::default().build()),
        Device::Cuda { device_id } => Ok(ort::CUDAExecutionProvider::default()
            .with_device_id(*device_id)
            .build()),
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}
```

---

## 6. Model Management

```rust
// src/model/manager.rs

pub struct ModelManager {
    embedding_models: RwLock<HashMap<String, Arc<FastEmbedProvider>>>,
    nli_models: RwLock<HashMap<String, Arc<OnnxNliProvider>>>,
    config: ModelManagerConfig,
}

pub struct ModelManagerConfig {
    pub cache_dir: PathBuf,           // ~/.cache/ratatoskr/models/
    pub default_device: Device,
    // Future Phase 5 additions:
    // pub memory_budget_bytes: Option<u64>,
    // pub idle_timeout: Option<Duration>,
}

pub enum ModelSource {
    HuggingFace { repo_id: String },
    Local { path: PathBuf },
}

impl ModelManager {
    pub fn new(config: ModelManagerConfig) -> Self;

    /// Get or lazily load an embedding model
    pub fn embedding(&self, model: LocalEmbeddingModel) -> Result<Arc<FastEmbedProvider>> {
        let key = model.cache_key();

        // Fast path: already loaded
        if let Some(provider) = self.embedding_models.read().get(&key) {
            return Ok(Arc::clone(provider));
        }

        // Slow path: load and cache
        let mut models = self.embedding_models.write();
        // Double-check after acquiring write lock
        if let Some(provider) = models.get(&key) {
            return Ok(Arc::clone(provider));
        }

        let provider = Arc::new(FastEmbedProvider::new(model)?);
        models.insert(key, Arc::clone(&provider));
        Ok(provider)
    }

    /// Get or lazily load an NLI model
    pub fn nli(&self, model: LocalNliModel) -> Result<Arc<OnnxNliProvider>>;

    /// Explicitly preload models (for latency-sensitive deployments)
    pub fn preload_embedding(&self, model: LocalEmbeddingModel) -> Result<()>;
    pub fn preload_nli(&self, model: LocalNliModel) -> Result<()>;

    /// Unload a model (for future memory management)
    pub fn unload_embedding(&self, model: &str) -> bool;
    pub fn unload_nli(&self, model: &str) -> bool;

    /// List currently loaded models
    pub fn loaded_models(&self) -> LoadedModels;
}

pub struct LoadedModels {
    pub embeddings: Vec<String>,
    pub nli: Vec<String>,
}
```

Double-checked locking ensures thread-safe lazy loading.

---

## 7. Trait Additions

```rust
// src/traits.rs additions

#[async_trait]
pub trait ModelGateway: Send + Sync {
    // ===== Existing methods =====

    async fn chat_stream(...) -> Result<...>;
    async fn chat(...) -> Result<ChatResponse>;
    fn capabilities(&self) -> Capabilities;
    async fn embed(&self, text: &str, model: &str) -> Result<Embedding>;
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>>;
    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>;
    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult>;
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize>;

    // ===== New methods =====

    /// Batch NLI inference — more efficient for multiple pairs
    async fn infer_nli_batch(
        &self,
        pairs: &[(&str, &str)],
        model: &str,
    ) -> Result<Vec<NliResult>> {
        // Default: sequential fallback
        let mut results = Vec::with_capacity(pairs.len());
        for (premise, hypothesis) in pairs {
            results.push(self.infer_nli(premise, hypothesis, model).await?);
        }
        Ok(results)
    }

    /// Non-streaming text generation
    async fn generate(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<GenerateResponse> {
        Err(RatatoskrError::NotImplemented("generate"))
    }

    /// Streaming text generation
    async fn generate_stream(
        &self,
        _prompt: &str,
        _options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        Err(RatatoskrError::NotImplemented("generate_stream"))
    }
}
```

```rust
// src/types/generate.rs (new file)

pub struct GenerateOptions {
    pub model: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

pub struct GenerateResponse {
    pub text: String,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub finish_reason: FinishReason,
}

pub enum GenerateEvent {
    Text(String),
    Done,
}
```

---

## 8. Updated Builder & Routing

```rust
// src/gateway/builder.rs additions

impl RatatoskrBuilder {
    // ===== Existing =====
    pub fn openrouter(mut self, api_key: impl Into<String>) -> Self;
    pub fn anthropic(mut self, api_key: impl Into<String>) -> Self;
    pub fn huggingface(mut self, api_key: impl Into<String>) -> Self;

    // ===== New: Local inference =====

    /// Enable local embeddings via fastembed
    #[cfg(feature = "local-inference")]
    pub fn local_embeddings(mut self, model: LocalEmbeddingModel) -> Self;

    /// Enable local NLI via ONNX
    #[cfg(feature = "local-inference")]
    pub fn local_nli(mut self, model: LocalNliModel) -> Self;

    /// Set device for local inference (default: CPU)
    #[cfg(feature = "local-inference")]
    pub fn device(mut self, device: Device) -> Self;

    /// Custom tokenizer mapping
    pub fn tokenizer_mapping(
        mut self,
        model_pattern: impl Into<String>,
        source: TokenizerSource,
    ) -> Self;

    /// Set model cache directory (default: ~/.cache/ratatoskr/models/)
    #[cfg(feature = "local-inference")]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self;
}
```

```rust
// src/gateway/routing.rs additions

pub enum EmbedProvider {
    HuggingFace,
    #[cfg(feature = "local-inference")]
    FastEmbed,
}

pub enum NliProvider {
    HuggingFace,
    #[cfg(feature = "local-inference")]
    Onnx,
}
```

Usage:

```rust
// API-only (lightweight)
let gateway = Ratatoskr::builder()
    .openrouter(api_key)
    .huggingface(hf_key)
    .build()?;

// With local inference
let gateway = Ratatoskr::builder()
    .openrouter(api_key)
    .local_embeddings(LocalEmbeddingModel::AllMiniLmL6V2)
    .local_nli(LocalNliModel::NliDebertaV3Base)
    .device(Device::Cuda { device_id: 0 })
    .build()?;
```

---

## 9. Updated Capabilities

```rust
// src/types/capabilities.rs additions

pub struct Capabilities {
    // Existing
    pub chat: bool,
    pub streaming: bool,
    pub tool_use: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub classification: bool,

    // New
    pub generate: bool,
    pub token_counting: bool,
    pub local_inference: bool,
}

impl Capabilities {
    /// Local inference capabilities (embeddings + NLI, no API needed)
    pub fn local_only() -> Self {
        Self {
            embeddings: true,
            nli: true,
            token_counting: true,
            local_inference: true,
            ..Default::default()
        }
    }

    /// Full capabilities (chat + local + API)
    pub fn full() -> Self {
        Self {
            chat: true,
            streaming: true,
            tool_use: true,
            embeddings: true,
            nli: true,
            classification: true,
            generate: true,
            token_counting: true,
            local_inference: true,
        }
    }
}
```

---

## 10. Project Structure

```
src/
├── lib.rs                      # Public API re-exports
├── error.rs                    # RatatoskrError, Result
├── traits.rs                   # ModelGateway trait (updated)
│
├── types/
│   ├── mod.rs
│   ├── message.rs
│   ├── options.rs
│   ├── response.rs
│   ├── tool.rs
│   ├── capabilities.rs         # Updated with new flags
│   ├── future.rs               # Embedding, NliResult, ClassifyResult
│   └── generate.rs             # NEW: GenerateOptions, GenerateResponse
│
├── gateway/
│   ├── mod.rs
│   ├── embedded.rs             # Updated to use new providers
│   ├── builder.rs              # Updated with local inference config
│   └── routing.rs              # Updated with new provider variants
│
├── providers/
│   ├── mod.rs
│   ├── huggingface.rs          # Existing API provider
│   ├── fastembed.rs            # NEW: local embeddings
│   └── onnx_nli.rs             # NEW: local NLI
│
├── tokenizer/                   # NEW module
│   ├── mod.rs                  # TokenizerProvider trait, registry
│   └── hf.rs                   # HuggingFace tokenizers impl
│
├── model/                       # NEW module
│   ├── mod.rs
│   ├── manager.rs              # ModelManager, lazy loading
│   ├── source.rs               # ModelSource, download logic
│   └── device.rs               # Device enum, execution providers
│
└── convert/
    └── mod.rs                  # ratatoskr ↔ llm conversions

tests/
├── # existing tests...
├── tokenizer_test.rs           # NEW
├── fastembed_test.rs           # NEW
├── onnx_nli_test.rs            # NEW
└── local_inference_live_test.rs # NEW (ignored, manual)
```

---

## Implementation Tasks

### ✅ Task 1: Add Dependencies (COMPLETED)
- ✅ Update `Cargo.toml` with new features and dependencies
- ✅ Added: fastembed v5, ort v2.0.0-rc, tokenizers v0.22, hf-hub v0.4, dirs v5
- ✅ Features: local-inference, cuda
- ✅ Verify: `cargo check --features local-inference`

### ✅ Task 2: Token Counting Module (COMPLETED)
- ✅ Create `src/tokenizer/mod.rs` with `TokenizerProvider` trait
- ✅ Create `src/tokenizer/hf.rs` with `HfTokenizer` implementation
- ✅ Create `TokenizerRegistry` with default mappings (claude, gpt-4, llama, mistral)
- ✅ Lazy loading with double-checked locking
- ✅ Alias support with recursive resolution
- ✅ Create `tests/tokenizer_test.rs` (5 tests passing)
- ✅ Added Configuration and DataError to RatatoskrError
- ✅ Verify: `cargo test --test tokenizer_test --features local-inference`

### ✅ Task 3: Generate Types (COMPLETED)
- ✅ Create `src/types/generate.rs` with `GenerateOptions`, `GenerateResponse`, `GenerateEvent`
- ✅ Builder pattern for GenerateOptions
- ✅ Update `src/types/mod.rs` exports
- ✅ Update `src/lib.rs` exports
- ✅ Verify: `cargo check`

### ✅ Task 4: Trait Additions (COMPLETED)
- ✅ Add `infer_nli_batch()` with default impl to `ModelGateway`
- ✅ Add `generate()` and `generate_stream()` stubs to `ModelGateway`
- ✅ Update `tests/traits_test.rs` (5 tests passing)
- ✅ Verify: `cargo test --test traits_test`

### ✅ Task 5: Device & Model Source Types (COMPLETED)
- ✅ Create `src/model/mod.rs`
- ✅ Create `src/model/device.rs` with `Device` enum (Cpu, Cuda)
- ✅ Create `src/model/source.rs` with `ModelSource` enum and download logic
- ✅ Create `src/model/manager.rs` stub (ModelManager, ModelManagerConfig, LoadedModels)
- ✅ Note: execution_provider helper deferred to Task 7
- ✅ Verify: `cargo check --features local-inference`

### ✅ Task 6: FastEmbed Provider (COMPLETED)
- ✅ Create `src/providers/fastembed.rs`
- ✅ Add `LocalEmbeddingModel` enum (4 variants: AllMiniLmL6V2, AllMiniLmL12V2, BgeSmallEn, BgeBaseEn)
- ✅ Implement `FastEmbedProvider` with embed/embed_batch (requires &mut self)
- ✅ Model properties: name(), dimensions(), cache_key()
- ✅ EmbeddingModelInfo struct
- ✅ Create `tests/fastembed_test.rs` (1 test passing)
- ✅ Verify: `cargo test --test fastembed_test --features local-inference`

**Commit:** `0a17cff` - feat(local-inference): add phase 3 foundation (tasks 1-6)

### ✅ Task 7: ONNX NLI Provider (COMPLETED)
- ✅ Create `src/providers/onnx_nli.rs`
- ✅ Add `LocalNliModel` enum (NliDebertaV3Base, NliDebertaV3Small, Custom)
- ✅ Implement `OnnxNliProvider` with infer_nli/infer_nli_batch
- ✅ HuggingFace Hub integration for model downloads
- ✅ Create `tests/onnx_nli_test.rs` (3 unit tests + 2 ignored live tests)
- ✅ Verify: `cargo test --test onnx_nli_test --features local-inference`

### ✅ Task 8: Model Manager (COMPLETED)
- ✅ Create `src/model/manager.rs`
- ✅ Implement lazy loading with double-checked locking
- ✅ Add preload/unload methods
- ✅ Thread-safe Arc<RwLock<>> wrapping for providers
- ✅ Create `tests/model_manager_test.rs` (5 unit tests + 2 ignored live tests)
- ✅ Verify: `cargo test --test model_manager_test --features local-inference`

### ✅ Task 9: Update Capabilities (COMPLETED)
- ✅ Add new fields to `Capabilities` struct (generate, tool_use, local_inference)
- ✅ Add `local_only()` constructor
- ✅ Update `full()` constructor
- ✅ Update merge() to include new fields
- ✅ Update `capabilities_test.rs` (now 9 tests)
- ✅ Verify: `cargo test --test capabilities_test`

### ✅ Task 10: Update Builder (COMPLETED)
- ✅ Add `local_embeddings()`, `local_nli()`, `device()`, `cache_dir()` methods
- ✅ Add `tokenizer_mapping()` method
- ✅ Update routing enums (EmbedProvider::FastEmbed, NliProvider::Onnx)
- ✅ Update `gateway_test.rs` (now 6 tests)
- ✅ Verify: `cargo test --test gateway_test --features local-inference`

### ✅ Task 11: Update EmbeddedGateway (COMPLETED)
- ✅ Integrate `ModelManager` and `TokenizerRegistry` into struct
- ✅ Implement `count_tokens()` using registry
- ✅ Update `capabilities()` to reflect local inference
- ✅ Builder constructs router with local providers and passes to gateway
- ⏸️ Actual routing of embed/NLI calls to local providers (deferred — infrastructure ready)
- ✅ Verify: `cargo test --features local-inference`

### Task 12: Implement generate() via llm crate
- Add `generate()` and `generate_stream()` implementations
- Create `tests/generate_test.rs`
- Verify: `cargo test --test generate_test`

### Task 13: Live Tests
- Create `tests/local_inference_live_test.rs` with `#[ignore]` tests
- Test local embeddings with real model
- Test local NLI with real model
- Test token counting
- Verify: `cargo test --test local_inference_live_test --features local-inference -- --ignored`

### Task 14: Full Test Suite & Lint
- Verify: `just pre-push`

### Task 15: Update Documentation
- Update `CLAUDE.md` with Phase 3-4 status
- Update `src/lib.rs` doc examples
- Update architecture appendix
- Verify: `cargo doc --no-deps --features local-inference`
