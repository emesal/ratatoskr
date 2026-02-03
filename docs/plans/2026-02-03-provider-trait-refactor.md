# Provider Trait Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** 🚧 IN PROGRESS (Tasks 1-6 complete)

**Goal:** Introduce capability-specific provider traits to establish a clean foundation for phase 5 (service mode), phase 6 (caching/retry/telemetry decorators), and future extensibility.

**Motivation:** The current architecture has routing infrastructure that isn't wired up, and lacks the abstractions needed for decorator patterns. This refactor establishes proper provider traits that enable:
- Service mode protocol handlers to dispatch to providers uniformly
- Decorator patterns: `CachingProvider<T>`, `RetryingProvider<T>`, `InstrumentedProvider<T>`
- Fallback chains: try providers in priority order, transparent local→remote fallback
- RAM-aware routing: local providers signal unavailability when memory-constrained
- User-extensible providers (future)

**Key Design Decisions:**
1. **Capability-specific traits** (not one god-trait) — providers implement only what they support
2. **Model param passed through** — providers self-report availability via `ModelNotAvailable` error
3. **Fallback chain pattern** — registry tries providers in priority order
4. **RAM budget awareness** — `ModelManager` tracks memory, local providers consult it
5. **Separate `StanceProvider`** — dedicated trait for stance detection models

---

## 1. Architecture Overview

### Current State

```
EmbeddedGateway
├── llm crate (direct usage for chat)
├── HuggingFaceClient (direct field)
├── ModelManager (dead_code — not wired up!)
├── CapabilityRouter (routing enums, not used in dispatch)
└── TokenizerRegistry
```

### Target State

```
EmbeddedGateway
├── ProviderRegistry (fallback chain per capability)
│   ├── embedding: Vec<Arc<dyn EmbeddingProvider>>  // ordered by priority
│   ├── nli: Vec<Arc<dyn NliProvider>>
│   ├── classify: Vec<Arc<dyn ClassifyProvider>>
│   ├── stance: Vec<Arc<dyn StanceProvider>>        // NEW: dedicated stance
│   ├── chat: Vec<Arc<dyn ChatProvider>>
│   └── generate: Vec<Arc<dyn GenerateProvider>>
├── ModelManager (RAM budget, lazy loading)
└── TokenizerRegistry
```

### Provider Trait Hierarchy

```
EmbeddingProvider              NliProvider                StanceProvider (NEW)
     │                              │                           │
     ├── FastEmbedProvider         ├── OnnxNliProvider         ├── DedicatedStanceProvider (future)
     └── HuggingFaceClient         └── HuggingFaceClient       └── ZeroShotStanceProvider (wraps ClassifyProvider)

ClassifyProvider               ChatProvider               GenerateProvider
     │                              │                           │
     └── HuggingFaceClient         └── LlmChatProvider         └── LlmChatProvider
```

### Fallback Chain Flow

```
User: gateway.embed("hello", "all-MiniLM-L6-v2")
                    │
                    ▼
        ┌─────────────────────┐
        │  ProviderRegistry   │
        │  embedding providers│
        └─────────┬───────────┘
                  │ try in order
                  ▼
        ┌─────────────────────┐
        │  FastEmbedProvider  │ ──► checks: right model? RAM available?
        │  (priority 0)       │ ──► if no: return ModelNotAvailable
        └─────────┬───────────┘
                  │ ModelNotAvailable
                  ▼
        ┌─────────────────────┐
        │  HuggingFaceClient  │ ──► calls API with model string
        │  (priority 1)       │ ──► returns embedding
        └─────────────────────┘
```

---

## 2. New Types

### 2.1 StanceResult (örlög requirement)

```rust
// src/types/future.rs additions

/// Stance detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StanceResult {
    pub favor: f32,
    pub against: f32,
    pub neutral: f32,
    pub label: StanceLabel,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StanceLabel {
    Favor,
    Against,
    Neutral,
}

impl StanceResult {
    pub fn from_scores(favor: f32, against: f32, neutral: f32, target: impl Into<String>) -> Self {
        let label = if favor >= against && favor >= neutral {
            StanceLabel::Favor
        } else if against >= favor && against >= neutral {
            StanceLabel::Against
        } else {
            StanceLabel::Neutral
        };
        Self { favor, against, neutral, label, target: target.into() }
    }
}
```

### 2.2 Model Management Types

```rust
// src/types/model.rs (new file)

/// Information about an available model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub capabilities: Vec<ModelCapability>,
    pub context_window: Option<usize>,
    pub dimensions: Option<usize>,  // for embedding models
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCapability {
    Chat,
    Generate,
    Embed,
    Nli,
    Classify,
}

/// Runtime status of a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Loading,
    Ready,
    Unavailable { reason: String },
}
```

### 2.3 Token Type

```rust
// src/types/token.rs (new file)

/// A single token from tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub start: usize,  // byte offset in original text
    pub end: usize,
}
```

---

## 3. Provider Traits

### 3.1 Core Traits

```rust
// src/providers/traits.rs (new file)

use async_trait::async_trait;
use std::pin::Pin;
use futures_util::Stream;

use crate::{
    Result, Embedding, NliResult, ClassifyResult, StanceResult,
    ChatResponse, ChatEvent, ChatOptions, Message, ToolDefinition,
    GenerateResponse, GenerateEvent, GenerateOptions,
};

/// Provider for text embeddings.
/// 
/// Providers receive the model string and self-report availability.
/// Return `ModelNotAvailable` to signal the registry should try the next provider.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Generate embedding for a single text.
    /// 
    /// Returns `ModelNotAvailable` if this provider cannot handle the model
    /// (wrong model, RAM constrained, etc.) — registry will try next provider.
    async fn embed(&self, text: &str, model: &str) -> Result<Embedding>;

    /// Generate embeddings for multiple texts (batch).
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        // Default: sequential fallback
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text, model).await?);
        }
        Ok(results)
    }
}

/// Provider for natural language inference.
#[async_trait]
pub trait NliProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Infer entailment/contradiction/neutral between premise and hypothesis.
    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>;

    /// Batch NLI inference.
    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (premise, hypothesis) in pairs {
            results.push(self.infer_nli(premise, hypothesis, model).await?);
        }
        Ok(results)
    }
}

/// Provider for zero-shot text classification.
#[async_trait]
pub trait ClassifyProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Zero-shot classification with candidate labels.
    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult>;
}

/// Provider for stance detection (dedicated trait, separate from ClassifyProvider).
/// 
/// Stance detection determines whether text expresses favor/against/neutral
/// toward a specific target topic.
#[async_trait]
pub trait StanceProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Detect stance toward a target topic.
    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult>;
}

/// Wrapper that implements StanceProvider using zero-shot classification.
/// Used as fallback when no dedicated stance model is available.
pub struct ZeroShotStanceProvider {
    inner: Arc<dyn ClassifyProvider>,
    /// Model to use for zero-shot classification
    model: String,
}

impl ZeroShotStanceProvider {
    pub fn new(inner: Arc<dyn ClassifyProvider>, model: impl Into<String>) -> Self {
        Self { inner, model: model.into() }
    }
}

#[async_trait]
impl StanceProvider for ZeroShotStanceProvider {
    fn name(&self) -> &str { "zero-shot-stance" }

    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        // Construct prompt that includes target
        let prompt = format!("{} [Target: {}]", text, target);
        let labels = ["favor", "against", "neutral"];
        
        // Use the configured model, ignore passed model (we're a fallback)
        let result = self.inner.classify_zero_shot(&prompt, &labels, &self.model).await?;
        
        let favor = *result.scores.get("favor").unwrap_or(&0.0);
        let against = *result.scores.get("against").unwrap_or(&0.0);
        let neutral = *result.scores.get("neutral").unwrap_or(&0.0);
        
        Ok(StanceResult::from_scores(favor, against, neutral, target))
    }
}

/// Provider for multi-turn chat.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Non-streaming chat completion.
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse>;

    /// Streaming chat completion.
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>>;
}

/// Provider for single-turn text generation.
#[async_trait]
pub trait GenerateProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Non-streaming text generation.
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse>;

    /// Streaming text generation.
    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>>;
}
```

### 3.2 Error Type Addition

```rust
// src/error.rs addition

#[derive(Debug, thiserror::Error)]
pub enum RatatoskrError {
    // ... existing variants ...

    /// Provider cannot handle this model (wrong model, RAM constrained, etc.)
    /// Registry should try the next provider in the fallback chain.
    #[error("model not available from this provider")]
    ModelNotAvailable,
}
```

With the fallback chain design, we don't need `ModelMatcher` enums. Providers self-report
whether they can handle a model via `ModelNotAvailable` error. The registry just tries
providers in priority order.

---

## 4. Provider Registry

```rust
// src/providers/registry.rs (new file)

use std::sync::Arc;
use std::pin::Pin;
use futures_util::Stream;
use super::traits::*;
use crate::{Result, RatatoskrError, Message, ToolDefinition, ChatOptions, ChatEvent, GenerateOptions, GenerateEvent};

/// Registry of providers with fallback chain semantics.
/// 
/// Providers are stored in priority order (index 0 = highest priority).
/// When a capability is requested, the registry tries providers in order
/// until one succeeds or returns a non-`ModelNotAvailable` error.
#[derive(Default)]
pub struct ProviderRegistry {
    embedding: Vec<Arc<dyn EmbeddingProvider>>,
    nli: Vec<Arc<dyn NliProvider>>,
    classify: Vec<Arc<dyn ClassifyProvider>>,
    stance: Vec<Arc<dyn StanceProvider>>,
    chat: Vec<Arc<dyn ChatProvider>>,
    generate: Vec<Arc<dyn GenerateProvider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // Registration methods (appends to end = lowest priority)
    // Call in priority order: first registered = highest priority
    
    pub fn add_embedding(&mut self, provider: Arc<dyn EmbeddingProvider>) {
        self.embedding.push(provider);
    }

    pub fn add_nli(&mut self, provider: Arc<dyn NliProvider>) {
        self.nli.push(provider);
    }

    pub fn add_classify(&mut self, provider: Arc<dyn ClassifyProvider>) {
        self.classify.push(provider);
    }

    pub fn add_stance(&mut self, provider: Arc<dyn StanceProvider>) {
        self.stance.push(provider);
    }

    pub fn add_chat(&mut self, provider: Arc<dyn ChatProvider>) {
        self.chat.push(provider);
    }

    pub fn add_generate(&mut self, provider: Arc<dyn GenerateProvider>) {
        self.generate.push(provider);
    }

    // Fallback chain execution
    // Tries providers in order; ModelNotAvailable means "try next"
    
    pub async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        for provider in &self.embedding {
            match provider.embed(text, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        for provider in &self.embedding {
            match provider.embed_batch(texts, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        for provider in &self.nli {
            match provider.infer_nli(premise, hypothesis, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        for provider in &self.nli {
            match provider.infer_nli_batch(pairs, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult> {
        for provider in &self.classify {
            match provider.classify_zero_shot(text, labels, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        for provider in &self.stance {
            match provider.classify_stance(text, target, model).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        for provider in &self.chat {
            match provider.chat(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        for provider in &self.chat {
            match provider.chat_stream(messages, tools, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        for provider in &self.generate {
            match provider.generate(prompt, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        for provider in &self.generate {
            match provider.generate_stream(prompt, options).await {
                Ok(result) => return Ok(result),
                Err(RatatoskrError::ModelNotAvailable) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(RatatoskrError::NoProvider)
    }

    // Capability introspection
    
    pub fn has_embedding(&self) -> bool { !self.embedding.is_empty() }
    pub fn has_nli(&self) -> bool { !self.nli.is_empty() }
    pub fn has_classify(&self) -> bool { !self.classify.is_empty() }
    pub fn has_stance(&self) -> bool { !self.stance.is_empty() }
    pub fn has_chat(&self) -> bool { !self.chat.is_empty() }
    pub fn has_generate(&self) -> bool { !self.generate.is_empty() }

    /// List all registered provider names per capability (in priority order).
    pub fn provider_names(&self) -> ProviderNames {
        ProviderNames {
            embedding: self.embedding.iter().map(|p| p.name().to_string()).collect(),
            nli: self.nli.iter().map(|p| p.name().to_string()).collect(),
            classify: self.classify.iter().map(|p| p.name().to_string()).collect(),
            stance: self.stance.iter().map(|p| p.name().to_string()).collect(),
            chat: self.chat.iter().map(|p| p.name().to_string()).collect(),
            generate: self.generate.iter().map(|p| p.name().to_string()).collect(),
        }
    }
}

/// Provider names per capability, in priority order.
#[derive(Debug, Clone)]
pub struct ProviderNames {
    pub embedding: Vec<String>,
    pub nli: Vec<String>,
    pub classify: Vec<String>,
    pub stance: Vec<String>,
    pub chat: Vec<String>,
    pub generate: Vec<String>,
}
```

---

## 5. Provider Implementations

### 5.1 HuggingFaceClient

HuggingFace can handle any model, so it never returns `ModelNotAvailable`. It's registered
as a low-priority fallback that accepts any model string and forwards to the API.

```rust
// src/providers/huggingface.rs additions

#[async_trait]
impl EmbeddingProvider for HuggingFaceClient {
    fn name(&self) -> &str { "huggingface" }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        // HuggingFace API can handle any model - pass it through
        self.embed_api(text, model).await
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        self.embed_batch_api(texts, model).await
    }
}

#[async_trait]
impl NliProvider for HuggingFaceClient {
    fn name(&self) -> &str { "huggingface" }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        self.infer_nli_api(premise, hypothesis, model).await
    }
}

#[async_trait]
impl ClassifyProvider for HuggingFaceClient {
    fn name(&self) -> &str { "huggingface" }

    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult> {
        self.classify_api(text, labels, model).await
    }
}
```

**Note:** The existing `HuggingFaceClient` methods already take model as a param, so this is
a straightforward trait impl. The client is registered as fallback and accepts any model.

### 5.2 FastEmbedProvider

Local providers check: (1) is this my model? (2) do I have RAM to load it?
If either fails, return `ModelNotAvailable` so registry tries the next provider.

```rust
// src/providers/fastembed.rs additions

use std::sync::Arc;
use crate::model::ModelManager;

pub struct FastEmbedProvider {
    model_name: String,
    model_info: EmbeddingModelInfo,
    local_model: LocalEmbeddingModel,
    manager: Arc<ModelManager>,
}

impl FastEmbedProvider {
    pub fn new(model: LocalEmbeddingModel, manager: Arc<ModelManager>) -> Self {
        Self {
            model_name: model.name().to_string(),
            model_info: model.into(),
            local_model: model,
            manager,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    fn name(&self) -> &str {
        &self.model_name
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        // Check 1: Is this the model we handle?
        if model != self.model_name {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        // Check 2: Can we load it (RAM budget)?
        if !self.manager.can_load(&self.model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        // Get or load the model through manager
        let loaded = self.manager.get_or_load_embedding(self.local_model.clone())?;
        
        // FastEmbed is sync, wrap in spawn_blocking
        let text = text.to_owned();
        let model_name = self.model_name.clone();
        let dimensions = self.model_info.dimensions;
        
        tokio::task::spawn_blocking(move || {
            let vectors = loaded.embed(vec![&text], None)
                .map_err(|e| RatatoskrError::Llm(e.to_string()))?;
            
            let values = vectors.into_iter().next()
                .ok_or_else(|| RatatoskrError::EmptyResponse)?;
            
            Ok(Embedding {
                values,
                model: model_name,
                dimensions,
            })
        })
        .await
        .map_err(|e| RatatoskrError::Llm(e.to_string()))?
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        // Same checks
        if model != self.model_name {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        if !self.manager.can_load(&self.model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        let loaded = self.manager.get_or_load_embedding(self.local_model.clone())?;
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model_name = self.model_name.clone();
        let dimensions = self.model_info.dimensions;
        
        tokio::task::spawn_blocking(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vectors = loaded.embed(text_refs, None)
                .map_err(|e| RatatoskrError::Llm(e.to_string()))?;
            
            Ok(vectors.into_iter().map(|values| Embedding {
                values,
                model: model_name.clone(),
                dimensions,
            }).collect())
        })
        .await
        .map_err(|e| RatatoskrError::Llm(e.to_string()))?
    }
}
```

### 5.3 OnnxNliProvider

Same pattern as FastEmbed: check model name, check RAM, defer to ModelManager.

```rust
// src/providers/onnx_nli.rs additions

use std::sync::Arc;
use crate::model::ModelManager;

pub struct OnnxNliProvider {
    model_name: String,
    local_model: LocalNliModel,
    manager: Arc<ModelManager>,
}

impl OnnxNliProvider {
    pub fn new(model: LocalNliModel, manager: Arc<ModelManager>) -> Self {
        Self {
            model_name: model.name().to_string(),
            local_model: model,
            manager,
        }
    }
}

#[async_trait]
impl NliProvider for OnnxNliProvider {
    fn name(&self) -> &str {
        &self.model_name
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        // Check 1: Is this the model we handle?
        if model != self.model_name {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        // Check 2: Can we load it (RAM budget)?
        if !self.manager.can_load(&self.model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        // Get or load through manager (returns Arc<LoadedNliModel>)
        let loaded = self.manager.get_or_load_nli(self.local_model.clone())?;
        
        // ONNX is sync, wrap in spawn_blocking
        let premise = premise.to_owned();
        let hypothesis = hypothesis.to_owned();
        
        tokio::task::spawn_blocking(move || {
            loaded.infer(&premise, &hypothesis)
        })
        .await
        .map_err(|e| RatatoskrError::Llm(e.to_string()))?
    }

    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        if model != self.model_name {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        if !self.manager.can_load(&self.model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }
        
        let loaded = self.manager.get_or_load_nli(self.local_model.clone())?;
        let pairs: Vec<(String, String)> = pairs.iter()
            .map(|(p, h)| (p.to_string(), h.to_string()))
            .collect();
        
        tokio::task::spawn_blocking(move || {
            loaded.infer_batch(&pairs)
        })
        .await
        .map_err(|e| RatatoskrError::Llm(e.to_string()))?
    }
}
```

**Note:** The `LoadedNliModel` returned by `ModelManager` wraps `Arc<Session>` and `Arc<Tokenizer>`
internally, making it safe to move into `spawn_blocking`.

### 5.4 LlmChatProvider (new)

```rust
// src/providers/llm_chat.rs (new file)

use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use llm::LLMProvider;

use crate::{
    Result, RatatoskrError, Message, ToolDefinition, ChatOptions, ChatResponse, ChatEvent,
    GenerateOptions, GenerateResponse, GenerateEvent,
};
use crate::convert::{to_llm_messages, to_llm_tools, from_llm_tool_calls, from_llm_usage};
use super::traits::{ChatProvider, GenerateProvider};

/// Wraps the llm crate's provider to implement our traits.
pub struct LlmChatProvider {
    inner: Box<dyn LLMProvider>,
    name: String,
}

impl LlmChatProvider {
    pub fn new(provider: Box<dyn LLMProvider>, name: impl Into<String>) -> Self {
        Self {
            inner: provider,
            name: name.into(),
        }
    }
}

#[async_trait]
impl ChatProvider for LlmChatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let (system_prompt, llm_messages) = to_llm_messages(messages);
        let llm_tools = tools.map(to_llm_tools);

        let response = if let Some(ref tools) = llm_tools {
            self.inner.chat_with_tools(&llm_messages, Some(tools)).await
        } else {
            self.inner.chat(&llm_messages).await
        }.map_err(RatatoskrError::from)?;

        // Convert response...
        Ok(ChatResponse {
            content: response.text().unwrap_or_default(),
            reasoning: response.thinking(),
            tool_calls: response.tool_calls().map(|tc| from_llm_tool_calls(&tc)).unwrap_or_default(),
            usage: response.usage().map(|u| from_llm_usage(&u)),
            model: Some(options.model.clone()),
            finish_reason: Default::default(),
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let (system_prompt, llm_messages) = to_llm_messages(messages);
        let llm_tools = tools.map(to_llm_tools);

        let stream = self.inner
            .chat_stream_with_tools(&llm_messages, llm_tools.as_deref())
            .await
            .map_err(RatatoskrError::from)?;

        let converted = stream.map(|result| {
            result
                .map(|chunk| ChatEvent::Content(chunk.to_string()))
                .map_err(RatatoskrError::from)
        });

        Ok(Box::pin(converted))
    }
}

#[async_trait]
impl GenerateProvider for LlmChatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        // Use llm's completion API if available, else wrap in chat
        use llm::completion::{CompletionProvider, CompletionRequest};
        
        let request = CompletionRequest::new(prompt.to_string())
            .max_tokens(options.max_tokens.map(|n| n as u32))
            .temperature(options.temperature);
        
        let response = self.inner.complete(&request).await
            .map_err(RatatoskrError::from)?;
        
        Ok(GenerateResponse {
            text: response,
            usage: None,
            model: Some(options.model.clone()),
            finish_reason: Default::default(),
        })
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        // llm crate doesn't have streaming completion, wrap chat
        let messages = vec![Message::user(prompt)];
        let chat_options = ChatOptions::default()
            .model(&options.model)
            .temperature(options.temperature.unwrap_or(1.0));
        
        let stream = self.chat_stream(&messages, None, &chat_options).await?;
        
        let converted = stream.map(|result| {
            result.map(|event| match event {
                ChatEvent::Content(text) => GenerateEvent::Text(text),
                ChatEvent::Done => GenerateEvent::Done,
                _ => GenerateEvent::Text(String::new()),
            })
        });
        
        Ok(Box::pin(converted))
    }
}
```

---

## 5.5 ModelManager (RAM Budget)

The `ModelManager` tracks loaded models and enforces RAM budget. Local providers
consult it before attempting to load models.

```rust
// src/model/manager.rs updates

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::{Result, RatatoskrError};
use crate::providers::{LocalEmbeddingModel, LocalNliModel};

/// Tracks loaded models and enforces RAM budget.
pub struct ModelManager {
    /// Maximum RAM budget in bytes (None = unlimited)
    ram_budget: Option<usize>,
    /// Currently loaded embedding models
    embedding_models: RwLock<HashMap<String, LoadedEmbeddingModel>>,
    /// Currently loaded NLI models  
    nli_models: RwLock<HashMap<String, LoadedNliModel>>,
    /// Estimated RAM usage per loaded model
    loaded_sizes: RwLock<HashMap<String, usize>>,
}

/// A loaded embedding model (wraps fastembed internals)
pub struct LoadedEmbeddingModel {
    inner: fastembed::TextEmbedding,
    model_name: String,
}

/// A loaded NLI model (wraps ONNX session + tokenizer in Arc for spawn_blocking)
pub struct LoadedNliModel {
    session: Arc<ort::Session>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
}

impl ModelManager {
    pub fn new(ram_budget: Option<usize>) -> Self {
        Self {
            ram_budget,
            embedding_models: RwLock::new(HashMap::new()),
            nli_models: RwLock::new(HashMap::new()),
            loaded_sizes: RwLock::new(HashMap::new()),
        }
    }

    /// Check if we have RAM budget to load a model.
    /// Returns false if loading would exceed budget.
    pub fn can_load(&self, model_name: &str) -> bool {
        let Some(budget) = self.ram_budget else {
            return true;  // No budget = unlimited
        };

        let sizes = self.loaded_sizes.read().unwrap();
        
        // Already loaded? OK
        if sizes.contains_key(model_name) {
            return true;
        }

        // Estimate size for this model (could be more sophisticated)
        let estimated_size = self.estimate_model_size(model_name);
        let current_usage: usize = sizes.values().sum();
        
        current_usage + estimated_size <= budget
    }

    /// Get or load an embedding model.
    pub fn get_or_load_embedding(
        &self,
        model: LocalEmbeddingModel,
    ) -> Result<Arc<LoadedEmbeddingModel>> {
        let model_name = model.name().to_string();
        
        // Fast path: already loaded
        {
            let models = self.embedding_models.read().unwrap();
            if let Some(loaded) = models.get(&model_name) {
                return Ok(Arc::new(loaded.clone()));
            }
        }

        // Slow path: load the model
        let mut models = self.embedding_models.write().unwrap();
        
        // Double-check after acquiring write lock
        if let Some(loaded) = models.get(&model_name) {
            return Ok(Arc::new(loaded.clone()));
        }

        // Check RAM budget
        if !self.can_load(&model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }

        // Load the model
        let inner = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(model.into())
                .with_show_download_progress(true)
        ).map_err(|e| RatatoskrError::Llm(e.to_string()))?;

        let loaded = LoadedEmbeddingModel {
            inner,
            model_name: model_name.clone(),
        };

        // Track size
        let size = self.estimate_model_size(&model_name);
        self.loaded_sizes.write().unwrap().insert(model_name.clone(), size);

        models.insert(model_name, loaded.clone());
        Ok(Arc::new(loaded))
    }

    /// Get or load an NLI model.
    pub fn get_or_load_nli(
        &self,
        model: LocalNliModel,
    ) -> Result<Arc<LoadedNliModel>> {
        // Similar pattern to embedding...
        todo!()
    }

    /// Unload a model to free RAM.
    pub fn unload(&self, model_name: &str) -> bool {
        let mut removed = false;
        
        if self.embedding_models.write().unwrap().remove(model_name).is_some() {
            removed = true;
        }
        if self.nli_models.write().unwrap().remove(model_name).is_some() {
            removed = true;
        }
        if removed {
            self.loaded_sizes.write().unwrap().remove(model_name);
        }
        
        removed
    }

    /// Estimate model size in bytes (could read from config or model metadata)
    fn estimate_model_size(&self, model_name: &str) -> usize {
        // Rough estimates based on model type
        match model_name {
            n if n.contains("MiniLM") => 100 * 1024 * 1024,      // ~100MB
            n if n.contains("bge-small") => 130 * 1024 * 1024,   // ~130MB
            n if n.contains("bge-base") => 440 * 1024 * 1024,    // ~440MB
            n if n.contains("deberta") => 400 * 1024 * 1024,     // ~400MB
            _ => 200 * 1024 * 1024,  // Default ~200MB
        }
    }

    /// Get current RAM usage.
    pub fn current_usage(&self) -> usize {
        self.loaded_sizes.read().unwrap().values().sum()
    }

    /// List currently loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        let mut models = Vec::new();
        models.extend(self.embedding_models.read().unwrap().keys().cloned());
        models.extend(self.nli_models.read().unwrap().keys().cloned());
        models
    }
}
```

---

## 6. Builder Updates

```rust
// src/gateway/builder.rs updates

impl RatatoskrBuilder {
    pub fn build(self) -> Result<EmbeddedGateway> {
        let mut registry = ProviderRegistry::new();
        
        // Create shared ModelManager for local providers
        #[cfg(feature = "local-inference")]
        let model_manager = Arc::new(ModelManager::new(self.ram_budget));

        // === Register LOCAL providers FIRST (higher priority) ===
        
        #[cfg(feature = "local-inference")]
        if let Some(model) = self.local_embedding_model {
            let provider = Arc::new(FastEmbedProvider::new(model, model_manager.clone()));
            registry.add_embedding(provider);  // Priority 0
        }

        #[cfg(feature = "local-inference")]
        if let Some(model) = self.local_nli_model {
            let provider = Arc::new(OnnxNliProvider::new(model, model_manager.clone()));
            registry.add_nli(provider);  // Priority 0
        }

        // === Register API providers as FALLBACKS (lower priority) ===
        
        #[cfg(feature = "huggingface")]
        if let Some(key) = self.huggingface_key {
            let client = Arc::new(HuggingFaceClient::new(key));
            
            // HuggingFace accepts any model, registered as fallback
            registry.add_embedding(client.clone());  // Priority 1 (after local)
            registry.add_nli(client.clone());
            registry.add_classify(client.clone());
            
            // Also register ZeroShotStanceProvider as stance fallback
            let stance_fallback = Arc::new(ZeroShotStanceProvider::new(
                client.clone(),
                "facebook/bart-large-mnli",
            ));
            registry.add_stance(stance_fallback);
        }

        // === Register chat providers ===
        
        if let Some(key) = &self.openrouter_key {
            let llm_provider = LLMBuilder::new()
                .backend(LLMBackend::OpenRouter)
                .api_key(key)
                .build()?;
            
            let chat_provider = Arc::new(LlmChatProvider::new(llm_provider, "openrouter"));
            registry.add_chat(chat_provider.clone());
            registry.add_generate(chat_provider);
        }
        
        // Similar for anthropic, openai, ollama, google...

        Ok(EmbeddedGateway::new(
            registry,
            #[cfg(feature = "local-inference")]
            model_manager,
            #[cfg(feature = "local-inference")]
            self.tokenizer_registry(),
        ))
    }
}
```

---

## 7. EmbeddedGateway Refactor

The gateway becomes a thin wrapper that delegates to the registry's fallback chain methods.

```rust
// src/gateway/embedded.rs refactor

pub struct EmbeddedGateway {
    registry: ProviderRegistry,
    #[cfg(feature = "local-inference")]
    model_manager: Arc<ModelManager>,
    #[cfg(feature = "local-inference")]
    tokenizer_registry: Arc<TokenizerRegistry>,
}

impl EmbeddedGateway {
    pub fn new(
        registry: ProviderRegistry,
        #[cfg(feature = "local-inference")]
        model_manager: Arc<ModelManager>,
        #[cfg(feature = "local-inference")]
        tokenizer_registry: Arc<TokenizerRegistry>,
    ) -> Self {
        Self {
            registry,
            #[cfg(feature = "local-inference")]
            model_manager,
            #[cfg(feature = "local-inference")]
            tokenizer_registry,
        }
    }
}

#[async_trait]
impl ModelGateway for EmbeddedGateway {
    // All methods delegate to registry, which handles fallback chain internally
    
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.registry.chat(messages, tools, options).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        self.registry.chat_stream(messages, tools, options).await
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        self.registry.embed(text, model).await
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        self.registry.embed_batch(texts, model).await
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        self.registry.infer_nli(premise, hypothesis, model).await
    }

    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        self.registry.infer_nli_batch(pairs, model).await
    }

    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult> {
        self.registry.classify_zero_shot(text, labels, model).await
    }

    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        self.registry.classify_stance(text, target, model).await
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        self.registry.generate(prompt, options).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        self.registry.generate_stream(prompt, options).await
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: self.registry.has_chat(),
            chat_streaming: self.registry.has_chat(),
            generate: self.registry.has_generate(),
            tool_use: self.registry.has_chat(),
            embeddings: self.registry.has_embedding(),
            nli: self.registry.has_nli(),
            classification: self.registry.has_classify(),
            stance: self.registry.has_stance(),
            token_counting: cfg!(feature = "local-inference"),
            local_inference: cfg!(feature = "local-inference"),
        }
    }

    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        #[cfg(feature = "local-inference")]
        {
            self.tokenizer_registry.count_tokens(text, model)
        }
        #[cfg(not(feature = "local-inference"))]
        {
            Err(RatatoskrError::NotImplemented("count_tokens"))
        }
    }

    fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>> {
        #[cfg(feature = "local-inference")]
        {
            self.tokenizer_registry.tokenize(text, model)
        }
        #[cfg(not(feature = "local-inference"))]
        {
            Err(RatatoskrError::NotImplemented("tokenize"))
        }
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        // Combine provider names with model manager's loaded models
        let names = self.registry.provider_names();
        
        let mut models = Vec::new();
        for name in names.embedding {
            models.push(ModelInfo {
                id: name.clone(),
                provider: name,
                capabilities: vec![ModelCapability::Embed],
                context_window: None,
                dimensions: None,  // Could look up from provider
            });
        }
        // ... similar for other capabilities
        models
    }

    fn model_status(&self, model: &str) -> ModelStatus {
        #[cfg(feature = "local-inference")]
        {
            if self.model_manager.loaded_models().contains(&model.to_string()) {
                ModelStatus::Ready
            } else if self.model_manager.can_load(model) {
                ModelStatus::Available
            } else {
                ModelStatus::Unavailable { 
                    reason: "RAM budget exceeded".into() 
                }
            }
        }
        #[cfg(not(feature = "local-inference"))]
        {
            // For API-only mode, always "ready" if provider exists
            ModelStatus::Ready
        }
    }
}
```

---

## 8. ModelGateway Trait Updates

Add new methods to the user-facing trait:

```rust
// src/traits.rs updates

#[async_trait]
pub trait ModelGateway: Send + Sync {
    // ===== Existing Phase 1-4 methods =====
    
    async fn chat_stream(...) -> Result<...>;
    async fn chat(...) -> Result<ChatResponse>;
    fn capabilities(&self) -> Capabilities;
    async fn embed(&self, text: &str, model: &str) -> Result<Embedding>;
    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>>;
    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult>;
    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>>;
    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult>;
    fn count_tokens(&self, text: &str, model: &str) -> Result<usize>;
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse>;
    async fn generate_stream(&self, prompt: &str, options: &GenerateOptions) -> Result<...>;

    // ===== New methods =====

    /// Stance detection toward a target topic.
    async fn classify_stance(
        &self,
        _text: &str,
        _target: &str,
        _model: &str,
    ) -> Result<StanceResult> {
        Err(RatatoskrError::NotImplemented("classify_stance"))
    }

    /// Tokenize text into individual tokens.
    fn tokenize(&self, _text: &str, _model: &str) -> Result<Vec<Token>> {
        Err(RatatoskrError::NotImplemented("tokenize"))
    }

    /// List all available models and their capabilities.
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![]
    }

    /// Get the status of a specific model.
    fn model_status(&self, _model: &str) -> ModelStatus {
        ModelStatus::Unavailable { reason: "Not implemented".into() }
    }
}
```

Also update `Capabilities` struct:

```rust
// src/types/capabilities.rs updates

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capabilities {
    pub chat: bool,
    pub chat_streaming: bool,
    pub generate: bool,
    pub tool_use: bool,
    pub embeddings: bool,
    pub nli: bool,
    pub classification: bool,
    pub stance: bool,           // NEW
    pub token_counting: bool,
    pub local_inference: bool,
}
```

---

## 9. Implementation Tasks

### Task 1: New Types ✅
- [x] Create `src/types/stance.rs` with `StanceResult`, `StanceLabel`
- [x] Create `src/types/model.rs` with `ModelInfo`, `ModelStatus`, `ModelCapability`
- [x] Create `src/types/token.rs` with `Token`
- [x] Update `src/types/capabilities.rs` to add `stance` field
- [x] Update `src/types/mod.rs` exports
- [x] Update `src/lib.rs` exports
- [x] Add `ModelNotAvailable` error variant to `RatatoskrError`
- [x] Add tests for new types

**Verify:** `cargo test --lib types::` ✅ (9 tests pass)

### Task 2: Provider Traits + Registry ✅
- [x] Create `src/providers/traits.rs` with all provider traits:
  - `EmbeddingProvider`
  - `NliProvider`  
  - `ClassifyProvider`
  - `StanceProvider`
  - `ChatProvider`
  - `GenerateProvider`
- [x] Create `ZeroShotStanceProvider` wrapper in traits.rs
- [x] Create `src/providers/registry.rs` with `ProviderRegistry` (fallback chain)
- [x] Update `src/providers/mod.rs` exports
- [x] Add unit tests for `ProviderRegistry` fallback behavior

**Verify:** `cargo test --lib providers::` ✅ (9 tests pass)

### Task 3: HuggingFaceClient Trait Impls ✅
- [x] Implement `EmbeddingProvider` for `HuggingFaceClient` (passthrough model param)
- [x] Implement `NliProvider` for `HuggingFaceClient`
- [x] Implement `ClassifyProvider` for `HuggingFaceClient`
- [x] Tests via existing HuggingFace tests

**Verify:** `cargo test --all-features --lib` ✅ (25 tests pass)

### Task 4: ModelManager RAM Budget ✅
- [x] Update `src/model/manager.rs` with RAM budget tracking
- [x] Add `can_load()` method for RAM budget checks
- [x] Add `estimate_model_size()` helper
- [x] Add `current_usage()`, `ram_budget()` getters
- [x] Update `embedding()` and `nli()` to check RAM before loading
- [x] Update `unload_*()` methods to track size changes
- [x] Add tests for RAM budget enforcement

**Verify:** `cargo test --features local-inference --lib model::manager` ✅

### Task 5: FastEmbedProvider Trait Impl ✅
- [x] Create `LocalEmbeddingProvider` that wraps `Arc<ModelManager>`
- [x] Implement `EmbeddingProvider`:
  - Check model name match → `ModelNotAvailable`
  - Delegate to manager for lazy loading (checks RAM budget)
- [x] Wrap sync calls in `spawn_blocking`
- [x] Export from providers module

**Verify:** `cargo test --features local-inference --lib` ✅

### Task 6: OnnxNliProvider Trait Impl ✅
- [x] Create `LocalNliProvider` that wraps `Arc<ModelManager>`
- [x] Implement `NliProvider`:
  - Check model name match → `ModelNotAvailable`
  - Delegate to manager for lazy loading (checks RAM budget)
- [x] Wrap sync calls in `spawn_blocking`
- [x] Export from providers module

**Verify:** `cargo test --features local-inference --lib` ✅

### Task 7: LlmChatProvider
- [ ] Create `src/providers/llm_chat.rs`
- [ ] Implement `ChatProvider` for `LlmChatProvider`
- [ ] Implement `GenerateProvider` for `LlmChatProvider`
- [ ] Handle type conversions (reuse existing convert module)
- [ ] Add tests

**Verify:** `cargo test --test llm_chat_provider_test`

### Task 8: Builder Refactor
- [ ] Add `ram_budget` field and `.ram_budget(bytes)` method
- [ ] Create `ProviderRegistry` in `build()`
- [ ] Create shared `ModelManager` for local providers
- [ ] Register local providers FIRST (priority 0)
- [ ] Register API providers as fallbacks (priority 1)
- [ ] Create `ZeroShotStanceProvider` as stance fallback
- [ ] Update builder tests

**Verify:** `cargo test --test builder_test`

### Task 9: EmbeddedGateway Refactor
- [ ] Replace old fields with `ProviderRegistry` + `ModelManager`
- [ ] Update all trait method impls to delegate to registry
- [ ] Implement `classify_stance`, `tokenize`, `list_models`, `model_status`
- [ ] Update `capabilities()` to include `stance` field
- [ ] Remove old `CapabilityRouter` usage

**Verify:** `cargo test --test gateway_test`

### Task 10: TokenizerRegistry Updates
- [ ] Add `tokenize()` method returning `Vec<Token>`
- [ ] Update tests

**Verify:** `cargo test --test tokenizer_test --features local-inference`

### Task 11: ModelGateway Trait Updates
- [ ] Add `classify_stance()` method with default stub
- [ ] Add `tokenize()` method with default stub
- [ ] Add `list_models()` method with default stub
- [ ] Add `model_status()` method with default stub
- [ ] Update existing trait test

**Verify:** `cargo test --test traits_test`

### Task 12: Delete Old Code
- [ ] Remove `src/gateway/routing.rs` (old `CapabilityRouter`)
- [ ] Clean up unused imports

**Verify:** `cargo check --features local-inference,huggingface`

### Task 13: Integration Tests
- [ ] Update existing integration tests
- [ ] Add test: local provider → API fallback on `ModelNotAvailable`
- [ ] Add test: RAM budget triggers fallback
- [ ] Add test: `classify_stance` via `ZeroShotStanceProvider`

**Verify:** `cargo test --features local-inference,huggingface`

### Task 14: Live Tests
- [ ] Add `classify_stance` live test
- [ ] Add RAM budget fallback live test
- [ ] Update existing live tests for new API

**Verify:** `cargo test --features local-inference,huggingface -- --ignored`

### Task 15: Documentation
- [ ] Update `CLAUDE.md` with new architecture
- [ ] Update `architecture.md` to match implementation
- [ ] Update `src/lib.rs` doc examples
- [ ] Run `cargo doc --no-deps`

### Task 16: Full Test Suite
- [ ] `just pre-push` passes
- [ ] All clippy warnings resolved

**Verify:** `just pre-push`

---

## 10. File Changes Summary

### New Files
- `src/types/stance.rs` — StanceResult, StanceLabel
- `src/types/model.rs` — ModelInfo, ModelStatus, ModelCapability
- `src/types/token.rs` — Token
- `src/providers/traits.rs` — all provider traits + ZeroShotStanceProvider
- `src/providers/registry.rs` — ProviderRegistry with fallback chain
- `src/providers/llm_chat.rs` — LlmChatProvider wrapper
- `tests/provider_traits_test.rs`
- `tests/llm_chat_provider_test.rs`

### Modified Files
- `src/error.rs` — add ModelNotAvailable variant
- `src/types/capabilities.rs` — add stance field
- `src/types/mod.rs` — add exports
- `src/lib.rs` — add exports
- `src/traits.rs` — add classify_stance, tokenize, list_models, model_status
- `src/providers/mod.rs` — add exports
- `src/providers/huggingface.rs` — implement provider traits
- `src/providers/fastembed.rs` — implement EmbeddingProvider + RAM checks
- `src/providers/onnx_nli.rs` — implement NliProvider + RAM checks
- `src/model/manager.rs` — add RAM budget, can_load, get_or_load methods
- `src/gateway/builder.rs` — use ProviderRegistry, add ram_budget
- `src/gateway/embedded.rs` — delegate to registry
- `src/gateway/mod.rs` — remove routing export
- `src/tokenizer/mod.rs` — add tokenize() method

### Deleted Files
- `src/gateway/routing.rs` — replaced by ProviderRegistry

---

## 11. Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| New types (stance, model, token) | ~100 | Low |
| Provider traits + ZeroShotStanceProvider | ~200 | Medium |
| ProviderRegistry (fallback chain) | ~200 | Medium |
| HuggingFace trait impls | ~80 | Low |
| ModelManager RAM budget | ~150 | Medium |
| FastEmbed trait impl + RAM checks | ~100 | Medium |
| OnnxNli trait impl + RAM checks | ~120 | Medium |
| LlmChatProvider | ~200 | Medium |
| Builder refactor | ~150 | Medium |
| Gateway refactor | ~100 | Low |
| Tokenizer updates | ~40 | Low |
| Tests | ~400 | Medium |
| **Total** | **~1840** | |

---

## 12. Resolved Questions

These questions were resolved during planning:

1. **Model parameter in traits:** ✅ RESOLVED — Provider traits DO take `model: &str` param. Providers self-report availability via `ModelNotAvailable` error. This enables transparent fallback (local → API when RAM constrained).

2. **HuggingFace as fallback:** ✅ RESOLVED — HuggingFace accepts any model string and passes it to the API. Registered as low-priority fallback.

3. **ModelManager role:** ✅ RESOLVED — ModelManager tracks RAM budget and handles lazy loading. Local providers consult it via `can_load()` before loading. Returns `ModelNotAvailable` if budget exceeded, triggering fallback to API.

4. **Stance detection:** ✅ RESOLVED — Separate `StanceProvider` trait. `ZeroShotStanceProvider` wrapper provides fallback using ClassifyProvider. Dedicated stance models can be added later.

---

## 13. Future Considerations (Phase 5+)

Things this design enables but doesn't implement:

1. **Decorator patterns:**
   ```rust
   CachingProvider<T: EmbeddingProvider>
   RetryingProvider<T: EmbeddingProvider>
   InstrumentedProvider<T: EmbeddingProvider>
   ```

2. **Service mode:** Registry can be serialized to config, protocol handler dispatches to registry methods.

3. **User-defined providers:** Implement trait, register with builder.

4. **LRU eviction:** ModelManager could evict least-recently-used models when RAM budget exceeded instead of failing.

5. **Dedicated stance models:** Research HuggingFace for stance-specific models (e.g., `cardiffnlp/twitter-roberta-base-stance`).
