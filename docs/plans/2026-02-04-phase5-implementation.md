# Phase 5: Service Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add gRPC service mode with `ratd` (daemon) and `rat` (CLI) binaries, enabling multiple clients to share a single gateway instance.

**Architecture:** Protobuf defines the wire format. `tonic` serves gRPC. `ratd` wraps an `EmbeddedGateway` behind gRPC handlers. `rat` connects via `ServiceClient` which implements `ModelGateway`. Config and secrets are TOML files with permission checks.

**Tech Stack:** tonic, prost, prost-build, clap, toml, tokio

**Progress:**
- [x] Task 1: Add gRPC Dependencies (f628cf2)
- [x] Task 2: Create Protobuf Service Definition (2b735d8)
- [x] Task 3: Proto Conversion Module (0941da0)
- [x] Task 4: gRPC Service Implementation (83d6712)
- [x] Task 5: Configuration Types (83d6712)
- [x] Task 6: ratd Binary (83d6712)
- [x] Task 7: ServiceClient (Client Library)
- [x] Task 8: rat CLI Binary
- [x] Task 9: systemd Unit File
- [ ] Task 10: Integration Tests
- [ ] Task 11: Update Documentation
- [ ] Task 12: Final Verification

**Implementation Notes:**
- `tonic-build` added as unconditional build-dep (build-deps can't be feature-gated); usage gated via `#[cfg]` in build.rs instead
- `server` feature includes `dep:dirs` for config path resolution
- `protoc` needed at build time — installed locally to `~/.local/bin/`
- `ModelCapability::Stance` maps to `MODEL_CAPABILITY_CLASSIFY` in proto (proto has 5 variants, native has 6)
- Streaming methods return `GrpcResult<Self::XStream>` not `GrpcResult<Response<Self::XStream>>` — tonic trait already wraps in `Response`
- `dirs` kept optional (gated by both `server` and `local-inference` features) rather than unconditional as plan suggested
- Proto↔native conversions centralized in `server::convert` as single source of truth (plan had client-side copies in `service_client.rs`)
- Client request building uses `From` trait impls instead of freestanding `to_proto_*` functions
- `server` module gated with `#[cfg(any(feature = "server", feature = "client"))]` — both features share proto types and conversions; `config` and `service` submodules gated to `server` only

---

## Task 1: Add gRPC Dependencies

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add feature flags and dependencies**

Add to `Cargo.toml`:

```toml
[features]
default = ["openai", "anthropic", "openrouter", "ollama", "google", "huggingface"]
# ... existing features ...
server = ["dep:tonic", "dep:prost", "dep:clap", "dep:toml"]
client = ["dep:tonic", "dep:prost", "dep:clap"]

[dependencies]
# ... existing deps ...

# gRPC (service mode)
tonic = { version = "0.13", optional = true }
prost = { version = "0.13", optional = true }
clap = { version = "4", features = ["derive", "env"], optional = true }
toml = { version = "0.8", optional = true }

[build-dependencies]
# ... existing ...
tonic-build = { version = "0.13", optional = true }

[features]
# update build-dependencies feature
server = ["dep:tonic", "dep:prost", "dep:clap", "dep:toml", "dep:tonic-build"]
```

**Step 2: Verify it compiles**

Run: `cargo check --features server,client`
Expected: Compiles (no proto files yet, but deps resolve)

**Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "feat(deps): add tonic/prost for gRPC service mode"
```

---

## Task 2: Create Protobuf Service Definition

**Files:**
- Create: `proto/ratatoskr.proto`
- Modify: `build.rs`

**Step 1: Create proto directory and service definition**

Create `proto/ratatoskr.proto`:

```protobuf
syntax = "proto3";
package ratatoskr.v1;

// =============================================================================
// Service Definition
// =============================================================================

service Ratatoskr {
    // Chat
    rpc Chat(ChatRequest) returns (ChatResponse);
    rpc ChatStream(ChatRequest) returns (stream ChatEvent);

    // Generation
    rpc Generate(GenerateRequest) returns (GenerateResponse);
    rpc GenerateStream(GenerateRequest) returns (stream GenerateEvent);

    // Embeddings
    rpc Embed(EmbedRequest) returns (EmbedResponse);
    rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);

    // NLI
    rpc InferNli(NliRequest) returns (NliResponse);
    rpc InferNliBatch(NliBatchRequest) returns (NliBatchResponse);

    // Classification
    rpc ClassifyZeroShot(ClassifyRequest) returns (ClassifyResponse);
    rpc ClassifyStance(StanceRequest) returns (StanceResponse);

    // Tokenization
    rpc CountTokens(TokenCountRequest) returns (TokenCountResponse);
    rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);

    // Model management
    rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
    rpc ModelStatus(ModelStatusRequest) returns (ModelStatusResponse);

    // Service health
    rpc Health(HealthRequest) returns (HealthResponse);
}

// =============================================================================
// Common Types
// =============================================================================

message Usage {
    uint32 prompt_tokens = 1;
    uint32 completion_tokens = 2;
    uint32 total_tokens = 3;
    optional uint32 reasoning_tokens = 4;
}

enum FinishReason {
    FINISH_REASON_UNSPECIFIED = 0;
    FINISH_REASON_STOP = 1;
    FINISH_REASON_LENGTH = 2;
    FINISH_REASON_TOOL_CALLS = 3;
    FINISH_REASON_CONTENT_FILTER = 4;
}

// =============================================================================
// Message Types
// =============================================================================

message Message {
    Role role = 1;
    string content = 2;
    repeated ToolCall tool_calls = 3;
    optional string name = 4;
    optional string tool_call_id = 5;  // For tool role messages
}

enum Role {
    ROLE_UNSPECIFIED = 0;
    ROLE_SYSTEM = 1;
    ROLE_USER = 2;
    ROLE_ASSISTANT = 3;
    ROLE_TOOL = 4;
}

message ToolCall {
    string id = 1;
    string name = 2;
    string arguments = 3;  // JSON string
}

message ToolDefinition {
    string name = 1;
    string description = 2;
    string parameters_json = 3;  // JSON schema as string
}

// =============================================================================
// Chat Types
// =============================================================================

message ChatOptions {
    string model = 1;
    optional float temperature = 2;
    optional uint32 max_tokens = 3;
    optional float top_p = 4;
    repeated string stop = 5;
    optional float frequency_penalty = 6;
    optional float presence_penalty = 7;
    optional uint64 seed = 8;
    optional ToolChoice tool_choice = 9;
    optional ResponseFormat response_format = 10;
    optional bool cache_prompt = 11;
    optional ReasoningConfig reasoning = 12;
}

message ToolChoice {
    oneof choice {
        bool auto = 1;       // true = auto
        bool none = 2;       // true = none
        bool required = 3;   // true = required
        string function = 4; // specific function name
    }
}

message ResponseFormat {
    oneof format {
        bool text = 1;
        bool json_object = 2;
        string json_schema = 3;  // JSON schema as string
    }
}

message ReasoningConfig {
    optional ReasoningEffort effort = 1;
    optional uint32 max_tokens = 2;
    optional bool exclude_from_output = 3;
}

enum ReasoningEffort {
    REASONING_EFFORT_UNSPECIFIED = 0;
    REASONING_EFFORT_LOW = 1;
    REASONING_EFFORT_MEDIUM = 2;
    REASONING_EFFORT_HIGH = 3;
}

message ChatRequest {
    repeated Message messages = 1;
    repeated ToolDefinition tools = 2;
    ChatOptions options = 3;
}

message ChatResponse {
    string content = 1;
    optional string reasoning = 2;
    repeated ToolCall tool_calls = 3;
    optional Usage usage = 4;
    optional string model = 5;
    FinishReason finish_reason = 6;
}

message ChatEvent {
    oneof event {
        string content = 1;
        string reasoning = 2;
        ToolCallStart tool_call_start = 3;
        ToolCallDelta tool_call_delta = 4;
        Usage usage = 5;
        bool done = 6;
    }
}

message ToolCallStart {
    uint32 index = 1;
    string id = 2;
    string name = 3;
}

message ToolCallDelta {
    uint32 index = 1;
    string arguments = 2;
}

// =============================================================================
// Generation Types
// =============================================================================

message GenerateRequest {
    string prompt = 1;
    GenerateOptions options = 2;
}

message GenerateOptions {
    string model = 1;
    optional uint32 max_tokens = 2;
    optional float temperature = 3;
    optional float top_p = 4;
    repeated string stop_sequences = 5;
}

message GenerateResponse {
    string text = 1;
    optional Usage usage = 2;
    optional string model = 3;
    FinishReason finish_reason = 4;
}

message GenerateEvent {
    oneof event {
        string text = 1;
        bool done = 2;
    }
}

// =============================================================================
// Embedding Types
// =============================================================================

message EmbedRequest {
    string text = 1;
    string model = 2;
}

message EmbedResponse {
    repeated float values = 1;
    string model = 2;
    uint32 dimensions = 3;
}

message EmbedBatchRequest {
    repeated string texts = 1;
    string model = 2;
}

message EmbedBatchResponse {
    repeated Embedding embeddings = 1;
}

message Embedding {
    repeated float values = 1;
    string model = 2;
    uint32 dimensions = 3;
}

// =============================================================================
// NLI Types
// =============================================================================

message NliRequest {
    string premise = 1;
    string hypothesis = 2;
    string model = 3;
}

message NliResponse {
    float entailment = 1;
    float contradiction = 2;
    float neutral = 3;
    NliLabel label = 4;
}

enum NliLabel {
    NLI_LABEL_UNSPECIFIED = 0;
    NLI_LABEL_ENTAILMENT = 1;
    NLI_LABEL_CONTRADICTION = 2;
    NLI_LABEL_NEUTRAL = 3;
}

message NliBatchRequest {
    repeated NliPair pairs = 1;
    string model = 2;
}

message NliPair {
    string premise = 1;
    string hypothesis = 2;
}

message NliBatchResponse {
    repeated NliResponse results = 1;
}

// =============================================================================
// Classification Types
// =============================================================================

message ClassifyRequest {
    string text = 1;
    repeated string labels = 2;
    string model = 3;
}

message ClassifyResponse {
    map<string, float> scores = 1;
    string top_label = 2;
    float confidence = 3;
}

message StanceRequest {
    string text = 1;
    string target = 2;
    string model = 3;
}

message StanceResponse {
    float favor = 1;
    float against = 2;
    float neutral = 3;
    StanceLabel label = 4;
    string target = 5;
}

enum StanceLabel {
    STANCE_LABEL_UNSPECIFIED = 0;
    STANCE_LABEL_FAVOR = 1;
    STANCE_LABEL_AGAINST = 2;
    STANCE_LABEL_NEUTRAL = 3;
}

// =============================================================================
// Tokenization Types
// =============================================================================

message TokenCountRequest {
    string text = 1;
    string model = 2;
}

message TokenCountResponse {
    uint32 count = 1;
}

message TokenizeRequest {
    string text = 1;
    string model = 2;
}

message TokenizeResponse {
    repeated Token tokens = 1;
}

message Token {
    uint32 id = 1;
    string text = 2;
    uint32 start = 3;
    uint32 end = 4;
}

// =============================================================================
// Model Management Types
// =============================================================================

message ListModelsRequest {}

message ListModelsResponse {
    repeated ModelInfo models = 1;
}

message ModelInfo {
    string id = 1;
    string provider = 2;
    repeated ModelCapability capabilities = 3;
    optional uint32 context_window = 4;
    optional uint32 dimensions = 5;
}

enum ModelCapability {
    MODEL_CAPABILITY_UNSPECIFIED = 0;
    MODEL_CAPABILITY_CHAT = 1;
    MODEL_CAPABILITY_GENERATE = 2;
    MODEL_CAPABILITY_EMBED = 3;
    MODEL_CAPABILITY_NLI = 4;
    MODEL_CAPABILITY_CLASSIFY = 5;
}

message ModelStatusRequest {
    string model = 1;
}

message ModelStatusResponse {
    ModelStatus status = 1;
    optional string reason = 2;  // For unavailable status
}

enum ModelStatus {
    MODEL_STATUS_UNSPECIFIED = 0;
    MODEL_STATUS_AVAILABLE = 1;
    MODEL_STATUS_LOADING = 2;
    MODEL_STATUS_READY = 3;
    MODEL_STATUS_UNAVAILABLE = 4;
}

// =============================================================================
// Health Types
// =============================================================================

message HealthRequest {}

message HealthResponse {
    bool healthy = 1;
    string version = 2;
    optional string git_sha = 3;
}
```

**Step 2: Update build.rs to compile proto**

Modify `build.rs` to add proto compilation (only when server feature is enabled):

```rust
fn main() {
    // Existing vergen setup...

    // Compile protobuf (only when server or client feature is enabled)
    #[cfg(any(feature = "server", feature = "client"))]
    {
        let proto_file = "proto/ratatoskr.proto";
        if std::path::Path::new(proto_file).exists() {
            tonic_build::configure()
                .build_server(cfg!(feature = "server"))
                .build_client(cfg!(feature = "client"))
                .compile_protos(&[proto_file], &["proto"])
                .expect("Failed to compile protos");
        }
    }
}
```

**Step 3: Verify proto compiles**

Run: `cargo build --features server`
Expected: Compiles, generates `ratatoskr.v1.rs` in target directory

**Step 4: Commit**

```bash
git add proto/ratatoskr.proto build.rs
git commit -m "feat(proto): add gRPC service definition"
```

---

## Task 3: Proto Conversion Module

**Files:**
- Create: `src/server/mod.rs`
- Create: `src/server/convert.rs`

**Step 1: Create server module**

Create `src/server/mod.rs`:

```rust
//! gRPC server implementation for ratd.
//!
//! This module provides the gRPC service that exposes the ModelGateway
//! interface over the network.
//!
//! # Transport Extensibility
//!
//! Currently only TCP transport is supported. The transport layer is
//! abstracted to allow future addition of Unix socket support for
//! tighter permission control on multi-user systems.

pub mod convert;
pub mod service;

// Re-export generated proto types
pub mod proto {
    tonic::include_proto!("ratatoskr.v1");
}

pub use service::RatatoskrService;
```

**Step 2: Create conversion functions**

Create `src/server/convert.rs`:

```rust
//! Conversions between ratatoskr types and protobuf types.

use crate::{
    ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason,
    GenerateEvent, GenerateOptions, GenerateResponse, Message, ModelCapability, ModelInfo,
    ModelStatus, NliLabel, NliResult, ReasoningConfig, ReasoningEffort, ResponseFormat, Role,
    StanceLabel, StanceResult, Token, ToolCall, ToolChoice, ToolDefinition, Usage,
};

use super::proto;

// =============================================================================
// From Proto (incoming requests)
// =============================================================================

impl From<proto::Message> for Message {
    fn from(p: proto::Message) -> Self {
        let role = match proto::Role::try_from(p.role).unwrap_or(proto::Role::Unspecified) {
            proto::Role::System => Role::System,
            proto::Role::User => Role::User,
            proto::Role::Assistant => Role::Assistant,
            proto::Role::Tool => Role::Tool {
                tool_call_id: p.tool_call_id.unwrap_or_default(),
            },
            proto::Role::Unspecified => Role::User,
        };

        Message {
            role,
            content: crate::MessageContent::Text(p.content),
            tool_calls: if p.tool_calls.is_empty() {
                None
            } else {
                Some(p.tool_calls.into_iter().map(Into::into).collect())
            },
            name: p.name,
        }
    }
}

impl From<proto::ToolCall> for ToolCall {
    fn from(p: proto::ToolCall) -> Self {
        ToolCall {
            id: p.id,
            name: p.name,
            arguments: p.arguments,
        }
    }
}

impl From<proto::ToolDefinition> for ToolDefinition {
    fn from(p: proto::ToolDefinition) -> Self {
        ToolDefinition {
            name: p.name,
            description: p.description,
            parameters: serde_json::from_str(&p.parameters_json).unwrap_or_default(),
        }
    }
}

impl From<proto::ChatOptions> for ChatOptions {
    fn from(p: proto::ChatOptions) -> Self {
        ChatOptions {
            model: p.model,
            temperature: p.temperature,
            max_tokens: p.max_tokens.map(|t| t as usize),
            top_p: p.top_p,
            stop: if p.stop.is_empty() { None } else { Some(p.stop) },
            frequency_penalty: p.frequency_penalty,
            presence_penalty: p.presence_penalty,
            seed: p.seed,
            tool_choice: p.tool_choice.map(Into::into),
            response_format: p.response_format.map(Into::into),
            cache_prompt: p.cache_prompt,
            reasoning: p.reasoning.map(Into::into),
            raw_provider_options: None,
        }
    }
}

impl From<proto::ToolChoice> for ToolChoice {
    fn from(p: proto::ToolChoice) -> Self {
        match p.choice {
            Some(proto::tool_choice::Choice::Auto(true)) => ToolChoice::Auto,
            Some(proto::tool_choice::Choice::None(true)) => ToolChoice::None,
            Some(proto::tool_choice::Choice::Required(true)) => ToolChoice::Required,
            Some(proto::tool_choice::Choice::Function(name)) => ToolChoice::Function { name },
            _ => ToolChoice::Auto,
        }
    }
}

impl From<proto::ResponseFormat> for ResponseFormat {
    fn from(p: proto::ResponseFormat) -> Self {
        match p.format {
            Some(proto::response_format::Format::Text(true)) => ResponseFormat::Text,
            Some(proto::response_format::Format::JsonObject(true)) => ResponseFormat::JsonObject,
            Some(proto::response_format::Format::JsonSchema(s)) => ResponseFormat::JsonSchema {
                schema: serde_json::from_str(&s).unwrap_or_default(),
            },
            _ => ResponseFormat::Text,
        }
    }
}

impl From<proto::ReasoningConfig> for ReasoningConfig {
    fn from(p: proto::ReasoningConfig) -> Self {
        ReasoningConfig {
            effort: p.effort.and_then(|e| {
                match proto::ReasoningEffort::try_from(e).ok()? {
                    proto::ReasoningEffort::Low => Some(ReasoningEffort::Low),
                    proto::ReasoningEffort::Medium => Some(ReasoningEffort::Medium),
                    proto::ReasoningEffort::High => Some(ReasoningEffort::High),
                    proto::ReasoningEffort::Unspecified => None,
                }
            }),
            max_tokens: p.max_tokens.map(|t| t as usize),
            exclude_from_output: p.exclude_from_output,
        }
    }
}

impl From<proto::GenerateOptions> for GenerateOptions {
    fn from(p: proto::GenerateOptions) -> Self {
        GenerateOptions {
            model: p.model,
            max_tokens: p.max_tokens.map(|t| t as usize),
            temperature: p.temperature,
            top_p: p.top_p,
            stop_sequences: p.stop_sequences,
        }
    }
}

// =============================================================================
// To Proto (outgoing responses)
// =============================================================================

impl From<Message> for proto::Message {
    fn from(m: Message) -> Self {
        let (role, tool_call_id) = match m.role {
            Role::System => (proto::Role::System as i32, None),
            Role::User => (proto::Role::User as i32, None),
            Role::Assistant => (proto::Role::Assistant as i32, None),
            Role::Tool { tool_call_id } => (proto::Role::Tool as i32, Some(tool_call_id)),
        };

        proto::Message {
            role,
            content: m.content.as_text().unwrap_or_default().to_string(),
            tool_calls: m.tool_calls.unwrap_or_default().into_iter().map(Into::into).collect(),
            name: m.name,
            tool_call_id,
        }
    }
}

impl From<ToolCall> for proto::ToolCall {
    fn from(t: ToolCall) -> Self {
        proto::ToolCall {
            id: t.id,
            name: t.name,
            arguments: t.arguments,
        }
    }
}

impl From<Usage> for proto::Usage {
    fn from(u: Usage) -> Self {
        proto::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u.reasoning_tokens,
        }
    }
}

impl From<FinishReason> for proto::FinishReason {
    fn from(f: FinishReason) -> Self {
        match f {
            FinishReason::Stop => proto::FinishReason::Stop,
            FinishReason::Length => proto::FinishReason::Length,
            FinishReason::ToolCalls => proto::FinishReason::ToolCalls,
            FinishReason::ContentFilter => proto::FinishReason::ContentFilter,
        }
    }
}

impl From<ChatResponse> for proto::ChatResponse {
    fn from(r: ChatResponse) -> Self {
        proto::ChatResponse {
            content: r.content,
            reasoning: r.reasoning,
            tool_calls: r.tool_calls.into_iter().map(Into::into).collect(),
            usage: r.usage.map(Into::into),
            model: r.model,
            finish_reason: proto::FinishReason::from(r.finish_reason) as i32,
        }
    }
}

impl From<ChatEvent> for proto::ChatEvent {
    fn from(e: ChatEvent) -> Self {
        let event = match e {
            ChatEvent::Content(s) => proto::chat_event::Event::Content(s),
            ChatEvent::Reasoning(s) => proto::chat_event::Event::Reasoning(s),
            ChatEvent::ToolCallStart { index, id, name } => {
                proto::chat_event::Event::ToolCallStart(proto::ToolCallStart {
                    index: index as u32,
                    id,
                    name,
                })
            }
            ChatEvent::ToolCallDelta { index, arguments } => {
                proto::chat_event::Event::ToolCallDelta(proto::ToolCallDelta {
                    index: index as u32,
                    arguments,
                })
            }
            ChatEvent::Usage(u) => proto::chat_event::Event::Usage(u.into()),
            ChatEvent::Done => proto::chat_event::Event::Done(true),
        };
        proto::ChatEvent { event: Some(event) }
    }
}

impl From<GenerateResponse> for proto::GenerateResponse {
    fn from(r: GenerateResponse) -> Self {
        proto::GenerateResponse {
            text: r.text,
            usage: r.usage.map(Into::into),
            model: r.model,
            finish_reason: proto::FinishReason::from(r.finish_reason) as i32,
        }
    }
}

impl From<GenerateEvent> for proto::GenerateEvent {
    fn from(e: GenerateEvent) -> Self {
        let event = match e {
            GenerateEvent::Text(s) => proto::generate_event::Event::Text(s),
            GenerateEvent::Done => proto::generate_event::Event::Done(true),
        };
        proto::GenerateEvent { event: Some(event) }
    }
}

impl From<Embedding> for proto::Embedding {
    fn from(e: Embedding) -> Self {
        proto::Embedding {
            values: e.values,
            model: e.model,
            dimensions: e.dimensions as u32,
        }
    }
}

impl From<Embedding> for proto::EmbedResponse {
    fn from(e: Embedding) -> Self {
        proto::EmbedResponse {
            values: e.values,
            model: e.model,
            dimensions: e.dimensions as u32,
        }
    }
}

impl From<NliResult> for proto::NliResponse {
    fn from(r: NliResult) -> Self {
        proto::NliResponse {
            entailment: r.entailment,
            contradiction: r.contradiction,
            neutral: r.neutral,
            label: match r.label {
                NliLabel::Entailment => proto::NliLabel::Entailment as i32,
                NliLabel::Contradiction => proto::NliLabel::Contradiction as i32,
                NliLabel::Neutral => proto::NliLabel::Neutral as i32,
            },
        }
    }
}

impl From<ClassifyResult> for proto::ClassifyResponse {
    fn from(r: ClassifyResult) -> Self {
        proto::ClassifyResponse {
            scores: r.scores,
            top_label: r.top_label,
            confidence: r.confidence,
        }
    }
}

impl From<StanceResult> for proto::StanceResponse {
    fn from(r: StanceResult) -> Self {
        proto::StanceResponse {
            favor: r.favor,
            against: r.against,
            neutral: r.neutral,
            label: match r.label {
                StanceLabel::Favor => proto::StanceLabel::Favor as i32,
                StanceLabel::Against => proto::StanceLabel::Against as i32,
                StanceLabel::Neutral => proto::StanceLabel::Neutral as i32,
            },
            target: r.target,
        }
    }
}

impl From<Token> for proto::Token {
    fn from(t: Token) -> Self {
        proto::Token {
            id: t.id,
            text: t.text,
            start: t.start as u32,
            end: t.end as u32,
        }
    }
}

impl From<ModelInfo> for proto::ModelInfo {
    fn from(m: ModelInfo) -> Self {
        proto::ModelInfo {
            id: m.id,
            provider: m.provider,
            capabilities: m.capabilities.into_iter().map(|c| match c {
                ModelCapability::Chat => proto::ModelCapability::Chat as i32,
                ModelCapability::Generate => proto::ModelCapability::Generate as i32,
                ModelCapability::Embed => proto::ModelCapability::Embed as i32,
                ModelCapability::Nli => proto::ModelCapability::Nli as i32,
                ModelCapability::Classify => proto::ModelCapability::Classify as i32,
            }).collect(),
            context_window: m.context_window.map(|w| w as u32),
            dimensions: m.dimensions.map(|d| d as u32),
        }
    }
}

impl From<ModelStatus> for proto::ModelStatusResponse {
    fn from(s: ModelStatus) -> Self {
        match s {
            ModelStatus::Available => proto::ModelStatusResponse {
                status: proto::ModelStatus::Available as i32,
                reason: None,
            },
            ModelStatus::Loading => proto::ModelStatusResponse {
                status: proto::ModelStatus::Loading as i32,
                reason: None,
            },
            ModelStatus::Ready => proto::ModelStatusResponse {
                status: proto::ModelStatus::Ready as i32,
                reason: None,
            },
            ModelStatus::Unavailable { reason } => proto::ModelStatusResponse {
                status: proto::ModelStatus::Unavailable as i32,
                reason: Some(reason),
            },
        }
    }
}
```

**Step 3: Add server module to lib.rs**

Add to `src/lib.rs` (after the other module declarations):

```rust
#[cfg(feature = "server")]
pub mod server;
```

**Step 4: Verify it compiles**

Run: `cargo check --features server`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/server/mod.rs src/server/convert.rs src/lib.rs
git commit -m "feat(server): add proto conversion module"
```

---

## Task 4: gRPC Service Implementation

**Files:**
- Create: `src/server/service.rs`

**Step 1: Create the gRPC service**

Create `src/server/service.rs`:

```rust
//! gRPC service implementation.

use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use tonic::{Request, Response, Status};

use crate::ModelGateway;

use super::convert;
use super::proto;
use super::proto::ratatoskr_server::Ratatoskr;

/// gRPC service that wraps a ModelGateway implementation.
pub struct RatatoskrService<G: ModelGateway> {
    gateway: Arc<G>,
}

impl<G: ModelGateway> RatatoskrService<G> {
    pub fn new(gateway: Arc<G>) -> Self {
        Self { gateway }
    }
}

type GrpcResult<T> = Result<Response<T>, Status>;
type StreamResponse<T> = Pin<Box<dyn futures_util::Stream<Item = Result<T, Status>> + Send>>;

/// Convert RatatoskrError to tonic::Status
fn to_status(err: crate::RatatoskrError) -> Status {
    use crate::RatatoskrError;
    match err {
        RatatoskrError::ModelNotFound(m) => Status::not_found(format!("Model not found: {}", m)),
        RatatoskrError::ModelNotAvailable => Status::not_found("Model not available"),
        RatatoskrError::RateLimited { retry_after } => {
            let msg = match retry_after {
                Some(d) => format!("Rate limited, retry after {:?}", d),
                None => "Rate limited".to_string(),
            };
            Status::resource_exhausted(msg)
        }
        RatatoskrError::AuthenticationFailed => Status::unauthenticated("Authentication failed"),
        RatatoskrError::InvalidInput(msg) => Status::invalid_argument(msg),
        RatatoskrError::NotImplemented(op) => {
            Status::unimplemented(format!("Not implemented: {}", op))
        }
        RatatoskrError::Unsupported => Status::unimplemented("Operation not supported"),
        e => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl<G: ModelGateway + 'static> Ratatoskr for RatatoskrService<G> {
    // =========================================================================
    // Chat
    // =========================================================================

    async fn chat(&self, request: Request<proto::ChatRequest>) -> GrpcResult<proto::ChatResponse> {
        let req = request.into_inner();
        let messages: Vec<_> = req.messages.into_iter().map(Into::into).collect();
        let tools: Vec<_> = req.tools.into_iter().map(Into::into).collect();
        let options = req.options.map(Into::into).unwrap_or_default();

        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        let response = self
            .gateway
            .chat(&messages, tools_ref, &options)
            .await
            .map_err(to_status)?;

        Ok(Response::new(response.into()))
    }

    type ChatStreamStream = StreamResponse<proto::ChatEvent>;

    async fn chat_stream(
        &self,
        request: Request<proto::ChatRequest>,
    ) -> GrpcResult<Response<Self::ChatStreamStream>> {
        let req = request.into_inner();
        let messages: Vec<_> = req.messages.into_iter().map(Into::into).collect();
        let tools: Vec<_> = req.tools.into_iter().map(Into::into).collect();
        let options = req.options.map(Into::into).unwrap_or_default();

        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        let stream = self
            .gateway
            .chat_stream(&messages, tools_ref, &options)
            .await
            .map_err(to_status)?;

        let mapped = stream.map(|result| match result {
            Ok(event) => Ok(event.into()),
            Err(e) => Err(to_status(e)),
        });

        Ok(Response::new(Box::pin(mapped)))
    }

    // =========================================================================
    // Generate
    // =========================================================================

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> GrpcResult<proto::GenerateResponse> {
        let req = request.into_inner();
        let options = req.options.map(Into::into).unwrap_or_else(|| {
            crate::GenerateOptions::new("")
        });

        let response = self
            .gateway
            .generate(&req.prompt, &options)
            .await
            .map_err(to_status)?;

        Ok(Response::new(response.into()))
    }

    type GenerateStreamStream = StreamResponse<proto::GenerateEvent>;

    async fn generate_stream(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> GrpcResult<Response<Self::GenerateStreamStream>> {
        let req = request.into_inner();
        let options = req.options.map(Into::into).unwrap_or_else(|| {
            crate::GenerateOptions::new("")
        });

        let stream = self
            .gateway
            .generate_stream(&req.prompt, &options)
            .await
            .map_err(to_status)?;

        let mapped = stream.map(|result| match result {
            Ok(event) => Ok(event.into()),
            Err(e) => Err(to_status(e)),
        });

        Ok(Response::new(Box::pin(mapped)))
    }

    // =========================================================================
    // Embeddings
    // =========================================================================

    async fn embed(&self, request: Request<proto::EmbedRequest>) -> GrpcResult<proto::EmbedResponse> {
        let req = request.into_inner();
        let embedding = self
            .gateway
            .embed(&req.text, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(embedding.into()))
    }

    async fn embed_batch(
        &self,
        request: Request<proto::EmbedBatchRequest>,
    ) -> GrpcResult<proto::EmbedBatchResponse> {
        let req = request.into_inner();
        let texts: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .gateway
            .embed_batch(&texts, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(proto::EmbedBatchResponse {
            embeddings: embeddings.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // NLI
    // =========================================================================

    async fn infer_nli(&self, request: Request<proto::NliRequest>) -> GrpcResult<proto::NliResponse> {
        let req = request.into_inner();
        let result = self
            .gateway
            .infer_nli(&req.premise, &req.hypothesis, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    async fn infer_nli_batch(
        &self,
        request: Request<proto::NliBatchRequest>,
    ) -> GrpcResult<proto::NliBatchResponse> {
        let req = request.into_inner();
        let pairs: Vec<(&str, &str)> = req
            .pairs
            .iter()
            .map(|p| (p.premise.as_str(), p.hypothesis.as_str()))
            .collect();

        let results = self
            .gateway
            .infer_nli_batch(&pairs, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(proto::NliBatchResponse {
            results: results.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // Classification
    // =========================================================================

    async fn classify_zero_shot(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> GrpcResult<proto::ClassifyResponse> {
        let req = request.into_inner();
        let labels: Vec<&str> = req.labels.iter().map(|s| s.as_str()).collect();
        let result = self
            .gateway
            .classify_zero_shot(&req.text, &labels, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    async fn classify_stance(
        &self,
        request: Request<proto::StanceRequest>,
    ) -> GrpcResult<proto::StanceResponse> {
        let req = request.into_inner();
        let result = self
            .gateway
            .classify_stance(&req.text, &req.target, &req.model)
            .await
            .map_err(to_status)?;

        Ok(Response::new(result.into()))
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    async fn count_tokens(
        &self,
        request: Request<proto::TokenCountRequest>,
    ) -> GrpcResult<proto::TokenCountResponse> {
        let req = request.into_inner();
        let count = self
            .gateway
            .count_tokens(&req.text, &req.model)
            .map_err(to_status)?;

        Ok(Response::new(proto::TokenCountResponse {
            count: count as u32,
        }))
    }

    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> GrpcResult<proto::TokenizeResponse> {
        let req = request.into_inner();
        let tokens = self
            .gateway
            .tokenize(&req.text, &req.model)
            .map_err(to_status)?;

        Ok(Response::new(proto::TokenizeResponse {
            tokens: tokens.into_iter().map(Into::into).collect(),
        }))
    }

    // =========================================================================
    // Model Management
    // =========================================================================

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> GrpcResult<proto::ListModelsResponse> {
        let models = self.gateway.list_models();
        Ok(Response::new(proto::ListModelsResponse {
            models: models.into_iter().map(Into::into).collect(),
        }))
    }

    async fn model_status(
        &self,
        request: Request<proto::ModelStatusRequest>,
    ) -> GrpcResult<proto::ModelStatusResponse> {
        let req = request.into_inner();
        let status = self.gateway.model_status(&req.model);
        Ok(Response::new(status.into()))
    }

    // =========================================================================
    // Health
    // =========================================================================

    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> GrpcResult<proto::HealthResponse> {
        Ok(Response::new(proto::HealthResponse {
            healthy: true,
            version: crate::PKG_VERSION.to_string(),
            git_sha: Some(crate::GIT_SHA.to_string()),
        }))
    }
}
```

**Step 2: Update server/mod.rs to export service**

The module file already exports `service`, no changes needed.

**Step 3: Verify it compiles**

Run: `cargo check --features server`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/server/service.rs
git commit -m "feat(server): add gRPC service implementation"
```

---

## Task 5: Configuration Types

**Files:**
- Create: `src/server/config.rs`

**Step 1: Create config types**

Create `src/server/config.rs`:

```rust
//! Configuration loading for ratd.
//!
//! Configuration is loaded from TOML files with the following resolution order:
//! 1. `--config <path>` (CLI flag)
//! 2. `~/.ratatoskr/config.toml` (user)
//! 3. `/etc/ratatoskr/config.toml` (system)
//!
//! Secrets are loaded separately with mandatory permission checks:
//! 1. `~/.ratatoskr/secrets.toml` (user, must be 0600)
//! 2. `/etc/ratatoskr/secrets.toml` (system, must be 0600)

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{RatatoskrError, Result};

/// Server configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
}

/// Server network configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Address to bind to (default: 127.0.0.1:9741).
    #[serde(default = "default_address")]
    pub address: String,
    #[serde(default)]
    pub limits: LimitsConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            address: default_address(),
            limits: LimitsConfig::default(),
        }
    }
}

fn default_address() -> String {
    "127.0.0.1:9741".to_string()
}

/// Resource limits.
#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: default_max_concurrent(),
            request_timeout_secs: default_timeout(),
        }
    }
}

fn default_max_concurrent() -> usize {
    100
}

fn default_timeout() -> u64 {
    30
}

/// Provider configurations.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openrouter: Option<ApiProviderConfig>,
    #[serde(default)]
    pub anthropic: Option<ApiProviderConfig>,
    #[serde(default)]
    pub openai: Option<ApiProviderConfig>,
    #[serde(default)]
    pub google: Option<ApiProviderConfig>,
    #[serde(default)]
    pub ollama: Option<OllamaConfig>,
    #[serde(default)]
    pub huggingface: Option<ApiProviderConfig>,
    #[serde(default)]
    pub local: Option<LocalConfig>,
}

/// API provider configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiProviderConfig {
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Ollama-specific configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

/// Local inference configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LocalConfig {
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    #[serde(default)]
    pub ram_budget_mb: Option<usize>,
}

fn default_device() -> String {
    "cpu".to_string()
}

/// Routing configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub chat: Option<String>,
    #[serde(default)]
    pub generate: Option<String>,
    #[serde(default)]
    pub embed: Option<String>,
    #[serde(default)]
    pub nli: Option<String>,
    #[serde(default)]
    pub classify: Option<String>,
}

/// Secrets configuration (API keys).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Secrets {
    #[serde(default)]
    pub openrouter: Option<ApiKeySecret>,
    #[serde(default)]
    pub anthropic: Option<ApiKeySecret>,
    #[serde(default)]
    pub openai: Option<ApiKeySecret>,
    #[serde(default)]
    pub google: Option<ApiKeySecret>,
    #[serde(default)]
    pub huggingface: Option<ApiKeySecret>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeySecret {
    pub api_key: String,
}

impl Config {
    /// Load configuration from the standard locations.
    ///
    /// Resolution order:
    /// 1. Explicit path (if provided)
    /// 2. ~/.ratatoskr/config.toml
    /// 3. /etc/ratatoskr/config.toml
    pub fn load(explicit_path: Option<&Path>) -> Result<Self> {
        let path = Self::resolve_config_path(explicit_path)?;
        let content = fs::read_to_string(&path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to read config file {:?}: {}", path, e))
        })?;
        toml::from_str(&content).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to parse config file {:?}: {}", path, e))
        })
    }

    /// Resolve the config file path.
    fn resolve_config_path(explicit: Option<&Path>) -> Result<PathBuf> {
        if let Some(path) = explicit {
            if path.exists() {
                return Ok(path.to_path_buf());
            }
            return Err(RatatoskrError::Configuration(format!(
                "Config file not found: {:?}",
                path
            )));
        }

        // User config
        if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".ratatoskr").join("config.toml");
            if user_config.exists() {
                return Ok(user_config);
            }
        }

        // System config
        let system_config = PathBuf::from("/etc/ratatoskr/config.toml");
        if system_config.exists() {
            return Ok(system_config);
        }

        Err(RatatoskrError::Configuration(
            "No config file found. Create ~/.ratatoskr/config.toml or /etc/ratatoskr/config.toml"
                .to_string(),
        ))
    }
}

impl Secrets {
    /// Load secrets from the standard locations with permission checks.
    ///
    /// Resolution order:
    /// 1. ~/.ratatoskr/secrets.toml (if exists, must be 0600)
    /// 2. /etc/ratatoskr/secrets.toml (if exists, must be 0600)
    ///
    /// Returns empty secrets if no file exists (providers may use env vars).
    pub fn load() -> Result<Self> {
        // Try user secrets first
        if let Some(home) = dirs::home_dir() {
            let user_secrets = home.join(".ratatoskr").join("secrets.toml");
            if user_secrets.exists() {
                Self::check_permissions(&user_secrets)?;
                return Self::load_from_file(&user_secrets);
            }
        }

        // Try system secrets
        let system_secrets = PathBuf::from("/etc/ratatoskr/secrets.toml");
        if system_secrets.exists() {
            Self::check_permissions(&system_secrets)?;
            return Self::load_from_file(&system_secrets);
        }

        // No secrets file, return empty (providers can fall back to env vars)
        Ok(Secrets::default())
    }

    fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to read secrets file {:?}: {}", path, e))
        })?;
        toml::from_str(&content).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to parse secrets file {:?}: {}", path, e))
        })
    }

    /// Check that the secrets file has secure permissions (0600).
    #[cfg(unix)]
    fn check_permissions(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::metadata(path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to stat secrets file {:?}: {}", path, e))
        })?;

        let mode = metadata.permissions().mode();
        // Check that only owner has read/write (0600 or 0400)
        if mode & 0o077 != 0 {
            return Err(RatatoskrError::Configuration(format!(
                "Secrets file {:?} has insecure permissions {:o}. Must be 0600 or 0400.",
                path,
                mode & 0o777
            )));
        }

        Ok(())
    }

    #[cfg(not(unix))]
    fn check_permissions(_path: &Path) -> Result<()> {
        // Permission check not available on non-Unix platforms
        Ok(())
    }

    /// Get API key for a provider, falling back to environment variable.
    pub fn api_key(&self, provider: &str) -> Option<String> {
        let from_file = match provider {
            "openrouter" => self.openrouter.as_ref().map(|s| s.api_key.clone()),
            "anthropic" => self.anthropic.as_ref().map(|s| s.api_key.clone()),
            "openai" => self.openai.as_ref().map(|s| s.api_key.clone()),
            "google" => self.google.as_ref().map(|s| s.api_key.clone()),
            "huggingface" => self.huggingface.as_ref().map(|s| s.api_key.clone()),
            _ => None,
        };

        from_file.or_else(|| {
            let env_var = match provider {
                "openrouter" => "OPENROUTER_API_KEY",
                "anthropic" => "ANTHROPIC_API_KEY",
                "openai" => "OPENAI_API_KEY",
                "google" => "GOOGLE_API_KEY",
                "huggingface" => "HF_API_KEY",
                _ => return None,
            };
            std::env::var(env_var).ok()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config {
            server: ServerConfig::default(),
            providers: ProvidersConfig::default(),
            routing: RoutingConfig::default(),
        };
        assert_eq!(config.server.address, "127.0.0.1:9741");
        assert_eq!(config.server.limits.max_concurrent_requests, 100);
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
            [server]
            address = "0.0.0.0:9741"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.address, "0.0.0.0:9741");
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
            [server]
            address = "127.0.0.1:9741"

            [server.limits]
            max_concurrent_requests = 50
            request_timeout_secs = 60

            [providers.openrouter]
            default_model = "anthropic/claude-sonnet-4"

            [providers.ollama]
            base_url = "http://localhost:11434"

            [routing]
            chat = "openrouter"
            embed = "local"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.limits.max_concurrent_requests, 50);
        assert_eq!(
            config.providers.openrouter.as_ref().unwrap().default_model,
            Some("anthropic/claude-sonnet-4".to_string())
        );
        assert_eq!(config.routing.chat, Some("openrouter".to_string()));
    }
}
```

**Step 2: Add dirs dependency**

The `dirs` crate is already used in the local-inference feature, but we need it unconditionally for config. Update `Cargo.toml`:

```toml
dirs = { version = "5", optional = true }
```

Change to:

```toml
dirs = "5"
```

And remove it from the `local-inference` feature deps.

**Step 3: Update server/mod.rs**

Add config module:

```rust
pub mod config;
pub mod convert;
pub mod service;
```

**Step 4: Verify it compiles**

Run: `cargo test --features server config`
Expected: Tests pass

**Step 5: Commit**

```bash
git add src/server/config.rs src/server/mod.rs Cargo.toml
git commit -m "feat(server): add configuration loading with permission checks"
```

---

## Task 6: ratd Binary

**Files:**
- Create: `src/bin/ratd.rs`

**Step 1: Create the daemon binary**

Create `src/bin/ratd.rs`:

```rust
//! ratd - Ratatoskr daemon
//!
//! Serves the ModelGateway over gRPC.

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use tonic::transport::Server;

use ratatoskr::server::config::{Config, Secrets};
use ratatoskr::server::proto::ratatoskr_server::RatatoskrServer;
use ratatoskr::server::RatatoskrService;
use ratatoskr::{Ratatoskr, RatatoskrBuilder};

/// Ratatoskr daemon - unified model gateway service
#[derive(Parser)]
#[command(name = "ratd")]
#[command(version = ratatoskr::version_string())]
#[command(about = "Ratatoskr model gateway daemon")]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load configuration
    let config = Config::load(args.config.as_deref())?;
    let secrets = Secrets::load()?;

    // Build the embedded gateway from config
    let gateway = build_gateway(&config, &secrets)?;

    // Parse address
    let addr: SocketAddr = config.server.address.parse().map_err(|e| {
        ratatoskr::RatatoskrError::Configuration(format!("Invalid address: {}", e))
    })?;

    eprintln!(
        "ratd {} starting on {}",
        ratatoskr::version_string(),
        addr
    );

    // Create gRPC service
    let service = RatatoskrService::new(Arc::new(gateway));
    let server = RatatoskrServer::new(service);

    // Start server
    Server::builder()
        .add_service(server)
        .serve(addr)
        .await?;

    Ok(())
}

/// Build an EmbeddedGateway from configuration.
fn build_gateway(
    config: &Config,
    secrets: &Secrets,
) -> Result<ratatoskr::EmbeddedGateway, ratatoskr::RatatoskrError> {
    let mut builder = Ratatoskr::builder();

    // Configure API providers
    if config.providers.openrouter.is_some() {
        if let Some(key) = secrets.api_key("openrouter") {
            builder = builder.openrouter(key);
        }
    }

    if config.providers.anthropic.is_some() {
        if let Some(key) = secrets.api_key("anthropic") {
            builder = builder.anthropic(key);
        }
    }

    if config.providers.openai.is_some() {
        if let Some(key) = secrets.api_key("openai") {
            builder = builder.openai(key);
        }
    }

    if config.providers.google.is_some() {
        if let Some(key) = secrets.api_key("google") {
            builder = builder.google(key);
        }
    }

    if let Some(ref ollama) = config.providers.ollama {
        builder = builder.ollama(&ollama.base_url);
    }

    #[cfg(feature = "huggingface")]
    if config.providers.huggingface.is_some() {
        if let Some(key) = secrets.api_key("huggingface") {
            builder = builder.huggingface(key);
        }
    }

    // Configure local inference
    #[cfg(feature = "local-inference")]
    if let Some(ref local) = config.providers.local {
        use ratatoskr::{Device, LocalEmbeddingModel, LocalNliModel};

        // Set device
        let device = match local.device.as_str() {
            "cuda" => Device::Cuda { device_id: 0 },
            _ => Device::Cpu,
        };
        builder = builder.device(device);

        // Set cache dir if specified
        if let Some(ref dir) = local.models_dir {
            builder = builder.cache_dir(dir);
        }

        // Set RAM budget if specified
        if let Some(budget_mb) = local.ram_budget_mb {
            builder = builder.ram_budget(budget_mb * 1024 * 1024);
        }

        // For now, enable default local models
        // TODO: Make this configurable per-model
        builder = builder.local_embeddings(LocalEmbeddingModel::AllMiniLmL6V2);
        builder = builder.local_nli(LocalNliModel::NliDebertaV3Small);
    }

    builder.build()
}
```

**Step 2: Update Cargo.toml to include binary**

Add to `Cargo.toml`:

```toml
[[bin]]
name = "ratd"
path = "src/bin/ratd.rs"
required-features = ["server"]
```

**Step 3: Verify it compiles**

Run: `cargo build --features server --bin ratd`
Expected: Binary builds successfully

**Step 4: Commit**

```bash
git add src/bin/ratd.rs Cargo.toml
git commit -m "feat(bin): add ratd daemon binary"
```

---

## Task 7: ServiceClient (Client Library)

**Files:**
- Create: `src/client/mod.rs`
- Create: `src/client/service_client.rs`

**Step 1: Create client module**

Create `src/client/mod.rs`:

```rust
//! Client library for connecting to ratd.
//!
//! Provides `ServiceClient`, which implements `ModelGateway` by forwarding
//! calls to a remote ratd instance over gRPC.

mod service_client;

pub use service_client::ServiceClient;
```

**Step 2: Create ServiceClient**

Create `src/client/service_client.rs`:

```rust
//! ServiceClient - ModelGateway implementation that connects to ratd.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use tonic::transport::Channel;

use crate::server::proto;
use crate::server::proto::ratatoskr_client::RatatoskrClient;
use crate::{
    Capabilities, ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason,
    GenerateEvent, GenerateOptions, GenerateResponse, Message, ModelCapability, ModelGateway,
    ModelInfo, ModelStatus, NliLabel, NliResult, RatatoskrError, Result, StanceLabel, StanceResult,
    Token, ToolDefinition, Usage,
};

/// A ModelGateway client that connects to a remote ratd server.
pub struct ServiceClient {
    inner: RatatoskrClient<Channel>,
}

impl ServiceClient {
    /// Connect to a ratd server at the given address.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = ServiceClient::connect("http://127.0.0.1:9741").await?;
    /// ```
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let inner = RatatoskrClient::connect(addr.clone())
            .await
            .map_err(|e| RatatoskrError::Http(format!("Failed to connect to {}: {}", addr, e)))?;
        Ok(Self { inner })
    }
}

/// Convert tonic::Status to RatatoskrError
fn from_status(status: tonic::Status) -> RatatoskrError {
    match status.code() {
        tonic::Code::NotFound => RatatoskrError::ModelNotAvailable,
        tonic::Code::ResourceExhausted => RatatoskrError::RateLimited { retry_after: None },
        tonic::Code::Unauthenticated => RatatoskrError::AuthenticationFailed,
        tonic::Code::InvalidArgument => RatatoskrError::InvalidInput(status.message().to_string()),
        tonic::Code::Unimplemented => {
            RatatoskrError::NotImplemented(Box::leak(status.message().to_string().into_boxed_str()))
        }
        _ => RatatoskrError::Http(status.message().to_string()),
    }
}

// =============================================================================
// Proto to Native conversions (for responses)
// =============================================================================

impl From<proto::ChatResponse> for ChatResponse {
    fn from(p: proto::ChatResponse) -> Self {
        ChatResponse {
            content: p.content,
            reasoning: p.reasoning,
            tool_calls: p.tool_calls.into_iter().map(|t| crate::ToolCall {
                id: t.id,
                name: t.name,
                arguments: t.arguments,
            }).collect(),
            usage: p.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u.reasoning_tokens,
            }),
            model: p.model,
            finish_reason: match proto::FinishReason::try_from(p.finish_reason) {
                Ok(proto::FinishReason::Stop) => FinishReason::Stop,
                Ok(proto::FinishReason::Length) => FinishReason::Length,
                Ok(proto::FinishReason::ToolCalls) => FinishReason::ToolCalls,
                Ok(proto::FinishReason::ContentFilter) => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            },
        }
    }
}

impl From<proto::ChatEvent> for ChatEvent {
    fn from(p: proto::ChatEvent) -> Self {
        match p.event {
            Some(proto::chat_event::Event::Content(s)) => ChatEvent::Content(s),
            Some(proto::chat_event::Event::Reasoning(s)) => ChatEvent::Reasoning(s),
            Some(proto::chat_event::Event::ToolCallStart(t)) => ChatEvent::ToolCallStart {
                index: t.index as usize,
                id: t.id,
                name: t.name,
            },
            Some(proto::chat_event::Event::ToolCallDelta(t)) => ChatEvent::ToolCallDelta {
                index: t.index as usize,
                arguments: t.arguments,
            },
            Some(proto::chat_event::Event::Usage(u)) => ChatEvent::Usage(Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u.reasoning_tokens,
            }),
            Some(proto::chat_event::Event::Done(_)) => ChatEvent::Done,
            None => ChatEvent::Done,
        }
    }
}

impl From<proto::GenerateResponse> for GenerateResponse {
    fn from(p: proto::GenerateResponse) -> Self {
        GenerateResponse {
            text: p.text,
            usage: p.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u.reasoning_tokens,
            }),
            model: p.model,
            finish_reason: match proto::FinishReason::try_from(p.finish_reason) {
                Ok(proto::FinishReason::Stop) => FinishReason::Stop,
                Ok(proto::FinishReason::Length) => FinishReason::Length,
                Ok(proto::FinishReason::ToolCalls) => FinishReason::ToolCalls,
                Ok(proto::FinishReason::ContentFilter) => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            },
        }
    }
}

impl From<proto::GenerateEvent> for GenerateEvent {
    fn from(p: proto::GenerateEvent) -> Self {
        match p.event {
            Some(proto::generate_event::Event::Text(s)) => GenerateEvent::Text(s),
            Some(proto::generate_event::Event::Done(_)) => GenerateEvent::Done,
            None => GenerateEvent::Done,
        }
    }
}

impl From<proto::EmbedResponse> for Embedding {
    fn from(p: proto::EmbedResponse) -> Self {
        Embedding {
            values: p.values,
            model: p.model,
            dimensions: p.dimensions as usize,
        }
    }
}

impl From<proto::Embedding> for Embedding {
    fn from(p: proto::Embedding) -> Self {
        Embedding {
            values: p.values,
            model: p.model,
            dimensions: p.dimensions as usize,
        }
    }
}

impl From<proto::NliResponse> for NliResult {
    fn from(p: proto::NliResponse) -> Self {
        NliResult {
            entailment: p.entailment,
            contradiction: p.contradiction,
            neutral: p.neutral,
            label: match proto::NliLabel::try_from(p.label) {
                Ok(proto::NliLabel::Entailment) => NliLabel::Entailment,
                Ok(proto::NliLabel::Contradiction) => NliLabel::Contradiction,
                Ok(proto::NliLabel::Neutral) => NliLabel::Neutral,
                _ => NliLabel::Neutral,
            },
        }
    }
}

impl From<proto::ClassifyResponse> for ClassifyResult {
    fn from(p: proto::ClassifyResponse) -> Self {
        ClassifyResult {
            scores: p.scores,
            top_label: p.top_label,
            confidence: p.confidence,
        }
    }
}

impl From<proto::StanceResponse> for StanceResult {
    fn from(p: proto::StanceResponse) -> Self {
        StanceResult {
            favor: p.favor,
            against: p.against,
            neutral: p.neutral,
            label: match proto::StanceLabel::try_from(p.label) {
                Ok(proto::StanceLabel::Favor) => StanceLabel::Favor,
                Ok(proto::StanceLabel::Against) => StanceLabel::Against,
                Ok(proto::StanceLabel::Neutral) => StanceLabel::Neutral,
                _ => StanceLabel::Neutral,
            },
            target: p.target,
        }
    }
}

impl From<proto::Token> for Token {
    fn from(p: proto::Token) -> Self {
        Token {
            id: p.id,
            text: p.text,
            start: p.start as usize,
            end: p.end as usize,
        }
    }
}

impl From<proto::ModelInfo> for ModelInfo {
    fn from(p: proto::ModelInfo) -> Self {
        ModelInfo {
            id: p.id,
            provider: p.provider,
            capabilities: p.capabilities.into_iter().filter_map(|c| {
                match proto::ModelCapability::try_from(c).ok()? {
                    proto::ModelCapability::Chat => Some(ModelCapability::Chat),
                    proto::ModelCapability::Generate => Some(ModelCapability::Generate),
                    proto::ModelCapability::Embed => Some(ModelCapability::Embed),
                    proto::ModelCapability::Nli => Some(ModelCapability::Nli),
                    proto::ModelCapability::Classify => Some(ModelCapability::Classify),
                    proto::ModelCapability::Unspecified => None,
                }
            }).collect(),
            context_window: p.context_window.map(|w| w as usize),
            dimensions: p.dimensions.map(|d| d as usize),
        }
    }
}

impl From<proto::ModelStatusResponse> for ModelStatus {
    fn from(p: proto::ModelStatusResponse) -> Self {
        match proto::ModelStatus::try_from(p.status) {
            Ok(proto::ModelStatus::Available) => ModelStatus::Available,
            Ok(proto::ModelStatus::Loading) => ModelStatus::Loading,
            Ok(proto::ModelStatus::Ready) => ModelStatus::Ready,
            Ok(proto::ModelStatus::Unavailable) => ModelStatus::Unavailable {
                reason: p.reason.unwrap_or_else(|| "Unknown".to_string()),
            },
            _ => ModelStatus::Unavailable {
                reason: "Unknown status".to_string(),
            },
        }
    }
}

// =============================================================================
// Native to Proto conversions (for requests)
// =============================================================================

fn to_proto_message(m: &Message) -> proto::Message {
    let (role, tool_call_id) = match &m.role {
        crate::Role::System => (proto::Role::System as i32, None),
        crate::Role::User => (proto::Role::User as i32, None),
        crate::Role::Assistant => (proto::Role::Assistant as i32, None),
        crate::Role::Tool { tool_call_id } => (proto::Role::Tool as i32, Some(tool_call_id.clone())),
    };

    proto::Message {
        role,
        content: m.content.as_text().unwrap_or_default().to_string(),
        tool_calls: m.tool_calls.as_ref().map(|tcs| {
            tcs.iter().map(|t| proto::ToolCall {
                id: t.id.clone(),
                name: t.name.clone(),
                arguments: t.arguments.clone(),
            }).collect()
        }).unwrap_or_default(),
        name: m.name.clone(),
        tool_call_id,
    }
}

fn to_proto_tool(t: &ToolDefinition) -> proto::ToolDefinition {
    proto::ToolDefinition {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters_json: serde_json::to_string(&t.parameters).unwrap_or_default(),
    }
}

fn to_proto_chat_options(o: &ChatOptions) -> proto::ChatOptions {
    proto::ChatOptions {
        model: o.model.clone(),
        temperature: o.temperature,
        max_tokens: o.max_tokens.map(|t| t as u32),
        top_p: o.top_p,
        stop: o.stop.clone().unwrap_or_default(),
        frequency_penalty: o.frequency_penalty,
        presence_penalty: o.presence_penalty,
        seed: o.seed,
        tool_choice: o.tool_choice.as_ref().map(|tc| {
            let choice = match tc {
                crate::ToolChoice::Auto => proto::tool_choice::Choice::Auto(true),
                crate::ToolChoice::None => proto::tool_choice::Choice::None(true),
                crate::ToolChoice::Required => proto::tool_choice::Choice::Required(true),
                crate::ToolChoice::Function { name } => proto::tool_choice::Choice::Function(name.clone()),
            };
            proto::ToolChoice { choice: Some(choice) }
        }),
        response_format: o.response_format.as_ref().map(|rf| {
            let format = match rf {
                crate::ResponseFormat::Text => proto::response_format::Format::Text(true),
                crate::ResponseFormat::JsonObject => proto::response_format::Format::JsonObject(true),
                crate::ResponseFormat::JsonSchema { schema } => {
                    proto::response_format::Format::JsonSchema(serde_json::to_string(schema).unwrap_or_default())
                }
            };
            proto::ResponseFormat { format: Some(format) }
        }),
        cache_prompt: o.cache_prompt,
        reasoning: o.reasoning.as_ref().map(|r| proto::ReasoningConfig {
            effort: r.effort.map(|e| match e {
                crate::ReasoningEffort::Low => proto::ReasoningEffort::Low as i32,
                crate::ReasoningEffort::Medium => proto::ReasoningEffort::Medium as i32,
                crate::ReasoningEffort::High => proto::ReasoningEffort::High as i32,
            }),
            max_tokens: r.max_tokens.map(|t| t as u32),
            exclude_from_output: r.exclude_from_output,
        }),
    }
}

fn to_proto_generate_options(o: &GenerateOptions) -> proto::GenerateOptions {
    proto::GenerateOptions {
        model: o.model.clone(),
        max_tokens: o.max_tokens.map(|t| t as u32),
        temperature: o.temperature,
        top_p: o.top_p,
        stop_sequences: o.stop_sequences.clone(),
    }
}

// =============================================================================
// ModelGateway Implementation
// =============================================================================

#[async_trait]
impl ModelGateway for ServiceClient {
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let request = proto::ChatRequest {
            messages: messages.iter().map(to_proto_message).collect(),
            tools: tools.map(|ts| ts.iter().map(to_proto_tool).collect()).unwrap_or_default(),
            options: Some(to_proto_chat_options(options)),
        };

        let response = self.inner.clone().chat_stream(request).await.map_err(from_status)?;
        let stream = response.into_inner().map(|result| {
            result.map(Into::into).map_err(from_status)
        });

        Ok(Box::pin(stream))
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let request = proto::ChatRequest {
            messages: messages.iter().map(to_proto_message).collect(),
            tools: tools.map(|ts| ts.iter().map(to_proto_tool).collect()).unwrap_or_default(),
            options: Some(to_proto_chat_options(options)),
        };

        let response = self.inner.clone().chat(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    fn capabilities(&self) -> Capabilities {
        // ServiceClient supports all capabilities (the server determines actual availability)
        Capabilities {
            chat: true,
            chat_streaming: true,
            generate: true,
            tool_use: true,
            embeddings: true,
            nli: true,
            classification: true,
            stance: true,
            token_counting: true,
            local_inference: false, // Server may have local inference, but client doesn't know
        }
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        let request = proto::EmbedRequest {
            text: text.to_string(),
            model: model.to_string(),
        };
        let response = self.inner.clone().embed(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        let request = proto::EmbedBatchRequest {
            texts: texts.iter().map(|s| s.to_string()).collect(),
            model: model.to_string(),
        };
        let response = self.inner.clone().embed_batch(request).await.map_err(from_status)?;
        Ok(response.into_inner().embeddings.into_iter().map(Into::into).collect())
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        let request = proto::NliRequest {
            premise: premise.to_string(),
            hypothesis: hypothesis.to_string(),
            model: model.to_string(),
        };
        let response = self.inner.clone().infer_nli(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn infer_nli_batch(&self, pairs: &[(&str, &str)], model: &str) -> Result<Vec<NliResult>> {
        let request = proto::NliBatchRequest {
            pairs: pairs.iter().map(|(p, h)| proto::NliPair {
                premise: p.to_string(),
                hypothesis: h.to_string(),
            }).collect(),
            model: model.to_string(),
        };
        let response = self.inner.clone().infer_nli_batch(request).await.map_err(from_status)?;
        Ok(response.into_inner().results.into_iter().map(Into::into).collect())
    }

    async fn classify_zero_shot(&self, text: &str, labels: &[&str], model: &str) -> Result<ClassifyResult> {
        let request = proto::ClassifyRequest {
            text: text.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            model: model.to_string(),
        };
        let response = self.inner.clone().classify_zero_shot(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn classify_stance(&self, text: &str, target: &str, model: &str) -> Result<StanceResult> {
        let request = proto::StanceRequest {
            text: text.to_string(),
            target: target.to_string(),
            model: model.to_string(),
        };
        let response = self.inner.clone().classify_stance(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        // Synchronous call - we need to block on async
        // This is a limitation; consider making count_tokens async in the trait
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| RatatoskrError::Configuration("No tokio runtime".into()))?;

        let request = proto::TokenCountRequest {
            text: text.to_string(),
            model: model.to_string(),
        };

        let mut client = self.inner.clone();
        let result = rt.block_on(async {
            client.count_tokens(request).await
        }).map_err(from_status)?;

        Ok(result.into_inner().count as usize)
    }

    fn tokenize(&self, text: &str, model: &str) -> Result<Vec<Token>> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| RatatoskrError::Configuration("No tokio runtime".into()))?;

        let request = proto::TokenizeRequest {
            text: text.to_string(),
            model: model.to_string(),
        };

        let mut client = self.inner.clone();
        let result = rt.block_on(async {
            client.tokenize(request).await
        }).map_err(from_status)?;

        Ok(result.into_inner().tokens.into_iter().map(Into::into).collect())
    }

    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse> {
        let request = proto::GenerateRequest {
            prompt: prompt.to_string(),
            options: Some(to_proto_generate_options(options)),
        };
        let response = self.inner.clone().generate(request).await.map_err(from_status)?;
        Ok(response.into_inner().into())
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateEvent>> + Send>>> {
        let request = proto::GenerateRequest {
            prompt: prompt.to_string(),
            options: Some(to_proto_generate_options(options)),
        };

        let response = self.inner.clone().generate_stream(request).await.map_err(from_status)?;
        let stream = response.into_inner().map(|result| {
            result.map(Into::into).map_err(from_status)
        });

        Ok(Box::pin(stream))
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let rt = match tokio::runtime::Handle::try_current() {
            Ok(rt) => rt,
            Err(_) => return vec![],
        };

        let mut client = self.inner.clone();
        let result = rt.block_on(async {
            client.list_models(proto::ListModelsRequest {}).await
        });

        match result {
            Ok(response) => response.into_inner().models.into_iter().map(Into::into).collect(),
            Err(_) => vec![],
        }
    }

    fn model_status(&self, model: &str) -> ModelStatus {
        let rt = match tokio::runtime::Handle::try_current() {
            Ok(rt) => rt,
            Err(_) => return ModelStatus::Unavailable { reason: "No runtime".into() },
        };

        let request = proto::ModelStatusRequest {
            model: model.to_string(),
        };

        let mut client = self.inner.clone();
        let result = rt.block_on(async {
            client.model_status(request).await
        });

        match result {
            Ok(response) => response.into_inner().into(),
            Err(e) => ModelStatus::Unavailable { reason: e.message().to_string() },
        }
    }
}
```

**Step 3: Add client module to lib.rs and export**

Add to `src/lib.rs`:

```rust
#[cfg(feature = "client")]
pub mod client;

// Re-export client types
#[cfg(feature = "client")]
pub use client::ServiceClient;
```

**Step 4: Verify it compiles**

Run: `cargo check --features server,client`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/client/mod.rs src/client/service_client.rs src/lib.rs
git commit -m "feat(client): add ServiceClient implementing ModelGateway"
```

---

## Task 8: rat CLI Binary

**Files:**
- Create: `src/bin/rat.rs`

**Step 1: Create the CLI binary**

Create `src/bin/rat.rs`:

```rust
//! rat - Ratatoskr CLI client
//!
//! Control and test interface for ratd.

use clap::{Parser, Subcommand};
use ratatoskr::client::ServiceClient;
use ratatoskr::ModelGateway;

/// Ratatoskr CLI client
#[derive(Parser)]
#[command(name = "rat")]
#[command(version = ratatoskr::version_string())]
#[command(about = "Ratatoskr model gateway client")]
struct Args {
    /// Server address
    #[arg(short, long, env = "RATD_ADDRESS", default_value = "http://127.0.0.1:9741")]
    address: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check service health
    Health,

    /// List available models
    Models,

    /// Get status of a specific model
    Status {
        /// Model name
        model: String,
    },

    /// Generate embeddings for text
    Embed {
        /// Text to embed
        text: String,
        /// Model to use
        #[arg(short, long, default_value = "sentence-transformers/all-MiniLM-L6-v2")]
        model: String,
    },

    /// Perform NLI inference
    Nli {
        /// Premise text
        premise: String,
        /// Hypothesis text
        hypothesis: String,
        /// Model to use
        #[arg(short, long, default_value = "cross-encoder/nli-deberta-v3-base")]
        model: String,
    },

    /// Chat with a model
    Chat {
        /// User message
        message: String,
        /// Model to use
        #[arg(short, long, default_value = "anthropic/claude-sonnet-4")]
        model: String,
    },

    /// Count tokens in text
    Tokens {
        /// Text to tokenize
        text: String,
        /// Model for tokenizer
        #[arg(short, long, default_value = "claude-sonnet")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let client = ServiceClient::connect(&args.address).await?;

    match args.command {
        Command::Health => {
            // Use a simple operation to check health
            let models = client.list_models();
            println!("Connected to ratd at {}", args.address);
            println!("Available providers: {}", models.len());
            println!("Status: healthy");
        }

        Command::Models => {
            let models = client.list_models();
            if models.is_empty() {
                println!("No models available");
            } else {
                for model in models {
                    println!(
                        "{} ({}): {:?}",
                        model.id, model.provider, model.capabilities
                    );
                }
            }
        }

        Command::Status { model } => {
            let status = client.model_status(&model);
            println!("{}: {:?}", model, status);
        }

        Command::Embed { text, model } => {
            let embedding = client.embed(&text, &model).await?;
            println!("Model: {}", embedding.model);
            println!("Dimensions: {}", embedding.dimensions);
            println!("Values: [{:.4}, {:.4}, ... ({} total)]",
                embedding.values.first().unwrap_or(&0.0),
                embedding.values.get(1).unwrap_or(&0.0),
                embedding.values.len()
            );
        }

        Command::Nli { premise, hypothesis, model } => {
            let result = client.infer_nli(&premise, &hypothesis, &model).await?;
            println!("Label: {:?}", result.label);
            println!("Entailment: {:.4}", result.entailment);
            println!("Contradiction: {:.4}", result.contradiction);
            println!("Neutral: {:.4}", result.neutral);
        }

        Command::Chat { message, model } => {
            use ratatoskr::ChatOptions;
            let response = client.chat(
                &[ratatoskr::Message::user(&message)],
                None,
                &ChatOptions::default().model(&model),
            ).await?;
            println!("{}", response.content);
        }

        Command::Tokens { text, model } => {
            let count = client.count_tokens(&text, &model)?;
            println!("{} tokens", count);
        }
    }

    Ok(())
}
```

**Step 2: Update Cargo.toml to include binary**

Add to `Cargo.toml`:

```toml
[[bin]]
name = "rat"
path = "src/bin/rat.rs"
required-features = ["client"]
```

**Step 3: Verify it compiles**

Run: `cargo build --features client --bin rat`
Expected: Binary builds successfully

**Step 4: Commit**

```bash
git add src/bin/rat.rs Cargo.toml
git commit -m "feat(bin): add rat CLI client"
```

---

## Task 9: systemd Unit File

**Files:**
- Create: `contrib/systemd/ratd.service`

**Step 1: Create contrib directory and unit file**

Create `contrib/systemd/ratd.service`:

```ini
# Ratatoskr Model Gateway Daemon
#
# Installation:
#   sudo cp contrib/systemd/ratd.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now ratd
#
# Logs:
#   journalctl -u ratd -f

[Unit]
Description=Ratatoskr Model Gateway
Documentation=https://github.com/emesal/ratatoskr
After=network.target

[Service]
Type=simple
User=ratatoskr
Group=ratatoskr

# Binary location (adjust if installed elsewhere)
ExecStart=/usr/local/bin/ratd

# Restart on failure
Restart=on-failure
RestartSec=5

# Environment
Environment=RUST_LOG=info

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes

# Allow reading config from home and /etc
ReadOnlyPaths=/etc/ratatoskr
ReadOnlyPaths=/home

# Allow writing to cache directory
ReadWritePaths=/var/cache/ratatoskr

# Resource limits
LimitNOFILE=65535
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

**Step 2: Commit**

```bash
git add contrib/systemd/ratd.service
git commit -m "docs: add systemd unit file for ratd"
```

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/service_test.rs`

**Step 1: Create integration test**

Create `tests/service_test.rs`:

```rust
//! Integration tests for gRPC service mode.
//!
//! These tests start an in-process ratd server and connect with a ServiceClient.

#![cfg(all(feature = "server", feature = "client"))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use ratatoskr::client::ServiceClient;
use ratatoskr::server::proto::ratatoskr_server::RatatoskrServer;
use ratatoskr::server::RatatoskrService;
use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};
use tokio::net::TcpListener;
use tonic::transport::Server;

/// Find an available port for testing.
async fn find_available_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap()
}

/// Start a test server and return the address.
async fn start_test_server() -> String {
    let addr = find_available_port().await;
    let addr_str = format!("http://{}", addr);

    // Create a minimal gateway for testing
    // This will fail on actual chat calls but works for health/model queries
    let gateway = Ratatoskr::builder()
        .openrouter("test-key") // Fake key, won't make real calls
        .build()
        .expect("Failed to build test gateway");

    let service = RatatoskrService::new(Arc::new(gateway));
    let server = RatatoskrServer::new(service);

    tokio::spawn(async move {
        Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    addr_str
}

#[tokio::test]
async fn test_client_connect() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await;
    assert!(client.is_ok(), "Failed to connect: {:?}", client.err());
}

#[tokio::test]
async fn test_client_list_models() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await.unwrap();

    let models = client.list_models();
    // Should have at least the openrouter provider
    assert!(!models.is_empty());
}

#[tokio::test]
async fn test_client_model_status() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await.unwrap();

    let status = client.model_status("nonexistent-model");
    // Should return some status (likely Ready for API-only mode)
    match status {
        ratatoskr::ModelStatus::Ready => {}
        ratatoskr::ModelStatus::Available => {}
        ratatoskr::ModelStatus::Unavailable { .. } => {}
        ratatoskr::ModelStatus::Loading => {}
    }
}

#[tokio::test]
async fn test_client_capabilities() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await.unwrap();

    let caps = client.capabilities();
    assert!(caps.chat, "ServiceClient should report chat capability");
    assert!(caps.chat_streaming, "ServiceClient should report streaming capability");
}

// Note: Actual chat/embed/etc calls require live API keys and are not tested here.
// Add #[ignore] tests for live testing similar to other live_test.rs files.

#[tokio::test]
#[ignore = "requires ratd running with valid API keys"]
async fn test_live_chat() {
    let client = ServiceClient::connect("http://127.0.0.1:9741")
        .await
        .expect("Connect to ratd (is it running?)");

    let response = client
        .chat(
            &[Message::user("Say hello in exactly 3 words.")],
            None,
            &ChatOptions::default().model("anthropic/claude-sonnet-4"),
        )
        .await
        .expect("Chat request failed");

    assert!(!response.content.is_empty());
    println!("Response: {}", response.content);
}
```

**Step 2: Verify tests compile and pass**

Run: `cargo test --features server,client service_test`
Expected: Tests pass (the ones that don't require live connections)

**Step 3: Commit**

```bash
git add tests/service_test.rs
git commit -m "test: add gRPC service integration tests"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1: Update README with service mode info**

Add a "Service Mode" section to `README.md` after the existing usage examples.

**Step 2: Update AGENTS.md project structure**

Update the project structure in `AGENTS.md` to include new directories:

```
src/
├── bin/
│   ├── ratd.rs          # Daemon entry point
│   └── rat.rs           # CLI client entry point
├── server/              # gRPC server (feature: server)
│   ├── mod.rs
│   ├── config.rs        # Config + secrets loading
│   ├── convert.rs       # Proto conversions
│   └── service.rs       # gRPC service impl
├── client/              # gRPC client (feature: client)
│   ├── mod.rs
│   └── service_client.rs
└── proto/
    └── ratatoskr.proto  # gRPC service definition
```

**Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: add service mode documentation"
```

---

## Task 12: Final Verification

**Step 1: Run full test suite**

Run: `just pre-push`
Expected: All tests pass, clippy clean, formatted

**Step 2: Build both binaries**

Run: `cargo build --release --features server,client`
Expected: Both `ratd` and `rat` binaries built

**Step 3: Manual smoke test (optional)**

```bash
# Terminal 1: Start server with minimal config
echo '[providers.openrouter]' > /tmp/config.toml
OPENROUTER_API_KEY=your-key ./target/release/ratd --config /tmp/config.toml

# Terminal 2: Test client
./target/release/rat health
./target/release/rat models
```

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: phase 5 cleanup and fixes"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Add gRPC dependencies | `Cargo.toml` |
| 2 | Create protobuf definition | `proto/ratatoskr.proto`, `build.rs` |
| 3 | Proto conversion module | `src/server/convert.rs` |
| 4 | gRPC service implementation | `src/server/service.rs` |
| 5 | Configuration types | `src/server/config.rs` |
| 6 | ratd binary | `src/bin/ratd.rs` |
| 7 | ServiceClient | `src/client/service_client.rs` |
| 8 | rat CLI binary | `src/bin/rat.rs` |
| 9 | systemd unit file | `contrib/systemd/ratd.service` |
| 10 | Integration tests | `tests/service_test.rs` |
| 11 | Documentation | `README.md`, `AGENTS.md` |
| 12 | Final verification | — |
