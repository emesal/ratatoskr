# Phase 5: Service Mode Design

> Turn ratatoskr into a shared gateway service that multiple clients can connect to.

## Overview

Phase 5 adds service mode — a daemon (`ratd`) that exposes the full `ModelGateway` interface over gRPC, allowing multiple clients (chibi, orlog, others) to share local models and API connections.

```
┌─────────────────────────────────────────────────────────────┐
│                          ratd                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   ProviderRegistry                     │  │
│  │   LocalEmbedding | LocalNLI | LlmChat | HuggingFace   │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    gRPC Server                         │  │
│  │              (tonic, TCP:9741)                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           chibi          orlog         rat CLI
```

## Binaries

Two separate binaries for privilege separation:

| Binary | Purpose | Typical user |
|--------|---------|--------------|
| `ratd` | Daemon, loads models, serves gRPC | System service or background process |
| `rat` | Client CLI for control and testing | Interactive user |

## Transport

**TCP on port 9741** (mnemonic: 9RAT) by default.

```rust
// Transport abstraction for future extensibility (Unix socket, etc.)
//
// EXTENSIBILITY: This trait exists to support alternative transports
// in the future (Unix sockets for tighter permission control, etc.).
// Implementations should be added here rather than changing the gRPC
// service layer. See architecture.md for rationale.
pub trait Transport: Send + Sync {
    // ...
}
```

Initial implementation: TCP only. Unix socket support can be added later via this abstraction without changing the protocol layer.

## Protocol

gRPC via `tonic` and `prost`. Protobuf provides:
- Strong typing across the wire
- Built-in streaming with backpressure
- Efficient binary serialization
- Good tooling (grpcurl, generated clients if we ever need non-Rust)

### Service Definition

```protobuf
syntax = "proto3";
package ratatoskr.v1;

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
```

Streaming variants (`ChatStream`, `GenerateStream`) are separate RPCs from their non-streaming counterparts — cleaner API, self-documenting.

### Error Handling

Use gRPC status codes with error details:

| Ratatoskr Error | gRPC Status | Notes |
|-----------------|-------------|-------|
| `ModelNotAvailable` | `NOT_FOUND` | Model not loaded or doesn't exist |
| `RateLimited` | `RESOURCE_EXHAUSTED` | Include retry-after in details |
| `ProviderError` | `INTERNAL` | Upstream API failure |
| `InvalidRequest` | `INVALID_ARGUMENT` | Bad input |
| `NotImplemented` | `UNIMPLEMENTED` | Capability not available |

Rich error details via `google.rpc.Status` pattern when needed (e.g., rate limit retry-after).

## Configuration

### Location Resolution

**Config file** (first found wins):
1. `--config <path>` (CLI flag)
2. `~/.ratatoskr/config.toml`
3. `/etc/ratatoskr/config.toml`

**Secrets file** (first found wins, must be mode 0600):
1. `~/.ratatoskr/secrets.toml`
2. `/etc/ratatoskr/secrets.toml`

User secrets override system secrets. Permission check is mandatory — `ratd` refuses to start if secrets file exists with wrong permissions.

### Config Format

```toml
# ~/.ratatoskr/config.toml

[server]
address = "127.0.0.1:9741"
# Future: socket = "/run/ratatoskr.sock"

[server.limits]
max_concurrent_requests = 100
request_timeout_secs = 30

[providers.openrouter]
# API key comes from secrets.toml
default_model = "anthropic/claude-sonnet-4"

[providers.ollama]
base_url = "http://localhost:11434"

[providers.local]
device = "cpu"  # or "cuda"
models_dir = "~/.cache/ratatoskr/models"
ram_budget_mb = 2048

[routing]
chat = "openrouter"
embed = "local"
nli = "local"
```

### Secrets Format

```toml
# ~/.ratatoskr/secrets.toml (mode 0600!)

[openrouter]
api_key = "sk-or-..."

[huggingface]
api_key = "hf_..."

[anthropic]
api_key = "sk-ant-..."
```

## Client Configuration

The `rat` CLI uses simple defaults with overrides:

| Priority | Source | Example |
|----------|--------|---------|
| 1 | CLI flag | `rat --address 192.168.1.10:9741 status` |
| 2 | Environment | `RATD_ADDRESS=192.168.1.10:9741` |
| 3 | Default | `127.0.0.1:9741` |

No config file for the client — keeps it simple.

## Service Management

### systemd

```ini
# /etc/systemd/system/ratd.service

[Unit]
Description=Ratatoskr Model Gateway
After=network.target

[Service]
Type=simple
User=ratatoskr
Group=ratatoskr
ExecStart=/usr/local/bin/ratd
Restart=on-failure
RestartSec=5

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes
ReadWritePaths=/var/cache/ratatoskr

[Install]
WantedBy=multi-user.target
```

Installation:
```bash
sudo useradd --system --no-create-home ratatoskr
sudo cp ratd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ratd
```

Future: OpenBSD rc.d, launchd plist as needed.

## CLI Interface

### ratd (daemon)

```bash
ratd                          # Start with default config resolution
ratd --config /path/to/config.toml
ratd --version
```

Daemon logs to stderr, suitable for systemd journal capture.

### rat (client)

```bash
# Service control
rat status                    # Connection status, loaded models
rat health                    # Health check

# Model operations (for testing/debugging)
rat embed "Hello world" --model all-MiniLM-L6-v2
rat chat "What is 2+2?" --model anthropic/claude-sonnet-4
rat nli "The cat sat" "An animal rested" --model nli-deberta-v3-base
rat tokenize "Hello world" --model claude-sonnet

# Model management
rat models list               # List available models
rat models status <model>     # Check specific model status
```

## Implementation Structure

```
src/
├── bin/
│   ├── ratd.rs              # Daemon entry point
│   └── rat.rs               # Client CLI entry point
├── server/
│   ├── mod.rs
│   ├── config.rs            # Config + secrets loading
│   ├── service.rs           # gRPC service implementation
│   └── transport.rs         # Transport abstraction (TCP, future Unix socket)
├── client/
│   ├── mod.rs
│   └── service_client.rs    # ServiceClient impl of ModelGateway
└── proto/
    └── ratatoskr.proto      # gRPC service definition
```

### Feature Flags

```toml
[features]
default = ["embedded"]        # Existing behavior
server = ["dep:tonic", "dep:prost"]
client = ["dep:tonic", "dep:prost"]
full = ["server", "client", "embedded"]
```

- `embedded` — Current behavior, direct provider access
- `server` — Builds `ratd`
- `client` — Builds `rat` and `ServiceClient`
- `full` — Everything

### ServiceClient

Implements `ModelGateway` by forwarding to gRPC:

```rust
pub struct ServiceClient {
    inner: ratatoskr::v1::ratatoskr_client::RatatoskrClient<tonic::transport::Channel>,
}

#[async_trait]
impl ModelGateway for ServiceClient {
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        options: &ChatOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let request = to_proto_chat_request(messages, tools, options);
        let response = self.inner.clone().chat_stream(request).await?;
        Ok(Box::pin(response.into_inner().map(|r| from_proto_chat_event(r?))))
    }

    // ... other methods
}
```

## Testing Strategy

1. **Unit tests** — Config parsing, proto conversions
2. **Integration tests** — Spin up `ratd` in-process, connect with `ServiceClient`, verify round-trip
3. **CLI tests** — Verify `rat` commands parse correctly, mock server responses

## Migration Path

Existing code uses `EmbeddedGateway` directly:

```rust
// Before (embedded only)
let gateway = Ratatoskr::builder()
    .openrouter(api_key)
    .build()?;

// After (service mode)
let gateway = Ratatoskr::connect("127.0.0.1:9741").await?;

// Or auto-detect
let gateway = Ratatoskr::auto()
    .service_address("127.0.0.1:9741")
    .fallback_embedded(|b| b.openrouter(api_key))
    .build()
    .await?;
```

The `ModelGateway` trait is unchanged — consumers don't know or care whether they're talking to embedded providers or a remote service.

## Open Questions (Deferred)

These are explicitly out of scope for initial Phase 5:

1. **Hot reload** — Config changes require restart for now
2. **Multi-tenancy** — Single set of API keys per service
3. **Metrics/observability** — Phase 6 concern
4. **Authentication** — Trust local connections for now

## Deliverables

- [ ] Protobuf service definition (`proto/ratatoskr.proto`)
- [ ] gRPC server implementation (`server/`)
- [ ] Config + secrets loading with permission checks
- [ ] `ratd` binary
- [ ] `ServiceClient` implementing `ModelGateway`
- [ ] `rat` CLI binary
- [ ] systemd unit file (`contrib/systemd/ratd.service`)
- [ ] Integration tests
- [ ] Documentation updates
