# Service Mode

Service mode lets multiple processes share a single gateway instance over gRPC. The daemon (`ratd`) wraps an `EmbeddedGateway` behind gRPC handlers; clients connect via `ServiceClient` which implements `ModelGateway` transparently.

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  chibi   │  │  orlog   │  │  rat CLI │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └──────┬──────┴─────────────┘
            │  gRPC (default :9741)
      ┌─────▼─────┐
      │   ratd    │
      │ (daemon)  │
      └─────┬─────┘
            │
      ┌─────▼──────────────┐
      │  EmbeddedGateway   │
      │  ProviderRegistry  │
      └────────────────────┘
```

## Setup

### 1. Build

```bash
# server (ratd daemon)
cargo build --release --features server

# client (rat CLI + ServiceClient library)
cargo build --release --features client

# both
cargo build --release --features server,client
```

### 2. Configuration

ratd uses two TOML files: a config file for settings and a secrets file for API keys.

#### Config file

Searched in order:
1. `--config <path>` (CLI flag)
2. `~/.ratatoskr/config.toml`
3. `/etc/ratatoskr/config.toml`

```toml
# ~/.ratatoskr/config.toml

[server]
address = "127.0.0.1:9741"    # default

[server.limits]
max_concurrent_requests = 100  # default
request_timeout_secs = 30      # default

# Enable providers by listing them. An empty table is enough to
# activate a provider — the API key comes from secrets or env vars.

[providers.openrouter]
# default_model = "anthropic/claude-sonnet-4"

[providers.anthropic]

[providers.ollama]
# base_url = "http://localhost:11434"  # default
```

#### Secrets file

Searched in order:
1. `~/.ratatoskr/secrets.toml`
2. `/etc/ratatoskr/secrets.toml`

**Must have mode 0600 or 0400** — ratd refuses to start if group/other bits are set.

```toml
# ~/.ratatoskr/secrets.toml  (chmod 600)

[openrouter]
api_key = "sk-or-..."

[anthropic]
api_key = "sk-ant-..."

[openai]
api_key = "sk-..."

[google]
api_key = "..."

[huggingface]
api_key = "hf_..."
```

#### Environment variables

As a fallback, API keys can also be set via environment variables:

| Provider     | Environment variable    |
|-------------|------------------------|
| OpenRouter  | `OPENROUTER_API_KEY`   |
| Anthropic   | `ANTHROPIC_API_KEY`    |
| OpenAI      | `OPENAI_API_KEY`       |
| Google      | `GOOGLE_API_KEY`       |
| HuggingFace | `HF_API_KEY`           |

### 3. Start the daemon

```bash
ratd                           # uses default config paths
ratd --config /path/to/config.toml
```

### 4. systemd (optional)

A systemd unit file with security hardening is provided at `contrib/systemd/ratd.service`:

```bash
sudo cp contrib/systemd/ratd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ratd
```

The unit expects a `ratatoskr` system user. See the unit file for details on sandboxing, resource limits, and path protections.

## Client Library

`ServiceClient` implements `ModelGateway`, so it's a drop-in replacement for `EmbeddedGateway`:

```rust
use ratatoskr::{ServiceClient, ModelGateway, ChatOptions, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ServiceClient::connect("http://127.0.0.1:9741").await?;

    let response = client
        .chat(
            &[Message::user("hello!")],
            None,
            &ChatOptions::default().model("anthropic/claude-sonnet-4"),
        )
        .await?;

    println!("{}", response.content);
    Ok(())
}
```

Add to your `Cargo.toml`:

```toml
ratatoskr = { version = "0.1", features = ["client"] }
```

The `ServiceClient` forwards all `ModelGateway` methods (chat, embeddings, NLI, classification, generation, tokenization) over gRPC. The server determines actual provider availability.

## CLI (`rat`)

```bash
rat health                              # check connectivity
rat models                              # list available models
rat status <model>                      # check model status
rat chat "hello!" -m anthropic/claude-sonnet-4
rat embed "some text" -m sentence-transformers/all-MiniLM-L6-v2
rat nli "it rained" "the ground is wet" -m cross-encoder/nli-deberta-v3-base
rat tokens "count me" -m claude-sonnet
```

Connect to a different address:

```bash
rat --address http://10.0.0.5:9741 health
RATD_ADDRESS=http://10.0.0.5:9741 rat health
```

## Feature Flags

| Feature  | Provides                                              |
|----------|-------------------------------------------------------|
| `server` | `ratd` binary, `RatatoskrService`, config loading     |
| `client` | `rat` binary, `ServiceClient`                         |

Both features share the proto types and conversion module (`server::convert`).
