//! ratd — Ratatoskr daemon.
//!
//! Serves the [`ModelGateway`](ratatoskr::ModelGateway) over gRPC,
//! enabling multiple clients to share a single gateway instance.

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use tonic::transport::Server;
use tracing::info;

use ratatoskr::Ratatoskr;
use ratatoskr::server::RatatoskrService;
use ratatoskr::server::config::{Config, Secrets};
use ratatoskr::server::proto::ratatoskr_server::RatatoskrServer;

/// Ratatoskr daemon — unified model gateway service.
#[derive(Parser)]
#[command(name = "ratd")]
#[command(version = ratatoskr::PKG_VERSION)]
#[command(about = "Ratatoskr model gateway daemon")]
struct Args {
    /// Path to configuration file.
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
    let addr: SocketAddr =
        config.server.address.parse().map_err(|e| {
            ratatoskr::RatatoskrError::Configuration(format!("Invalid address: {e}"))
        })?;

    info!(version = ratatoskr::version_string(), %addr, "ratd starting");

    // Create gRPC service and start server
    let service = RatatoskrService::new(Arc::new(gateway));
    let server = RatatoskrServer::new(service);

    Server::builder().add_service(server).serve(addr).await?;

    Ok(())
}

/// Build an [`EmbeddedGateway`](ratatoskr::EmbeddedGateway) from configuration.
fn build_gateway(
    config: &Config,
    secrets: &Secrets,
) -> Result<ratatoskr::EmbeddedGateway, ratatoskr::RatatoskrError> {
    let mut builder = Ratatoskr::builder();

    // Configure API providers — only register when section is present AND key is available
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

        let device = match local.device.as_str() {
            "cuda" => Device::Cuda { device_id: 0 },
            _ => Device::Cpu,
        };
        builder = builder.device(device);

        if let Some(ref dir) = local.models_dir {
            builder = builder.cache_dir(dir);
        }

        if let Some(budget_mb) = local.ram_budget_mb {
            builder = builder.ram_budget(budget_mb * 1024 * 1024);
        }

        // Enable default local models
        // TODO: make this configurable per-model in config.toml
        builder = builder.local_embeddings(LocalEmbeddingModel::AllMiniLmL6V2);
        builder = builder.local_nli(LocalNliModel::NliDebertaV3Small);
    }

    // Apply routing preferences from config
    builder = builder.routing(config.routing.clone());

    // Apply parameter discovery config
    if config.discovery.enabled {
        builder = builder.discovery(config.discovery.clone().into());
    } else {
        builder = builder.disable_parameter_discovery();
    }

    builder.build()
}
