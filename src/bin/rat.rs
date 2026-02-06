//! rat — ratatoskr CLI client
//!
//! Control and test interface for ratd.

use clap::{Parser, Subcommand};
use ratatoskr::client::ServiceClient;
use ratatoskr::{ModelGateway, ModelMetadata, ParameterAvailability};

/// Ratatoskr CLI client
#[derive(Parser)]
#[command(name = "rat")]
#[command(version = ratatoskr::PKG_VERSION)]
#[command(about = "Ratatoskr model gateway client")]
struct Args {
    /// Server address
    #[arg(
        short,
        long,
        env = "RATD_ADDRESS",
        default_value = "http://127.0.0.1:9741"
    )]
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

    /// Fetch and display model metadata
    Metadata {
        /// Model identifier (e.g., "anthropic/claude-sonnet-4")
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = ServiceClient::connect(&args.address).await?;

    match args.command {
        Command::Health => {
            let models = client.list_models();
            println!("connected to ratd at {}", args.address);
            println!("available providers: {}", models.len());
            println!("status: healthy");
        }

        Command::Models => {
            let models = client.list_models();
            if models.is_empty() {
                println!("no models available");
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
            println!("{model}: {status:?}");
        }

        Command::Embed { text, model } => {
            let embedding = client.embed(&text, &model).await?;
            println!("model: {}", embedding.model);
            println!("dimensions: {}", embedding.dimensions);
            println!(
                "values: [{:.4}, {:.4}, ... ({} total)]",
                embedding.values.first().unwrap_or(&0.0),
                embedding.values.get(1).unwrap_or(&0.0),
                embedding.values.len()
            );
        }

        Command::Nli {
            premise,
            hypothesis,
            model,
        } => {
            let result = client.infer_nli(&premise, &hypothesis, &model).await?;
            println!("label: {:?}", result.label);
            println!("entailment: {:.4}", result.entailment);
            println!("contradiction: {:.4}", result.contradiction);
            println!("neutral: {:.4}", result.neutral);
        }

        Command::Chat { message, model } => {
            use ratatoskr::ChatOptions;
            let response = client
                .chat(
                    &[ratatoskr::Message::user(&message)],
                    None,
                    &ChatOptions::default().model(&model),
                )
                .await?;
            println!("{}", response.content);
        }

        Command::Tokens { text, model } => {
            let count = client.count_tokens(&text, &model)?;
            println!("{count} tokens");
        }

        Command::Metadata { model } => {
            // Fetch first (ensures cache is populated), then read from cache/registry.
            let metadata = client.fetch_model_metadata(&model).await?;
            print_metadata(&metadata);
        }
    }

    Ok(())
}

/// Display model metadata in a readable format.
fn print_metadata(m: &ModelMetadata) {
    println!("model:          {}", m.info.id);
    println!("provider:       {}", m.info.provider);
    println!(
        "capabilities:   {}",
        m.info
            .capabilities
            .iter()
            .map(|c| format!("{c:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(", ")
    );

    if let Some(ctx) = m.info.context_window {
        println!("context window: {ctx}");
    }
    if let Some(max) = m.max_output_tokens {
        println!("max output:     {max}");
    }

    if let Some(ref pricing) = m.pricing {
        if let Some(prompt) = pricing.prompt_cost_per_mtok {
            println!("input cost:     ${prompt}/Mtok");
        }
        if let Some(completion) = pricing.completion_cost_per_mtok {
            println!("output cost:    ${completion}/Mtok");
        }
    }

    if !m.parameters.is_empty() {
        println!("\nparameters:");
        let mut params: Vec<_> = m.parameters.iter().collect();
        params.sort_by_key(|(name, _)| name.as_str().to_string());
        for (name, avail) in params {
            let desc = match avail {
                ParameterAvailability::Mutable { range } => {
                    let parts: Vec<String> = [
                        range.min.map(|v| format!("min={v}")),
                        range.max.map(|v| format!("max={v}")),
                        range.default.map(|v| format!("default={v}")),
                    ]
                    .into_iter()
                    .flatten()
                    .collect();
                    if parts.is_empty() {
                        "mutable".to_string()
                    } else {
                        format!("mutable ({})", parts.join(", "))
                    }
                }
                ParameterAvailability::ReadOnly { value } => format!("read-only = {value}"),
                ParameterAvailability::Opaque => "opaque".to_string(),
                ParameterAvailability::Unsupported => "unsupported".to_string(),
            };
            println!("  {name}: {desc}");
        }
    }
}
