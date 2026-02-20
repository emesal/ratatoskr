//! `rat-registry` — maintainer CLI for managing the model registry.
//!
//! Embeds `EmbeddedGateway` to fetch model metadata directly from providers
//! (no ratd dependency). Reads/writes the `RemoteRegistry` JSON format using
//! the same serde types as ratatoskr proper.
//!
//! Build: `cargo build --bin rat-registry --features registry-tool`

use std::path::{Path, PathBuf};
use std::{env, fs, process};

use clap::{Parser, Subcommand};
use dialoguer::Confirm;

use ratatoskr::registry::remote::RemoteRegistry;
use ratatoskr::{EmbeddedGateway, ModelGateway, ModelRegistry, PricingInfo, Ratatoskr};

// ── CLI ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "rat-registry", about = "manage the ratatoskr model registry")]
struct Args {
    /// path to registry.json
    #[arg(
        long,
        env = "REGISTRY_PATH",
        default_value = "../registry/registry.json"
    )]
    path: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// model management
    #[command(subcommand)]
    Model(ModelCommand),
    /// preset management
    #[command(subcommand)]
    Preset(PresetCommand),
}

#[derive(Subcommand)]
enum ModelCommand {
    /// fetch metadata from providers and add a model to the registry
    Add {
        /// model identifier (e.g. "anthropic/claude-sonnet-4")
        model_id: String,
    },
    /// list registered models
    List,
    /// remove a model from the registry
    Remove {
        /// model identifier
        model_id: String,
    },
    /// re-fetch metadata for all models from providers and merge in-place
    Refresh,
}

#[derive(Subcommand)]
enum PresetCommand {
    /// set a preset entry: tier → slot → model (with optional generation parameter defaults)
    Set {
        /// cost tier (e.g. "free", "budget", "premium", or any custom tier)
        tier: String,
        /// capability slot (e.g. "chat", "agentic", "embed")
        slot: String,
        /// model identifier (must already be in the registry)
        model_id: String,
        #[arg(long, help = "default temperature (0.0–2.0)")]
        temperature: Option<f32>,
        #[arg(long, help = "default top_p nucleus sampling (0.0–1.0)")]
        top_p: Option<f32>,
        #[arg(long, help = "default top_k")]
        top_k: Option<usize>,
        #[arg(long, help = "default max output tokens")]
        max_tokens: Option<usize>,
        #[arg(long, help = "default frequency penalty")]
        frequency_penalty: Option<f32>,
        #[arg(long, help = "default presence penalty")]
        presence_penalty: Option<f32>,
        #[arg(long, help = "default random seed")]
        seed: Option<u64>,
    },
    /// display preset table
    List,
    /// remove an entire cost tier from presets
    Remove {
        /// cost tier to remove
        tier: String,
    },
}

// ── registry I/O ────────────────────────────────────────────────────

fn load_registry(path: &Path) -> Result<RemoteRegistry, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

fn save_registry(path: &Path, registry: &RemoteRegistry) -> Result<(), Box<dyn std::error::Error>> {
    let content = serde_json::to_string_pretty(registry)?;
    fs::write(path, format!("{content}\n"))?;
    Ok(())
}

// ── gateway ─────────────────────────────────────────────────────────

fn build_gateway() -> Result<EmbeddedGateway, Box<dyn std::error::Error>> {
    let mut builder = Ratatoskr::builder();

    if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        builder = builder.openrouter(Some(key));
    }
    if let Ok(key) = env::var("HF_API_KEY") {
        builder = builder.huggingface(key);
    }

    Ok(builder.build()?)
}

// ── helpers ─────────────────────────────────────────────────────────

/// collect the set of known capability slots from existing presets.
fn known_slots(registry: &RemoteRegistry) -> std::collections::BTreeSet<String> {
    registry
        .presets
        .values()
        .flat_map(|slots| slots.keys().cloned())
        .collect()
}

/// check whether a model id is referenced by any preset, returning details.
fn preset_references(registry: &RemoteRegistry, model_id: &str) -> Vec<String> {
    let mut refs = Vec::new();
    for (tier, slots) in &registry.presets {
        for (slot, entry) in slots {
            if entry.model() == model_id {
                refs.push(format!("{tier}.{slot}"));
            }
        }
    }
    refs
}

/// confirm a prompt with the user; returns false if declined.
fn confirm(prompt: &str) -> bool {
    Confirm::new()
        .with_prompt(prompt)
        .default(false)
        .interact()
        .unwrap_or(false)
}

fn format_pricing(pricing: &Option<PricingInfo>) -> String {
    match pricing {
        Some(p) => {
            let parts: Vec<String> = [
                p.prompt_cost_per_mtok.map(|v| format!("${v}/Mtok in")),
                p.completion_cost_per_mtok.map(|v| format!("${v}/Mtok out")),
            ]
            .into_iter()
            .flatten()
            .collect();
            if parts.is_empty() {
                "—".to_string()
            } else {
                parts.join(", ")
            }
        }
        None => "—".to_string(),
    }
}

// ── commands ────────────────────────────────────────────────────────

async fn model_add(path: &Path, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = load_registry(path)?;

    // check for existing entry
    if registry.models.iter().any(|m| m.info.id == model_id) {
        if !confirm(&format!("model '{model_id}' already exists — overwrite?")) {
            println!("aborted.");
            return Ok(());
        }
        registry.models.retain(|m| m.info.id != model_id);
    }

    // fetch metadata from providers
    println!("fetching metadata for '{model_id}'...");
    let gateway = build_gateway()?;
    let metadata = gateway.fetch_model_metadata(model_id).await?;

    // insert sorted by id
    let pos = registry
        .models
        .binary_search_by(|m| m.info.id.as_str().cmp(model_id))
        .unwrap_or_else(|i| i);
    registry.models.insert(pos, metadata);

    save_registry(path, &registry)?;
    println!(
        "added '{model_id}' ({} models total)",
        registry.models.len()
    );
    Ok(())
}

fn model_list(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let registry = load_registry(path)?;

    if registry.models.is_empty() {
        println!("no models registered.");
        return Ok(());
    }

    // header
    println!(
        "{:<45} {:<12} {:<20} {:>8}  PRICING",
        "MODEL", "PROVIDER", "CAPABILITIES", "CONTEXT"
    );
    println!("{}", "─".repeat(110));

    for m in &registry.models {
        let caps = m
            .info
            .capabilities
            .iter()
            .map(|c| format!("{c:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(", ");
        let ctx = m
            .info
            .context_window
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "—".to_string());

        println!(
            "{:<45} {:<12} {:<20} {:>8}  {}",
            m.info.id,
            m.info.provider,
            caps,
            ctx,
            format_pricing(&m.pricing),
        );
    }

    println!("\n{} models", registry.models.len());
    Ok(())
}

fn model_remove(path: &Path, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = load_registry(path)?;

    if !registry.models.iter().any(|m| m.info.id == model_id) {
        eprintln!("model '{model_id}' not found in registry.");
        process::exit(1);
    }

    // warn if referenced by presets
    let refs = preset_references(&registry, model_id);
    let prompt = if refs.is_empty() {
        format!("remove '{model_id}'?")
    } else {
        format!(
            "remove '{model_id}'? (referenced by preset(s): {})",
            refs.join(", ")
        )
    };

    if !confirm(&prompt) {
        println!("aborted.");
        return Ok(());
    }

    registry.models.retain(|m| m.info.id != model_id);
    save_registry(path, &registry)?;
    println!("removed '{model_id}'.");
    Ok(())
}

fn preset_set(
    path: &Path,
    tier: &str,
    slot: &str,
    model_id: &str,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    use ratatoskr::{PresetEntry, PresetParameters};

    let mut registry = load_registry(path)?;

    // model must exist in registry
    if !registry.models.iter().any(|m| m.info.id == model_id) {
        eprintln!("model '{model_id}' not in registry — add it first.");
        process::exit(1);
    }

    // warn on novel capability slot
    let slots = known_slots(&registry);
    if !slots.is_empty()
        && !slots.contains(slot)
        && !confirm(&format!(
            "slot '{slot}' doesn't exist yet (known: {}). add it?",
            slots.iter().cloned().collect::<Vec<_>>().join(", ")
        ))
    {
        println!("aborted.");
        return Ok(());
    }

    // warn if tier has no presets yet
    if !registry.presets.contains_key(tier)
        && !confirm(&format!("tier '{tier}' has no presets yet. create it?"))
    {
        println!("aborted.");
        return Ok(());
    }

    let params = PresetParameters {
        temperature,
        top_p,
        top_k,
        max_tokens,
        frequency_penalty,
        presence_penalty,
        seed,
        ..Default::default()
    };

    let entry = if params.is_empty() {
        PresetEntry::Bare(model_id.to_owned())
    } else {
        PresetEntry::WithParams {
            model: model_id.to_owned(),
            parameters: params,
        }
    };

    registry
        .presets
        .entry(tier.to_owned())
        .or_default()
        .insert(slot.to_string(), entry);

    save_registry(path, &registry)?;
    println!("set {tier}.{slot} = {model_id}");
    Ok(())
}

fn preset_list(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let registry = load_registry(path)?;

    if registry.presets.is_empty() {
        println!("no presets configured.");
        return Ok(());
    }

    // collect all slots across tiers for column headers
    let all_slots: Vec<String> = known_slots(&registry).into_iter().collect();

    // header
    print!("{:<12}", "TIER");
    for slot in &all_slots {
        print!(" {:<40}", slot.to_uppercase());
    }
    println!();
    println!("{}", "─".repeat(12 + all_slots.len() * 41));

    // rows — `*` suffix marks presets with parameter defaults
    for (tier, slots) in &registry.presets {
        print!("{tier:<12}");
        for slot in &all_slots {
            let cell = match slots.get(slot) {
                None => "—".to_owned(),
                Some(entry) => {
                    if entry.parameters().is_some() {
                        format!("{}*", entry.model())
                    } else {
                        entry.model().to_owned()
                    }
                }
            };
            print!(" {:<40}", cell);
        }
        println!();
    }

    Ok(())
}

fn preset_remove(path: &Path, tier: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = load_registry(path)?;

    if !registry.presets.contains_key(tier) {
        eprintln!("tier '{tier}' has no presets.");
        process::exit(1);
    }

    let slot_count = registry.presets[tier].len();
    if !confirm(&format!("remove tier '{tier}' ({slot_count} slot(s))?")) {
        println!("aborted.");
        return Ok(());
    }

    registry.presets.remove(tier);
    save_registry(path, &registry)?;
    println!("removed tier '{tier}'.");
    Ok(())
}

async fn model_refresh(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = load_registry(path)?;

    if registry.models.is_empty() {
        println!("no models in registry — nothing to refresh.");
        return Ok(());
    }

    let total = registry.models.len();
    println!("refreshing {total} models...");

    let gateway = build_gateway()?;

    // collect model IDs to iterate (avoid borrow conflict with registry)
    let model_ids: Vec<String> = registry.models.iter().map(|m| m.info.id.clone()).collect();

    let mut successes = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    // load existing models into ModelRegistry for merge semantics
    let mut model_reg = ModelRegistry::default();
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

    // extract merged models back into sorted vec, preserving presets
    let mut merged_models: Vec<_> = model_reg.list().into_iter().cloned().collect();
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

// ── main ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();
    let path = &args.path;

    let result = match args.command {
        Command::Model(ModelCommand::Add { ref model_id }) => model_add(path, model_id).await,
        Command::Model(ModelCommand::List) => model_list(path),
        Command::Model(ModelCommand::Remove { ref model_id }) => model_remove(path, model_id),
        Command::Model(ModelCommand::Refresh) => model_refresh(path).await,
        Command::Preset(PresetCommand::Set {
            tier,
            ref slot,
            ref model_id,
            temperature,
            top_p,
            top_k,
            max_tokens,
            frequency_penalty,
            presence_penalty,
            seed,
        }) => preset_set(
            path,
            &tier,
            slot,
            model_id,
            temperature,
            top_p,
            top_k,
            max_tokens,
            frequency_penalty,
            presence_penalty,
            seed,
        ),
        Command::Preset(PresetCommand::List) => preset_list(path),
        Command::Preset(PresetCommand::Remove { tier }) => preset_remove(path, &tier),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
