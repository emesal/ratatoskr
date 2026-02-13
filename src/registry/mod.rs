//! Model registry — centralised model metadata with layered merge.
//!
//! The registry holds [`ModelMetadata`] entries from multiple sources:
//! 1. **Embedded seed** — compiled-in JSON, always available
//! 2. **Cached remote** — curated data from `emesal/ratatoskr-registry`
//! 3. **Live provider data** — runtime API queries
//!
//! Merge priority: later data overrides earlier (live > remote > embedded).
//! Within a merge, per-parameter overrides are applied (not whole-entry replacement).
//!
//! See [`remote`] for the fetch/cache mechanism.

pub mod remote;

use std::collections::{BTreeMap, HashMap};

use tracing::{info, warn};

use crate::{ModelCapability, ModelMetadata};

/// Centralised model metadata registry.
///
/// Thread-safe reads are ensured by storing data in a `HashMap` keyed by model ID.
/// For concurrent access (e.g., from `EmbeddedGateway`), wrap in `Arc<RwLock<>>`.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: HashMap<String, ModelMetadata>,
    /// Autoconfig presets: `(cost_tier, capability_string) → model_id`.
    presets: BTreeMap<String, BTreeMap<String, String>>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a model entry, replacing any existing entry with the same ID.
    pub fn insert(&mut self, metadata: ModelMetadata) {
        self.entries.insert(metadata.info.id.clone(), metadata);
    }

    /// Get metadata for a model by ID.
    pub fn get(&self, model: &str) -> Option<&ModelMetadata> {
        self.entries.get(model)
    }

    /// List all model metadata entries.
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.entries.values().collect()
    }

    /// Merge a single entry: update existing parameters or insert new.
    ///
    /// If the model already exists, parameters from `incoming` override the
    /// existing entry's parameters (per-key), and scalar fields (pricing,
    /// max_output_tokens, context_window) are replaced if present in `incoming`.
    /// If the model doesn't exist, it's inserted directly.
    pub fn merge(&mut self, incoming: ModelMetadata) {
        let id = incoming.info.id.clone();
        if let Some(existing) = self.entries.get_mut(&id) {
            // Merge parameters (incoming overrides existing per-key).
            for (name, avail) in incoming.parameters {
                existing.parameters.insert(name, avail);
            }
            // Scalar overrides: only replace if incoming has a value.
            if incoming.pricing.is_some() {
                existing.pricing = incoming.pricing;
            }
            if incoming.max_output_tokens.is_some() {
                existing.max_output_tokens = incoming.max_output_tokens;
            }
            if incoming.info.context_window.is_some() {
                existing.info.context_window = incoming.info.context_window;
            }
            // Merge capabilities (union).
            for cap in incoming.info.capabilities {
                if !existing.info.capabilities.contains(&cap) {
                    existing.info.capabilities.push(cap);
                }
            }
        } else {
            self.entries.insert(id, incoming);
        }
    }

    /// Merge a batch of entries.
    pub fn merge_batch(&mut self, batch: Vec<ModelMetadata>) {
        for entry in batch {
            self.merge(entry);
        }
    }

    /// Filter entries by capability.
    pub fn filter_by_capability(&self, capability: ModelCapability) -> Vec<&ModelMetadata> {
        self.entries
            .values()
            .filter(|m| m.info.capabilities.contains(&capability))
            .collect()
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ===== Presets =====

    /// Look up a preset model ID for the given tier and capability.
    pub fn preset(&self, tier: &str, capability: &str) -> Option<&str> {
        self.presets
            .get(tier)
            .and_then(|cap_map| cap_map.get(capability))
            .map(String::as_str)
    }

    /// Get all presets for a cost tier.
    pub fn presets_for_tier(&self, tier: &str) -> Option<&BTreeMap<String, String>> {
        self.presets.get(tier)
    }

    /// Insert or update a single preset.
    pub fn set_preset(&mut self, tier: &str, capability: &str, model_id: &str) {
        self.presets
            .entry(tier.to_owned())
            .or_default()
            .insert(capability.to_owned(), model_id.to_owned());
    }

    /// Merge incoming presets (incoming overrides existing per-key).
    pub fn merge_presets(&mut self, incoming: BTreeMap<String, BTreeMap<String, String>>) {
        for (tier, cap_map) in incoming {
            let existing = self.presets.entry(tier).or_default();
            for (capability, model_id) in cap_map {
                existing.insert(capability, model_id);
            }
        }
    }

    /// Warn about preset model IDs that aren't present in the registry entries.
    ///
    /// This is advisory — the model may still be resolvable at runtime via provider APIs.
    fn validate_presets(&self) {
        for (tier, cap_map) in &self.presets {
            for (capability, model_id) in cap_map {
                if !self.entries.contains_key(model_id) {
                    warn!(
                        tier = %tier,
                        capability = %capability,
                        model_id = %model_id,
                        "preset references model not in registry"
                    );
                }
            }
        }
    }

    /// Load cached remote registry data and merge into this registry.
    ///
    /// Reads from the local cache file only (no network I/O). If the file
    /// is missing or corrupt, this is a no-op (logged at warn level).
    pub fn with_cached_remote(mut self, path: &std::path::Path) -> Self {
        if let Some(payload) = remote::load_cached(path) {
            info!(
                count = payload.models.len(),
                "loaded cached remote registry"
            );
            self.merge_batch(payload.models);
            self.merge_presets(payload.presets);
            self.validate_presets();
        }
        self
    }

    /// Create a registry pre-populated with the embedded seed data.
    ///
    /// The seed contains a curated set of well-known models with parameter
    /// metadata, pricing, and capabilities. It's always available as a
    /// fallback when live provider APIs are unreachable.
    pub fn with_embedded_seed() -> Self {
        let mut registry = Self::new();
        match crate::registry::remote::parse_payload(EMBEDDED_SEED) {
            Ok(payload) => {
                for entry in payload.models {
                    registry.insert(entry);
                }
                registry.merge_presets(payload.presets);
                registry.validate_presets();
            }
            Err(e) => {
                // This should never happen — seed is compiled in and tested.
                // Log the error but don't panic; an empty registry is usable.
                warn!(error = %e, "failed to parse embedded model seed");
            }
        }
        registry
    }
}

/// Raw JSON seed data compiled into the binary.
const EMBEDDED_SEED: &str = include_str!("seed.json");
