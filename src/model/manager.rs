//! Model manager for lazy loading and caching.
//!
//! Provides thread-safe lazy loading of embedding and NLI models with
//! double-checked locking to ensure models are loaded at most once.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::error::{RatatoskrError, Result};
use crate::providers::{FastEmbedProvider, LocalEmbeddingModel, LocalNliModel, OnnxNliProvider};

use super::Device;

/// Configuration for the model manager.
#[derive(Debug, Clone)]
pub struct ModelManagerConfig {
    /// Cache directory for downloaded models.
    pub cache_dir: PathBuf,

    /// Default device for inference.
    pub default_device: Device,

    /// Maximum RAM budget in bytes for loaded models.
    /// None means unlimited.
    pub ram_budget: Option<usize>,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            cache_dir: std::env::var("RATATOSKR_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    dirs::cache_dir()
                        .unwrap_or_else(|| PathBuf::from(".cache"))
                        .join("ratatoskr")
                        .join("models")
                }),
            default_device: Device::default(),
            ram_budget: None,
        }
    }
}

/// Information about currently loaded models.
#[derive(Debug, Clone, Default)]
pub struct LoadedModels {
    /// Loaded embedding models.
    pub embeddings: Vec<String>,

    /// Loaded NLI models.
    pub nli: Vec<String>,
}

/// Model manager for lazy loading and caching local inference models.
///
/// Thread-safe with double-checked locking to ensure models are loaded at most once.
/// Models are stored in Arc so they can be shared across threads.
///
/// Tracks RAM usage and enforces budget constraints. When RAM budget would be
/// exceeded, returns `ModelNotAvailable` so the registry can try API fallback.
pub struct ModelManager {
    embedding_models: RwLock<HashMap<String, Arc<RwLock<FastEmbedProvider>>>>,
    nli_models: RwLock<HashMap<String, Arc<RwLock<OnnxNliProvider>>>>,
    /// Tracks estimated RAM usage per loaded model (key = model name).
    loaded_sizes: RwLock<HashMap<String, usize>>,
    config: ModelManagerConfig,
}

impl ModelManager {
    /// Create a new model manager with the given configuration.
    pub fn new(config: ModelManagerConfig) -> Self {
        Self {
            embedding_models: RwLock::new(HashMap::new()),
            nli_models: RwLock::new(HashMap::new()),
            loaded_sizes: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create a model manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ModelManagerConfig::default())
    }

    /// Get or lazily load an embedding model.
    ///
    /// Uses double-checked locking to ensure thread-safe lazy loading.
    /// Returns `ModelNotAvailable` if loading would exceed RAM budget.
    pub fn embedding(&self, model: LocalEmbeddingModel) -> Result<Arc<RwLock<FastEmbedProvider>>> {
        let key = model.cache_key();
        let model_name = model.name();

        // Fast path: check if already loaded (read lock)
        {
            let models = self.embedding_models.read().map_err(|e| {
                RatatoskrError::Configuration(format!("Failed to acquire read lock: {}", e))
            })?;

            if let Some(provider) = models.get(&key) {
                return Ok(Arc::clone(provider));
            }
        }

        // Check RAM budget before attempting to load
        if !self.can_load(model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }

        // Slow path: need to load (write lock)
        let mut models = self.embedding_models.write().map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to acquire write lock: {}", e))
        })?;

        // Double-check after acquiring write lock
        if let Some(provider) = models.get(&key) {
            return Ok(Arc::clone(provider));
        }

        // Actually load the model
        let provider = FastEmbedProvider::new(model)?;
        let arc_provider = Arc::new(RwLock::new(provider));
        models.insert(key.clone(), Arc::clone(&arc_provider));

        // Track RAM usage
        let size = self.estimate_model_size(model_name);
        if let Ok(mut sizes) = self.loaded_sizes.write() {
            sizes.insert(key, size);
        }

        Ok(arc_provider)
    }

    /// Get or lazily load an NLI model.
    ///
    /// Uses double-checked locking to ensure thread-safe lazy loading.
    /// Returns `ModelNotAvailable` if loading would exceed RAM budget.
    pub fn nli(&self, model: LocalNliModel) -> Result<Arc<RwLock<OnnxNliProvider>>> {
        let key = model.cache_key();
        let model_name = model.name().to_string();

        // Fast path: check if already loaded (read lock)
        {
            let models = self.nli_models.read().map_err(|e| {
                RatatoskrError::Configuration(format!("Failed to acquire read lock: {}", e))
            })?;

            if let Some(provider) = models.get(&key) {
                return Ok(Arc::clone(provider));
            }
        }

        // Check RAM budget before attempting to load
        if !self.can_load(&model_name) {
            return Err(RatatoskrError::ModelNotAvailable);
        }

        // Slow path: need to load (write lock)
        let mut models = self.nli_models.write().map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to acquire write lock: {}", e))
        })?;

        // Double-check after acquiring write lock
        if let Some(provider) = models.get(&key) {
            return Ok(Arc::clone(provider));
        }

        // Actually load the model
        let provider = OnnxNliProvider::new(model, self.config.default_device)?;
        let arc_provider = Arc::new(RwLock::new(provider));
        models.insert(key.clone(), Arc::clone(&arc_provider));

        // Track RAM usage
        let size = self.estimate_model_size(&model_name);
        if let Ok(mut sizes) = self.loaded_sizes.write() {
            sizes.insert(key, size);
        }

        Ok(arc_provider)
    }

    /// Explicitly preload an embedding model.
    ///
    /// Useful for latency-sensitive deployments where you want models
    /// loaded before the first request.
    pub fn preload_embedding(&self, model: LocalEmbeddingModel) -> Result<()> {
        let _ = self.embedding(model)?;
        Ok(())
    }

    /// Explicitly preload an NLI model.
    ///
    /// Useful for latency-sensitive deployments where you want models
    /// loaded before the first request.
    pub fn preload_nli(&self, model: LocalNliModel) -> Result<()> {
        let _ = self.nli(model)?;
        Ok(())
    }

    /// Unload an embedding model from cache.
    ///
    /// Returns true if the model was found and removed.
    pub fn unload_embedding(&self, model: &LocalEmbeddingModel) -> bool {
        let key = model.cache_key();
        let removed = self
            .embedding_models
            .write()
            .map(|mut models| models.remove(&key).is_some())
            .unwrap_or(false);

        if removed && let Ok(mut sizes) = self.loaded_sizes.write() {
            sizes.remove(&key);
        }
        removed
    }

    /// Unload an NLI model from cache.
    ///
    /// Returns true if the model was found and removed.
    pub fn unload_nli(&self, model: &LocalNliModel) -> bool {
        let key = model.cache_key();
        let removed = self
            .nli_models
            .write()
            .map(|mut models| models.remove(&key).is_some())
            .unwrap_or(false);

        if removed && let Ok(mut sizes) = self.loaded_sizes.write() {
            sizes.remove(&key);
        }
        removed
    }

    /// Get information about currently loaded models.
    pub fn loaded_models(&self) -> LoadedModels {
        let embeddings = self
            .embedding_models
            .read()
            .map(|models| models.keys().cloned().collect())
            .unwrap_or_default();

        let nli = self
            .nli_models
            .read()
            .map(|models| models.keys().cloned().collect())
            .unwrap_or_default();

        LoadedModels { embeddings, nli }
    }

    /// Check if we have RAM budget to load a model.
    ///
    /// Returns true if:
    /// - No RAM budget is configured (unlimited), or
    /// - The model is already loaded, or
    /// - Loading the model would not exceed the budget.
    pub fn can_load(&self, model_name: &str) -> bool {
        let Some(budget) = self.config.ram_budget else {
            return true; // No budget = unlimited
        };

        // Check if already loaded (by model name, not cache key)
        let sizes = match self.loaded_sizes.read() {
            Ok(s) => s,
            Err(_) => return false,
        };

        // If any loaded model contains this name, it's already loaded
        for key in sizes.keys() {
            if key.contains(model_name) {
                return true;
            }
        }

        // Estimate size and check budget
        let estimated_size = self.estimate_model_size(model_name);
        let current_usage: usize = sizes.values().sum();

        current_usage + estimated_size <= budget
    }

    /// Estimate model size in bytes based on model name.
    ///
    /// These are rough estimates based on typical model sizes.
    /// Could be made more accurate with model metadata.
    fn estimate_model_size(&self, model_name: &str) -> usize {
        match model_name {
            // Embedding models
            n if n.contains("MiniLM-L6") => 90 * 1024 * 1024,    // ~90MB
            n if n.contains("MiniLM-L12") => 120 * 1024 * 1024,  // ~120MB
            n if n.contains("bge-small") || n.contains("BGE-small") => 130 * 1024 * 1024, // ~130MB
            n if n.contains("bge-base") || n.contains("BGE-base") => 440 * 1024 * 1024,   // ~440MB
            // NLI models
            n if n.contains("deberta") && n.contains("small") => 300 * 1024 * 1024, // ~300MB
            n if n.contains("deberta") && n.contains("base") => 400 * 1024 * 1024,  // ~400MB
            // Default estimate
            _ => 200 * 1024 * 1024, // ~200MB default
        }
    }

    /// Get current total RAM usage of loaded models.
    pub fn current_usage(&self) -> usize {
        self.loaded_sizes
            .read()
            .map(|sizes| sizes.values().sum())
            .unwrap_or(0)
    }

    /// Get the configured RAM budget.
    pub fn ram_budget(&self) -> Option<usize> {
        self.config.ram_budget
    }

    /// Get the configuration.
    pub fn config(&self) -> &ModelManagerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_manager_config_default() {
        let config = ModelManagerConfig::default();
        assert!(config.cache_dir.to_string_lossy().contains("ratatoskr"));
        assert_eq!(config.default_device, Device::Cpu);
        assert!(config.ram_budget.is_none());
    }

    #[test]
    fn test_loaded_models_default() {
        let loaded = LoadedModels::default();
        assert!(loaded.embeddings.is_empty());
        assert!(loaded.nli.is_empty());
    }

    #[test]
    fn test_model_manager_new() {
        let manager = ModelManager::with_defaults();
        let loaded = manager.loaded_models();
        assert!(loaded.embeddings.is_empty());
        assert!(loaded.nli.is_empty());
        assert_eq!(manager.current_usage(), 0);
        assert!(manager.ram_budget().is_none());
    }

    #[test]
    fn test_can_load_unlimited_budget() {
        let manager = ModelManager::with_defaults();
        // No budget = unlimited, should always return true
        assert!(manager.can_load("any-model"));
        assert!(manager.can_load("all-MiniLM-L6-v2"));
        assert!(manager.can_load("nli-deberta-v3-base"));
    }

    #[test]
    fn test_can_load_with_budget() {
        let config = ModelManagerConfig {
            ram_budget: Some(100 * 1024 * 1024), // 100MB budget
            ..Default::default()
        };
        let manager = ModelManager::new(config);

        // MiniLM-L6 is ~90MB, should fit
        assert!(manager.can_load("all-MiniLM-L6-v2"));

        // BGE-base is ~440MB, should not fit
        assert!(!manager.can_load("BGE-base-en"));

        // Deberta-base is ~400MB, should not fit
        assert!(!manager.can_load("nli-deberta-v3-base"));
    }

    #[test]
    fn test_estimate_model_size() {
        let manager = ModelManager::with_defaults();

        // Embedding models
        assert_eq!(
            manager.estimate_model_size("all-MiniLM-L6-v2"),
            90 * 1024 * 1024
        );
        assert_eq!(
            manager.estimate_model_size("all-MiniLM-L12-v2"),
            120 * 1024 * 1024
        );
        assert_eq!(
            manager.estimate_model_size("BGE-small-en"),
            130 * 1024 * 1024
        );
        assert_eq!(
            manager.estimate_model_size("BGE-base-en"),
            440 * 1024 * 1024
        );

        // NLI models
        assert_eq!(
            manager.estimate_model_size("nli-deberta-v3-small"),
            300 * 1024 * 1024
        );
        assert_eq!(
            manager.estimate_model_size("nli-deberta-v3-base"),
            400 * 1024 * 1024
        );

        // Unknown model gets default
        assert_eq!(
            manager.estimate_model_size("some-unknown-model"),
            200 * 1024 * 1024
        );
    }

    #[test]
    fn test_config_with_ram_budget() {
        let config = ModelManagerConfig {
            ram_budget: Some(1024 * 1024 * 1024), // 1GB
            ..Default::default()
        };
        let manager = ModelManager::new(config);

        assert_eq!(manager.ram_budget(), Some(1024 * 1024 * 1024));
    }
}
