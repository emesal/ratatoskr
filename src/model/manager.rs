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
pub struct ModelManager {
    embedding_models: RwLock<HashMap<String, Arc<RwLock<FastEmbedProvider>>>>,
    nli_models: RwLock<HashMap<String, Arc<RwLock<OnnxNliProvider>>>>,
    config: ModelManagerConfig,
}

impl ModelManager {
    /// Create a new model manager with the given configuration.
    pub fn new(config: ModelManagerConfig) -> Self {
        Self {
            embedding_models: RwLock::new(HashMap::new()),
            nli_models: RwLock::new(HashMap::new()),
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
    pub fn embedding(
        &self,
        model: LocalEmbeddingModel,
    ) -> Result<Arc<RwLock<FastEmbedProvider>>> {
        let key = model.cache_key();

        // Fast path: check if already loaded (read lock)
        {
            let models = self.embedding_models.read().map_err(|e| {
                RatatoskrError::Configuration(format!("Failed to acquire read lock: {}", e))
            })?;

            if let Some(provider) = models.get(&key) {
                return Ok(Arc::clone(provider));
            }
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
        models.insert(key, Arc::clone(&arc_provider));

        Ok(arc_provider)
    }

    /// Get or lazily load an NLI model.
    ///
    /// Uses double-checked locking to ensure thread-safe lazy loading.
    pub fn nli(&self, model: LocalNliModel) -> Result<Arc<RwLock<OnnxNliProvider>>> {
        let key = model.cache_key();

        // Fast path: check if already loaded (read lock)
        {
            let models = self.nli_models.read().map_err(|e| {
                RatatoskrError::Configuration(format!("Failed to acquire read lock: {}", e))
            })?;

            if let Some(provider) = models.get(&key) {
                return Ok(Arc::clone(provider));
            }
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
        models.insert(key, Arc::clone(&arc_provider));

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
        self.embedding_models
            .write()
            .map(|mut models| models.remove(&key).is_some())
            .unwrap_or(false)
    }

    /// Unload an NLI model from cache.
    ///
    /// Returns true if the model was found and removed.
    pub fn unload_nli(&self, model: &LocalNliModel) -> bool {
        let key = model.cache_key();
        self.nli_models
            .write()
            .map(|mut models| models.remove(&key).is_some())
            .unwrap_or(false)
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
    }
}
