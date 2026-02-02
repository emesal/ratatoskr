//! Model manager for lazy loading and caching.
//!
//! This module will be fully implemented in Task 8.

use std::path::PathBuf;

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

/// Model manager (stub - will be implemented in Task 8).
pub struct ModelManager {
    #[allow(dead_code)]
    config: ModelManagerConfig,
}

impl ModelManager {
    /// Create a new model manager.
    pub fn new(config: ModelManagerConfig) -> Self {
        Self { config }
    }
}
