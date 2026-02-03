//! Model source and download logic.

use crate::error::{RatatoskrError, Result};
use std::path::PathBuf;

/// Source for a model.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Load from HuggingFace Hub repository.
    HuggingFace {
        /// Repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        repo_id: String,
        /// Optional specific file within the repo.
        file: Option<String>,
    },

    /// Load from local file or directory.
    Local {
        /// Path to model file or directory.
        path: PathBuf,
    },
}

impl ModelSource {
    /// Create a HuggingFace source.
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            file: None,
        }
    }

    /// Create a HuggingFace source with a specific file.
    pub fn huggingface_file(repo_id: impl Into<String>, file: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            file: Some(file.into()),
        }
    }

    /// Create a local source.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local { path: path.into() }
    }

    /// Download or resolve the model to a local path.
    ///
    /// For HuggingFace sources, downloads the model if not cached.
    /// For local sources, returns the path as-is.
    pub fn resolve(&self) -> Result<PathBuf> {
        match self {
            Self::HuggingFace { repo_id, file } => {
                let api = hf_hub::api::sync::Api::new().map_err(|e| {
                    RatatoskrError::Configuration(format!("Failed to initialize HF API: {}", e))
                })?;

                let repo = api.model(repo_id.clone());

                let path = if let Some(file) = file {
                    repo.get(file).map_err(|e| {
                        RatatoskrError::Configuration(format!(
                            "Failed to download {} from {}: {}",
                            file, repo_id, e
                        ))
                    })?
                } else {
                    // Download the entire repo
                    repo.get("model.onnx")
                        .or_else(|_| repo.get("model.safetensors"))
                        .or_else(|_| repo.get("pytorch_model.bin"))
                        .map_err(|e| {
                            RatatoskrError::Configuration(format!(
                                "Failed to find model file in {}: {}",
                                repo_id, e
                            ))
                        })?
                };

                Ok(path)
            }
            Self::Local { path } => {
                if !path.exists() {
                    return Err(RatatoskrError::Configuration(format!(
                        "Local model path does not exist: {}",
                        path.display()
                    )));
                }
                Ok(path.clone())
            }
        }
    }
}
