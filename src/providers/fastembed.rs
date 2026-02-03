//! Local embeddings via fastembed-rs.

use crate::error::{RatatoskrError, Result};
use crate::types::Embedding;

/// Supported local embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalEmbeddingModel {
    /// all-MiniLM-L6-v2 (384 dims, fast, good quality).
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2 (384 dims, slightly better).
    AllMiniLmL12V2,
    /// BGE-small-en (384 dims, strong retrieval).
    BgeSmallEn,
    /// BGE-base-en (768 dims, higher quality).
    BgeBaseEn,
}

impl LocalEmbeddingModel {
    /// Get the model name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "all-MiniLM-L12-v2",
            Self::BgeSmallEn => "BGE-small-en",
            Self::BgeBaseEn => "BGE-base-en",
        }
    }

    /// Get the embedding dimensions.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::AllMiniLmL12V2 | Self::BgeSmallEn => 384,
            Self::BgeBaseEn => 768,
        }
    }

    /// Get cache key for model manager.
    pub(crate) fn cache_key(&self) -> String {
        format!("fastembed:{}", self.name())
    }
}

impl From<LocalEmbeddingModel> for fastembed::EmbeddingModel {
    fn from(model: LocalEmbeddingModel) -> Self {
        match model {
            LocalEmbeddingModel::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            LocalEmbeddingModel::AllMiniLmL12V2 => fastembed::EmbeddingModel::AllMiniLML12V2,
            LocalEmbeddingModel::BgeSmallEn => fastembed::EmbeddingModel::BGESmallENV15,
            LocalEmbeddingModel::BgeBaseEn => fastembed::EmbeddingModel::BGEBaseENV15,
        }
    }
}

/// Information about an embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    /// Model name.
    pub name: String,
    /// Embedding dimensions.
    pub dimensions: usize,
}

impl From<LocalEmbeddingModel> for EmbeddingModelInfo {
    fn from(model: LocalEmbeddingModel) -> Self {
        Self {
            name: model.name().to_string(),
            dimensions: model.dimensions(),
        }
    }
}

/// Local embedding provider using fastembed-rs.
pub struct FastEmbedProvider {
    model: fastembed::TextEmbedding,
    model_info: EmbeddingModelInfo,
}

impl FastEmbedProvider {
    /// Create a new provider with the specified model.
    ///
    /// Downloads the model if not cached locally.
    pub fn new(model: LocalEmbeddingModel) -> Result<Self> {
        let cache_dir = std::env::var("RATATOSKR_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::cache_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
                    .join("ratatoskr")
                    .join("models")
            });

        let options = fastembed::InitOptions::new(model.into())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir);

        let model_instance = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to load embedding model: {}", e))
        })?;

        Ok(Self {
            model: model_instance,
            model_info: model.into(),
        })
    }

    /// Generate embedding for a single text.
    pub fn embed(&mut self, text: &str) -> Result<Embedding> {
        let vectors = self
            .model
            .embed(vec![text.to_string()], None)
            .map_err(|e| RatatoskrError::DataError(format!("Embedding failed: {}", e)))?;

        let values = vectors
            .into_iter()
            .next()
            .ok_or_else(|| RatatoskrError::DataError("No embedding returned".to_string()))?;

        Ok(Embedding {
            values,
            model: self.model_info.name.clone(),
            dimensions: self.model_info.dimensions,
        })
    }

    /// Generate embeddings for multiple texts.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        let vectors = self
            .model
            .embed(texts_owned, None)
            .map_err(|e| RatatoskrError::DataError(format!("Batch embedding failed: {}", e)))?;

        Ok(vectors
            .into_iter()
            .map(|values| Embedding {
                values,
                model: self.model_info.name.clone(),
                dimensions: self.model_info.dimensions,
            })
            .collect())
    }

    /// Get model information.
    pub fn model_info(&self) -> &EmbeddingModelInfo {
        &self.model_info
    }
}
