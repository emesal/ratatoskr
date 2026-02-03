//! HuggingFace tokenizers implementation.

use super::TokenizerProvider;
use crate::error::{RatatoskrError, Result};
use std::path::Path;

/// HuggingFace tokenizers implementation.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    /// Load tokenizer from HuggingFace Hub.
    ///
    /// Downloads the tokenizer if not cached locally.
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to initialize HF API: {}", e))
        })?;

        let repo = api.model(repo_id.to_string());

        // Try to get tokenizer.json
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            RatatoskrError::Configuration(format!(
                "Failed to download tokenizer from {}: {}",
                repo_id, e
            ))
        })?;

        Self::from_file(&tokenizer_path)
    }

    /// Load tokenizer from local file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            RatatoskrError::Configuration(format!(
                "Failed to load tokenizer from {:?}: {}",
                path, e
            ))
        })?;

        Ok(Self { inner })
    }
}

impl TokenizerProvider for HfTokenizer {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RatatoskrError::DataError(format!("Tokenization failed: {}", e)))?;

        Ok(encoding.len())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RatatoskrError::DataError(format!("Tokenization failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }
}
