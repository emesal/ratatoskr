//! Token counting system for accurate LLM input measurement.
//!
//! This module provides a flexible tokenizer registry that maps model names to
//! appropriate tokenizers. Tokenizers are loaded lazily from HuggingFace Hub
//! and cached for reuse.

#[cfg(feature = "local-inference")]
mod hf;

#[cfg(feature = "local-inference")]
pub use hf::HfTokenizer;

use crate::error::Result;
#[cfg(feature = "local-inference")]
use std::collections::HashMap;
#[cfg(feature = "local-inference")]
use std::path::PathBuf;
#[cfg(feature = "local-inference")]
use std::sync::{Arc, RwLock};

/// Trait for tokenizer implementations, allowing future expansion.
pub trait TokenizerProvider: Send + Sync {
    /// Count tokens in the given text.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Tokenize text into token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
}

/// Source for a tokenizer model.
#[cfg(feature = "local-inference")]
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// Load from HuggingFace Hub repository.
    HuggingFace { repo_id: String },
    /// Load from local file path.
    Local { path: PathBuf },
    /// Alias to another model's tokenizer.
    Alias { target: String },
}

/// Registry mapping model names to tokenizers.
///
/// Provides lazy loading and caching of tokenizers with default mappings
/// for common model families.
#[cfg(feature = "local-inference")]
pub struct TokenizerRegistry {
    /// Loaded tokenizers cache.
    tokenizers: RwLock<HashMap<String, Arc<dyn TokenizerProvider>>>,
    /// Model pattern → tokenizer source mappings.
    model_mappings: HashMap<String, TokenizerSource>,
}

#[cfg(feature = "local-inference")]
impl TokenizerRegistry {
    /// Create a new registry with default model mappings.
    pub fn new() -> Self {
        let mut model_mappings = HashMap::new();

        // Default mappings for common model families
        model_mappings.insert(
            "claude".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "Xenova/claude-tokenizer".to_string(),
            },
        );
        model_mappings.insert(
            "gpt-4".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "Xenova/gpt-4".to_string(),
            },
        );
        model_mappings.insert(
            "gpt-3.5".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "Xenova/gpt-4".to_string(),
            },
        );
        model_mappings.insert(
            "llama".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "meta-llama/Llama-3.2-1B".to_string(),
            },
        );
        model_mappings.insert(
            "meta-llama".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "meta-llama/Llama-3.2-1B".to_string(),
            },
        );
        model_mappings.insert(
            "mistral".to_string(),
            TokenizerSource::HuggingFace {
                repo_id: "mistralai/Mistral-7B-v0.1".to_string(),
            },
        );

        Self {
            tokenizers: RwLock::new(HashMap::new()),
            model_mappings,
        }
    }

    /// Register a custom model → tokenizer mapping.
    ///
    /// This allows users to override default mappings or add new ones.
    pub fn register(&mut self, model_pattern: impl Into<String>, source: TokenizerSource) {
        self.model_mappings.insert(model_pattern.into(), source);
    }

    /// Count tokens for the given text using the specified model's tokenizer.
    ///
    /// Lazily loads the tokenizer if not already cached.
    pub fn count_tokens(&self, text: &str, model: &str) -> Result<usize> {
        let provider = self.get_or_load_tokenizer(model)?;
        provider.count_tokens(text)
    }

    /// Tokenize text using the specified model's tokenizer.
    ///
    /// Lazily loads the tokenizer if not already cached.
    pub fn tokenize(&self, text: &str, model: &str) -> Result<Vec<u32>> {
        let provider = self.get_or_load_tokenizer(model)?;
        provider.tokenize(text)
    }

    /// Get or lazily load a tokenizer for the given model.
    fn get_or_load_tokenizer(&self, model: &str) -> Result<Arc<dyn TokenizerProvider>> {
        // Fast path: already loaded
        if let Some(provider) = self.tokenizers.read().unwrap().get(model) {
            return Ok(Arc::clone(provider));
        }

        // Slow path: need to load
        let source = self.resolve_source(model)?;
        let provider = self.load_tokenizer(&source)?;

        // Cache it
        let mut cache = self.tokenizers.write().unwrap();
        // Double-check after acquiring write lock
        if let Some(existing) = cache.get(model) {
            return Ok(Arc::clone(existing));
        }

        let provider: Arc<dyn TokenizerProvider> = Arc::new(provider);
        cache.insert(model.to_string(), Arc::clone(&provider));
        Ok(provider)
    }

    /// Resolve model name to tokenizer source.
    ///
    /// Matches against registered patterns (longest match wins).
    #[doc(hidden)]
    pub fn resolve_source(&self, model: &str) -> Result<TokenizerSource> {
        // Try exact match first
        if let Some(source) = self.model_mappings.get(model) {
            // Resolve aliases recursively
            return match source {
                TokenizerSource::Alias { target } => self.resolve_source(target),
                _ => Ok(source.clone()),
            };
        }

        // Try prefix matching (longest match wins)
        let mut best_match: Option<(&str, &TokenizerSource)> = None;

        for (pattern, source) in &self.model_mappings {
            if model.starts_with(pattern) {
                if let Some((best_pattern, _)) = best_match {
                    if pattern.len() > best_pattern.len() {
                        best_match = Some((pattern, source));
                    }
                } else {
                    best_match = Some((pattern, source));
                }
            }
        }

        if let Some((_, source)) = best_match {
            // Resolve aliases
            match source {
                TokenizerSource::Alias { target } => self.resolve_source(target),
                _ => Ok(source.clone()),
            }
        } else {
            Err(crate::error::RatatoskrError::Configuration(format!(
                "No tokenizer configured for model: {}",
                model
            )))
        }
    }

    /// Load a tokenizer from the given source.
    fn load_tokenizer(&self, source: &TokenizerSource) -> Result<HfTokenizer> {
        match source {
            TokenizerSource::HuggingFace { repo_id } => HfTokenizer::from_hub(repo_id),
            TokenizerSource::Local { path } => HfTokenizer::from_file(path),
            TokenizerSource::Alias { .. } => {
                unreachable!("Aliases should be resolved in resolve_source")
            }
        }
    }
}

#[cfg(feature = "local-inference")]
impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
