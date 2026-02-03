//! Tests for model manager.

#![cfg(feature = "local-inference")]

use ratatoskr::{LocalEmbeddingModel, LocalNliModel, LoadedModels, ModelManager, ModelManagerConfig};
use std::path::PathBuf;

#[test]
fn test_model_manager_config_default() {
    let config = ModelManagerConfig::default();
    assert!(config.cache_dir.to_string_lossy().contains("ratatoskr"));
}

#[test]
fn test_model_manager_config_custom_cache_dir() {
    let config = ModelManagerConfig {
        cache_dir: PathBuf::from("/custom/cache"),
        ..Default::default()
    };
    assert_eq!(config.cache_dir, PathBuf::from("/custom/cache"));
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

#[test]
fn test_model_manager_unload_nonexistent() {
    let manager = ModelManager::with_defaults();

    // Unloading a model that was never loaded should return false
    let removed = manager.unload_embedding(&LocalEmbeddingModel::AllMiniLmL6V2);
    assert!(!removed);

    let removed = manager.unload_nli(&LocalNliModel::NliDebertaV3Small);
    assert!(!removed);
}

// Live test - requires model download
#[test]
#[ignore]
fn test_model_manager_embedding_lazy_load() {
    let manager = ModelManager::with_defaults();

    // First call should load the model
    let provider = manager.embedding(LocalEmbeddingModel::AllMiniLmL6V2).unwrap();
    assert!(provider.read().unwrap().model_info().dimensions == 384);

    // Second call should return cached provider
    let provider2 = manager.embedding(LocalEmbeddingModel::AllMiniLmL6V2).unwrap();
    assert!(std::sync::Arc::ptr_eq(&provider, &provider2));

    // Should show up in loaded models
    let loaded = manager.loaded_models();
    assert_eq!(loaded.embeddings.len(), 1);
    assert!(loaded.embeddings[0].contains("MiniLM"));
}

// Live test - requires model download
#[test]
#[ignore]
fn test_model_manager_embedding_unload() {
    let manager = ModelManager::with_defaults();

    // Load a model
    let _ = manager.embedding(LocalEmbeddingModel::AllMiniLmL6V2).unwrap();
    assert_eq!(manager.loaded_models().embeddings.len(), 1);

    // Unload it
    let removed = manager.unload_embedding(&LocalEmbeddingModel::AllMiniLmL6V2);
    assert!(removed);
    assert!(manager.loaded_models().embeddings.is_empty());

    // Unloading again should return false
    let removed = manager.unload_embedding(&LocalEmbeddingModel::AllMiniLmL6V2);
    assert!(!removed);
}
