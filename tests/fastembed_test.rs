//! Tests for fastembed provider.

#![cfg(feature = "local-inference")]

use ratatoskr::providers::LocalEmbeddingModel;

#[test]
fn test_local_embedding_model_properties() {
    let model = LocalEmbeddingModel::AllMiniLmL6V2;
    assert_eq!(model.name(), "all-MiniLM-L6-v2");
    assert_eq!(model.dimensions(), 384);

    let model = LocalEmbeddingModel::BgeBaseEn;
    assert_eq!(model.name(), "BGE-base-en");
    assert_eq!(model.dimensions(), 768);
}

// Note: Tests that actually load models are in live tests, as they require
// network access to download models and are slow.
