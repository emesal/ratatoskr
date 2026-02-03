//! Live tests for local inference capabilities.
//!
//! These tests require actual model downloads and execution.
//! Run with:
//! ```bash
//! cargo test --test local_inference_live_test --features local-inference -- --ignored
//! ```
//!
//! First run will download models (~100MB+ for embeddings, ~500MB+ for NLI).

#![cfg(feature = "local-inference")]

use ratatoskr::providers::{FastEmbedProvider, LocalEmbeddingModel};

// ============================================================================
// Local Embedding Tests
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_live_local_embedding() {
    let mut provider = FastEmbedProvider::new(LocalEmbeddingModel::AllMiniLmL6V2)
        .expect("Failed to create embedding provider");

    let embedding = provider.embed("Hello, world!").expect("Embedding failed");

    assert_eq!(embedding.dimensions, 384);
    assert_eq!(embedding.values.len(), 384);
    // Values should be normalized (L2 norm ≈ 1)
    let norm: f32 = embedding.values.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Expected L2 norm ≈ 1, got {}",
        norm
    );
}

#[tokio::test]
#[ignore]
async fn test_live_local_embedding_batch() {
    let mut provider = FastEmbedProvider::new(LocalEmbeddingModel::AllMiniLmL6V2)
        .expect("Failed to create embedding provider");

    let texts = vec!["First sentence", "Second sentence", "Third sentence"];
    let embeddings = provider
        .embed_batch(&texts)
        .expect("Batch embedding failed");

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.dimensions, 384);
    }

    // Similar sentences should have higher cosine similarity
    let emb1 = &embeddings[0].values;
    let emb2 = &embeddings[1].values;

    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        dot_product > 0.5,
        "Similar sentences should have cosine sim > 0.5, got {}",
        dot_product
    );
}

#[tokio::test]
#[ignore]
async fn test_live_local_embedding_different_models() {
    // Test AllMiniLmL6V2 (384 dims)
    let mut provider_small = FastEmbedProvider::new(LocalEmbeddingModel::AllMiniLmL6V2)
        .expect("Failed to create small embedding provider");
    let emb_small = provider_small
        .embed("Test")
        .expect("Small embedding failed");
    assert_eq!(emb_small.dimensions, 384);

    // Test BgeSmallEn (384 dims)
    let mut provider_bge = FastEmbedProvider::new(LocalEmbeddingModel::BgeSmallEn)
        .expect("Failed to create BGE embedding provider");
    let emb_bge = provider_bge.embed("Test").expect("BGE embedding failed");
    assert_eq!(emb_bge.dimensions, 384);
}

// ============================================================================
// Local NLI Tests
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_live_local_nli() {
    use ratatoskr::providers::{LocalNliModel, OnnxNliProvider};
    use ratatoskr::{Device, NliLabel};

    let mut provider = OnnxNliProvider::new(LocalNliModel::NliDebertaV3Small, Device::Cpu)
        .expect("Failed to create NLI provider");

    // Clear entailment
    let result = provider
        .infer_nli("A cat is sleeping", "An animal is resting")
        .expect("NLI inference failed");

    assert!(
        matches!(result.label, NliLabel::Entailment),
        "Expected entailment, got {:?}",
        result.label
    );
    assert!(result.entailment > 0.5, "Entailment score should be > 0.5");

    // Clear contradiction
    let result = provider
        .infer_nli("The sky is blue", "The sky is green")
        .expect("NLI inference failed");

    assert!(
        matches!(result.label, NliLabel::Contradiction),
        "Expected contradiction, got {:?}",
        result.label
    );
    assert!(
        result.contradiction > 0.5,
        "Contradiction score should be > 0.5"
    );
}

#[tokio::test]
#[ignore]
async fn test_live_local_nli_batch() {
    use ratatoskr::Device;
    use ratatoskr::providers::{LocalNliModel, OnnxNliProvider};

    let mut provider = OnnxNliProvider::new(LocalNliModel::NliDebertaV3Small, Device::Cpu)
        .expect("Failed to create NLI provider");

    let pairs = [
        ("Dogs are animals", "Dogs are living things"), // entailment
        ("It is sunny", "It is raining"),               // contradiction
        ("I went to the store", "I bought something"),  // neutral
    ];

    let results = provider
        .infer_nli_batch(&pairs)
        .expect("Batch NLI inference failed");

    assert_eq!(results.len(), 3);
    // Just verify we got results; exact labels depend on model confidence
    for result in &results {
        assert!(result.entailment >= 0.0 && result.entailment <= 1.0);
        assert!(result.neutral >= 0.0 && result.neutral <= 1.0);
        assert!(result.contradiction >= 0.0 && result.contradiction <= 1.0);
    }
}

// ============================================================================
// Token Counting Tests
// ============================================================================

#[test]
#[ignore]
fn test_live_token_counting() {
    use ratatoskr::tokenizer::TokenizerRegistry;

    let registry = TokenizerRegistry::new();

    // Test Claude tokenizer (via alias)
    let count = registry
        .count_tokens("Hello, world!", "claude-sonnet-4")
        .expect("Token counting failed");
    assert!(
        count > 0 && count < 10,
        "Expected 2-5 tokens, got {}",
        count
    );

    // Test with a longer text
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let count_long = registry
        .count_tokens(&long_text, "claude-3-opus")
        .expect("Token counting failed");
    assert!(count_long > 50, "Expected > 50 tokens, got {}", count_long);
}

#[test]
#[ignore]
fn test_live_token_counting_different_models() {
    use ratatoskr::tokenizer::TokenizerRegistry;

    let registry = TokenizerRegistry::new();
    let text = "Hello world, this is a test sentence.";

    // Different model families may have different tokenizations
    let claude_count = registry
        .count_tokens(text, "claude-sonnet-4")
        .expect("Claude token count failed");

    let gpt_count = registry
        .count_tokens(text, "gpt-4")
        .expect("GPT-4 token count failed");

    // Both should be reasonable (not zero, not huge)
    assert!(claude_count > 0 && claude_count < 50);
    assert!(gpt_count > 0 && gpt_count < 50);
}

// ============================================================================
// Model Manager Tests
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_live_model_manager_lazy_loading() {
    use ratatoskr::model::ModelManager;

    let manager = ModelManager::with_defaults();

    // Initially no models loaded
    let loaded = manager.loaded_models();
    assert!(loaded.embeddings.is_empty());

    // Load embedding model
    let provider = manager
        .embedding(LocalEmbeddingModel::AllMiniLmL6V2)
        .expect("Failed to get embedding provider");

    // Model should now be loaded
    let loaded = manager.loaded_models();
    assert_eq!(loaded.embeddings.len(), 1);

    // Second call should return cached provider
    let provider2 = manager
        .embedding(LocalEmbeddingModel::AllMiniLmL6V2)
        .expect("Failed to get cached provider");

    // Should be the same Arc
    assert!(std::sync::Arc::ptr_eq(&provider, &provider2));

    // Test embedding works (need to lock the RwLock)
    let emb = provider
        .write()
        .unwrap()
        .embed("Test")
        .expect("Embedding failed");
    assert_eq!(emb.dimensions, 384);
}

// ============================================================================
// Gateway Integration Tests
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_live_gateway_count_tokens() {
    use ratatoskr::{ModelGateway, Ratatoskr};

    let gateway = Ratatoskr::builder()
        .openrouter("dummy-key-for-chat") // Need at least one provider
        .build()
        .expect("Failed to build gateway");

    let count = gateway
        .count_tokens("Hello, world!", "claude-sonnet-4")
        .expect("Token counting failed");

    assert!(count > 0);
}

// ============================================================================
// RAM Budget and Fallback Tests
// ============================================================================

/// Test that LocalEmbeddingProvider returns ModelNotAvailable for wrong model.
#[tokio::test]
#[ignore]
async fn test_live_local_provider_wrong_model() {
    use ratatoskr::RatatoskrError;
    use ratatoskr::model::ModelManager;
    use ratatoskr::providers::LocalEmbeddingProvider;
    use ratatoskr::providers::traits::EmbeddingProvider;
    use std::sync::Arc;

    let manager = Arc::new(ModelManager::with_defaults());
    let provider = LocalEmbeddingProvider::new(LocalEmbeddingModel::AllMiniLmL6V2, manager);

    // Request a different model than what provider handles
    let result = provider.embed("test", "some-other-model").await;

    assert!(
        matches!(result, Err(RatatoskrError::ModelNotAvailable)),
        "expected ModelNotAvailable for wrong model, got {:?}",
        result
    );
}

/// Test that LocalEmbeddingProvider returns ModelNotAvailable when RAM budget exceeded.
#[tokio::test]
#[ignore]
async fn test_live_local_provider_ram_budget_exceeded() {
    use ratatoskr::model::{ModelManager, ModelManagerConfig};
    use ratatoskr::providers::LocalEmbeddingProvider;
    use ratatoskr::providers::traits::EmbeddingProvider;
    use ratatoskr::{Device, RatatoskrError};
    use std::sync::Arc;

    // Create manager with impossible RAM budget (1 byte)
    let config = ModelManagerConfig {
        ram_budget: Some(1),
        default_device: Device::Cpu,
        ..Default::default()
    };
    let manager = Arc::new(ModelManager::new(config));
    let provider = LocalEmbeddingProvider::new(LocalEmbeddingModel::AllMiniLmL6V2, manager);

    // Request should fail due to RAM budget
    let result = provider.embed("test", "all-MiniLM-L6-v2").await;

    assert!(
        matches!(result, Err(RatatoskrError::ModelNotAvailable)),
        "expected ModelNotAvailable due to RAM budget, got {:?}",
        result
    );
}

/// Test LocalEmbeddingProvider trait implementation with correct model.
/// This loads the actual model - slow on first run.
#[tokio::test]
#[ignore]
async fn test_live_local_embedding_provider_trait() {
    use ratatoskr::model::ModelManager;
    use ratatoskr::providers::LocalEmbeddingProvider;
    use ratatoskr::providers::traits::EmbeddingProvider;
    use std::sync::Arc;

    let manager = Arc::new(ModelManager::with_defaults());
    let provider = LocalEmbeddingProvider::new(LocalEmbeddingModel::AllMiniLmL6V2, manager);

    // Request the correct model
    let result = provider.embed("Hello world!", "all-MiniLM-L6-v2").await;

    let embedding = result.expect("embedding should succeed with correct model");
    assert_eq!(embedding.dimensions, 384);
    assert_eq!(embedding.model, "all-MiniLM-L6-v2");

    // Verify embedding is normalized
    let norm: f32 = embedding.values.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Expected L2 norm ≈ 1, got {}",
        norm
    );
}

/// Test LocalNliProvider trait implementation.
#[tokio::test]
#[ignore]
async fn test_live_local_nli_provider_trait() {
    use ratatoskr::NliLabel;
    use ratatoskr::model::ModelManager;
    use ratatoskr::providers::traits::NliProvider;
    use ratatoskr::providers::{LocalNliModel, LocalNliProvider};
    use std::sync::Arc;

    let manager = Arc::new(ModelManager::with_defaults());
    let provider = LocalNliProvider::new(LocalNliModel::NliDebertaV3Small, manager);

    // Test with the correct model name
    let result = provider
        .infer_nli(
            "The cat is sleeping on the mat.",
            "An animal is resting.",
            "nli-deberta-v3-small",
        )
        .await;

    let nli = result.expect("NLI should succeed with correct model");
    assert!(
        matches!(nli.label, NliLabel::Entailment),
        "Expected entailment, got {:?}",
        nli.label
    );
}
