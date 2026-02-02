//! Live integration tests for HuggingFaceClient.
//!
//! These tests hit the real HuggingFace Inference API and are `#[ignore]` by default.
//!
//! Run with: `HF_API_KEY=hf_xxx cargo test --test huggingface_live_test --features huggingface -- --ignored`
#![cfg(feature = "huggingface")]

use ratatoskr::providers::HuggingFaceClient;

fn get_api_key() -> String {
    std::env::var("HF_API_KEY").expect("HF_API_KEY environment variable must be set for live tests")
}

/// Test live embedding with sentence-transformers model.
#[tokio::test]
#[ignore = "requires HF_API_KEY"]
async fn test_live_embed() {
    let client = HuggingFaceClient::new(get_api_key());
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    let result = client.embed("hello world", model).await;

    let embedding = result.expect("live embed should succeed");
    assert_eq!(embedding.model, model);
    assert!(embedding.dimensions > 0, "embedding should have dimensions");
    assert_eq!(
        embedding.values.len(),
        embedding.dimensions,
        "values length should match dimensions"
    );

    // MiniLM-L6-v2 produces 384-dimensional embeddings
    assert_eq!(embedding.dimensions, 384);
}

/// Test live batch embedding.
#[tokio::test]
#[ignore = "requires HF_API_KEY"]
async fn test_live_embed_batch() {
    let client = HuggingFaceClient::new(get_api_key());
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    let result = client
        .embed_batch(&["hello world", "goodbye moon"], model)
        .await;

    let embeddings = result.expect("live embed_batch should succeed");
    assert_eq!(embeddings.len(), 2, "should return 2 embeddings");

    for embedding in &embeddings {
        assert_eq!(embedding.model, model);
        assert_eq!(embedding.dimensions, 384);
    }

    // Verify different texts produce different embeddings
    assert_ne!(
        embeddings[0].values, embeddings[1].values,
        "different texts should produce different embeddings"
    );
}

/// Test live zero-shot classification.
#[tokio::test]
#[ignore = "requires HF_API_KEY"]
async fn test_live_classify() {
    let client = HuggingFaceClient::new(get_api_key());
    let model = "facebook/bart-large-mnli";

    let result = client
        .classify(
            "I absolutely love this product!",
            &["positive", "negative", "neutral"],
            model,
        )
        .await;

    let classification = result.expect("live classify should succeed");

    assert!(
        classification.confidence > 0.0 && classification.confidence <= 1.0,
        "confidence should be between 0 and 1"
    );
    assert!(
        !classification.top_label.is_empty(),
        "should have a top label"
    );
    assert_eq!(
        classification.scores.len(),
        3,
        "should have scores for all labels"
    );

    // Positive sentiment should be detected
    assert_eq!(
        classification.top_label, "positive",
        "should detect positive sentiment"
    );
}

/// Test live NLI inference.
#[tokio::test]
#[ignore = "requires HF_API_KEY"]
async fn test_live_nli() {
    let client = HuggingFaceClient::new(get_api_key());
    let model = "facebook/bart-large-mnli";

    // Premise clearly entails hypothesis
    let result = client
        .infer_nli(
            "A person is riding a horse.",
            "Someone is on an animal.",
            model,
        )
        .await;

    let nli = result.expect("live infer_nli should succeed");

    assert!(
        nli.entailment >= 0.0 && nli.entailment <= 1.0,
        "entailment should be between 0 and 1"
    );
    assert!(
        nli.neutral >= 0.0 && nli.neutral <= 1.0,
        "neutral should be between 0 and 1"
    );
    assert!(
        nli.contradiction >= 0.0 && nli.contradiction <= 1.0,
        "contradiction should be between 0 and 1"
    );

    // Scores should roughly sum to 1
    let sum = nli.entailment + nli.neutral + nli.contradiction;
    assert!(
        (sum - 1.0).abs() < 0.01,
        "scores should sum to ~1, got {}",
        sum
    );

    // This premise/hypothesis pair should have high entailment
    assert!(
        nli.entailment > 0.5,
        "entailment should be high for this pair, got {}",
        nli.entailment
    );
}
