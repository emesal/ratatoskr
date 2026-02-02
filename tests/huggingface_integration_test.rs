//! Wiremock integration tests for HuggingFaceClient.
//!
//! These tests verify correct HTTP interaction and error handling using mocked responses.
#![cfg(feature = "huggingface")]

use ratatoskr::{RatatoskrError, providers::HuggingFaceClient};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Test successful single embedding request.
#[tokio::test]
async fn test_embed_success() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    // HuggingFace returns [[f32; dim]] for single input
    let embedding_response = serde_json::json!([[0.1, 0.2, 0.3, 0.4, 0.5]]);

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(embedding_response))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client.embed("hello world", model).await;

    let embedding = result.expect("embed should succeed");
    assert_eq!(embedding.dimensions, 5);
    assert_eq!(embedding.model, model);
    assert!((embedding.values[0] - 0.1).abs() < 0.001);
    assert!((embedding.values[4] - 0.5).abs() < 0.001);
}

/// Test successful batch embedding request.
#[tokio::test]
async fn test_embed_batch_success() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    // HuggingFace returns [[[f32; dim]]] for batch (extra nesting)
    let batch_response = serde_json::json!([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]);

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client.embed_batch(&["hello", "world"], model).await;

    let embeddings = result.expect("embed_batch should succeed");
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].dimensions, 3);
    assert_eq!(embeddings[1].dimensions, 3);
    assert!((embeddings[0].values[0] - 0.1).abs() < 0.001);
    assert!((embeddings[1].values[0] - 0.4).abs() < 0.001);
}

/// Test successful zero-shot classification request.
#[tokio::test]
async fn test_classify_success() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    // HuggingFace zero-shot returns labels and scores
    let classify_response = serde_json::json!({
        "labels": ["positive", "negative", "neutral"],
        "scores": [0.85, 0.10, 0.05]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(classify_response))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client
        .classify("I love this!", &["positive", "negative", "neutral"], model)
        .await;

    let classification = result.expect("classify should succeed");
    assert_eq!(classification.top_label, "positive");
    assert!((classification.confidence - 0.85).abs() < 0.001);
    assert_eq!(classification.scores.len(), 3);
}

/// Test successful NLI inference request.
#[tokio::test]
async fn test_infer_nli_success() {
    let mock_server = MockServer::start().await;
    let model = "facebook/bart-large-mnli";

    // NLI via zero-shot returns entailment/neutral/contradiction scores
    let nli_response = serde_json::json!({
        "labels": ["entailment", "neutral", "contradiction"],
        "scores": [0.8, 0.15, 0.05]
    });

    Mock::given(method("POST"))
        .and(path(format!("/models/{}", model)))
        .and(header("Authorization", "Bearer test_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(nli_response))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client
        .infer_nli("The cat sat on the mat.", "A feline is on a rug.", model)
        .await;

    let nli = result.expect("infer_nli should succeed");
    assert!((nli.entailment - 0.8).abs() < 0.001);
    assert!((nli.neutral - 0.15).abs() < 0.001);
    assert!((nli.contradiction - 0.05).abs() < 0.001);
    assert!(matches!(nli.label, ratatoskr::NliLabel::Entailment));
}

/// Test 401 Unauthorized returns AuthenticationFailed error.
#[tokio::test]
async fn test_error_401_unauthorized() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .respond_with(ResponseTemplate::new(401))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("bad_key", mock_server.uri());
    let result = client.embed("hello", model).await;

    assert!(
        matches!(result, Err(RatatoskrError::AuthenticationFailed)),
        "expected AuthenticationFailed, got {:?}",
        result
    );
}

/// Test 404 Not Found returns ModelNotFound error.
#[tokio::test]
async fn test_error_404_model_not_found() {
    let mock_server = MockServer::start().await;
    let model = "nonexistent/model";

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .respond_with(ResponseTemplate::new(404))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client.embed("hello", model).await;

    match result {
        Err(RatatoskrError::ModelNotFound(m)) => assert_eq!(m, model),
        other => panic!("expected ModelNotFound, got {:?}", other),
    }
}

/// Test 429 Too Many Requests returns RateLimited error with retry-after.
#[tokio::test]
async fn test_error_429_rate_limited() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "30"))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client.embed("hello", model).await;

    match result {
        Err(RatatoskrError::RateLimited { retry_after }) => {
            assert_eq!(
                retry_after,
                Some(std::time::Duration::from_secs(30)),
                "retry_after should be 30 seconds"
            );
        }
        other => panic!("expected RateLimited, got {:?}", other),
    }
}

/// Test 503 Service Unavailable (model loading) returns Api error.
#[tokio::test]
async fn test_error_503_model_loading() {
    let mock_server = MockServer::start().await;
    let model = "sentence-transformers/all-MiniLM-L6-v2";

    Mock::given(method("POST"))
        .and(path(format!("/pipeline/feature-extraction/{}", model)))
        .respond_with(ResponseTemplate::new(503))
        .mount(&mock_server)
        .await;

    let client = HuggingFaceClient::with_base_url("test_key", mock_server.uri());
    let result = client.embed("hello", model).await;

    match result {
        Err(RatatoskrError::Api { status, message }) => {
            assert_eq!(status, 503);
            assert!(message.contains("loading") || message.contains("retry"));
        }
        other => panic!("expected Api {{ status: 503 }}, got {:?}", other),
    }
}
