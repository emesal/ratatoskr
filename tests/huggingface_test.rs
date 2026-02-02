//! Unit tests for HuggingFace response parsing
#![cfg(feature = "huggingface")]

use ratatoskr::Embedding;

#[test]
fn test_parse_embedding_response() {
    // HuggingFace returns nested array for single input: [[f32; dim]]
    let json = r#"[[0.1, 0.2, 0.3, 0.4, 0.5]]"#;
    let parsed: Vec<Vec<f32>> = serde_json::from_str(json).unwrap();

    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].len(), 5);
    assert!((parsed[0][0] - 0.1).abs() < 0.001);
}

#[test]
fn test_parse_batch_embedding_response() {
    // HuggingFace returns [[[f32; dim]]] for batch (extra nesting)
    let json = r#"[[[0.1, 0.2]], [[0.3, 0.4]]]"#;
    let parsed: Vec<Vec<Vec<f32>>> = serde_json::from_str(json).unwrap();

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0][0].len(), 2);
}

#[test]
fn test_embedding_to_struct() {
    let values = vec![0.1, 0.2, 0.3];
    let embedding = Embedding {
        values: values.clone(),
        model: "test-model".to_string(),
        dimensions: values.len(),
    };

    assert_eq!(embedding.dimensions, 3);
    assert_eq!(embedding.model, "test-model");
}
