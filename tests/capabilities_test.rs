use ratatoskr::{Capabilities, Embedding, NliLabel, NliResult};

#[test]
fn test_capabilities_default() {
    let caps = Capabilities::default();
    assert!(!caps.chat);
    assert!(!caps.embeddings);
}

#[test]
fn test_capabilities_chat_only() {
    let caps = Capabilities::chat_only();
    assert!(caps.chat);
    assert!(caps.chat_streaming);
    assert!(!caps.embeddings);
}

#[test]
fn test_embedding_stub() {
    let emb = Embedding {
        values: vec![0.1, 0.2, 0.3],
        model: "text-embedding-3".into(),
        dimensions: 3,
    };
    assert_eq!(emb.dimensions, 3);
}

#[test]
fn test_nli_result_stub() {
    let nli = NliResult {
        entailment: 0.8,
        contradiction: 0.1,
        neutral: 0.1,
        label: NliLabel::Entailment,
    };
    assert!(matches!(nli.label, NliLabel::Entailment));
}
