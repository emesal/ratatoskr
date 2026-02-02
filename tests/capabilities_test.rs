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

#[test]
fn test_capabilities_huggingface_only() {
    let caps = Capabilities::huggingface_only();
    assert!(!caps.chat, "huggingface_only should not have chat");
    assert!(
        !caps.chat_streaming,
        "huggingface_only should not have chat_streaming"
    );
    assert!(caps.embeddings, "huggingface_only should have embeddings");
    assert!(caps.nli, "huggingface_only should have nli");
    assert!(
        caps.classification,
        "huggingface_only should have classification"
    );
    assert!(
        !caps.token_counting,
        "huggingface_only should not have token_counting"
    );
}

#[test]
fn test_capabilities_merge() {
    let chat = Capabilities::chat_only();
    let hf = Capabilities::huggingface_only();
    let merged = chat.merge(&hf);

    // Should have both chat and HF capabilities
    assert!(merged.chat, "merged should have chat");
    assert!(merged.chat_streaming, "merged should have chat_streaming");
    assert!(merged.embeddings, "merged should have embeddings");
    assert!(merged.nli, "merged should have nli");
    assert!(merged.classification, "merged should have classification");
    assert!(
        !merged.token_counting,
        "merged should not have token_counting (neither source has it)"
    );
}

#[test]
fn test_capabilities_merge_is_symmetric() {
    let a = Capabilities::chat_only();
    let b = Capabilities::huggingface_only();

    let ab = a.merge(&b);
    let ba = b.merge(&a);

    // Merge should be symmetric (a.merge(b) == b.merge(a))
    assert_eq!(ab.chat, ba.chat);
    assert_eq!(ab.chat_streaming, ba.chat_streaming);
    assert_eq!(ab.embeddings, ba.embeddings);
    assert_eq!(ab.nli, ba.nli);
    assert_eq!(ab.classification, ba.classification);
    assert_eq!(ab.token_counting, ba.token_counting);
}
