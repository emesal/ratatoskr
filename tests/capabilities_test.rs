use ratatoskr::{Capabilities, Embedding, ModelCapability, NliLabel, NliResult};

#[test]
fn test_capabilities_default() {
    let caps = Capabilities::default();
    assert!(!caps.has(ModelCapability::Chat));
    assert!(!caps.has(ModelCapability::Embed));
}

#[test]
fn test_capabilities_chat_only() {
    let caps = Capabilities::chat_only();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::ChatStreaming));
    assert!(caps.has(ModelCapability::Generate));
    assert!(caps.has(ModelCapability::ToolUse));
    assert!(!caps.has(ModelCapability::Embed));
    assert!(!caps.has(ModelCapability::LocalInference));
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
    assert!(!caps.has(ModelCapability::Chat), "huggingface_only should not have chat");
    assert!(
        !caps.has(ModelCapability::ChatStreaming),
        "huggingface_only should not have chat_streaming"
    );
    assert!(caps.has(ModelCapability::Embed), "huggingface_only should have embeddings");
    assert!(caps.has(ModelCapability::Nli), "huggingface_only should have nli");
    assert!(caps.has(ModelCapability::Classify), "huggingface_only should have classification");
    assert!(
        !caps.has(ModelCapability::TokenCounting),
        "huggingface_only should not have token_counting"
    );
}

#[test]
fn test_capabilities_merge() {
    let chat = Capabilities::chat_only();
    let hf = Capabilities::huggingface_only();
    let merged = chat.merge(&hf);

    assert!(merged.has(ModelCapability::Chat), "merged should have chat");
    assert!(merged.has(ModelCapability::ChatStreaming), "merged should have chat_streaming");
    assert!(merged.has(ModelCapability::Embed), "merged should have embeddings");
    assert!(merged.has(ModelCapability::Nli), "merged should have nli");
    assert!(merged.has(ModelCapability::Classify), "merged should have classification");
    assert!(
        !merged.has(ModelCapability::TokenCounting),
        "merged should not have token_counting (neither source has it)"
    );
}

#[test]
fn test_capabilities_merge_is_symmetric() {
    let a = Capabilities::chat_only();
    let b = Capabilities::huggingface_only();

    let ab = a.merge(&b);
    let ba = b.merge(&a);

    // Merge should be symmetric
    assert_eq!(ab, ba);
}

#[test]
fn test_capabilities_local_only() {
    let caps = Capabilities::local_only();
    assert!(!caps.has(ModelCapability::Chat), "local_only should not have chat");
    assert!(
        !caps.has(ModelCapability::ChatStreaming),
        "local_only should not have chat_streaming"
    );
    assert!(!caps.has(ModelCapability::Generate), "local_only should not have generate");
    assert!(!caps.has(ModelCapability::ToolUse), "local_only should not have tool_use");
    assert!(caps.has(ModelCapability::Embed), "local_only should have embeddings");
    assert!(caps.has(ModelCapability::Nli), "local_only should have nli");
    assert!(!caps.has(ModelCapability::Classify), "local_only should not have classification");
    assert!(caps.has(ModelCapability::TokenCounting), "local_only should have token_counting");
    assert!(
        caps.has(ModelCapability::LocalInference),
        "local_only should have local_inference"
    );
}

#[test]
fn test_capabilities_full() {
    let caps = Capabilities::full();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::ChatStreaming));
    assert!(caps.has(ModelCapability::Generate));
    assert!(caps.has(ModelCapability::ToolUse));
    assert!(caps.has(ModelCapability::Embed));
    assert!(caps.has(ModelCapability::Nli));
    assert!(caps.has(ModelCapability::Classify));
    assert!(caps.has(ModelCapability::TokenCounting));
    assert!(caps.has(ModelCapability::LocalInference));
}

#[test]
fn test_capabilities_insert() {
    let mut caps = Capabilities::default();
    assert!(!caps.has(ModelCapability::Chat));
    caps.insert(ModelCapability::Chat);
    assert!(caps.has(ModelCapability::Chat));
}

#[test]
fn test_capabilities_from_iter() {
    let caps: Capabilities = [ModelCapability::Chat, ModelCapability::Embed]
        .into_iter()
        .collect();
    assert!(caps.has(ModelCapability::Chat));
    assert!(caps.has(ModelCapability::Embed));
    assert!(!caps.has(ModelCapability::Nli));
}

#[test]
fn test_capabilities_roundtrip_hashset() {
    use std::collections::HashSet;
    let original = Capabilities::chat_only();
    let set: HashSet<ModelCapability> = original.clone().into();
    let roundtripped = Capabilities::from(set);
    assert_eq!(original, roundtripped);
}
