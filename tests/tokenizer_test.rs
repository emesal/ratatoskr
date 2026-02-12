//! Tests for token counting functionality.

#![cfg(feature = "local-inference")]

use ratatoskr::tokenizer::{TokenizerRegistry, TokenizerSource};

#[test]
fn test_tokenizer_registry_default_mappings() {
    let registry = TokenizerRegistry::new();

    // Check that default mappings exist
    let claude_result = registry.resolve_source("claude-sonnet-4");
    assert!(claude_result.is_ok());

    let gpt4_result = registry.resolve_source("gpt-4-turbo");
    assert!(gpt4_result.is_ok());

    let llama_result = registry.resolve_source("llama-3-70b");
    assert!(llama_result.is_ok());
}

#[test]
fn test_tokenizer_registry_custom_mapping() {
    let mut registry = TokenizerRegistry::new();

    // Add a custom mapping
    registry.register(
        "my-custom-model",
        TokenizerSource::HuggingFace {
            repo_id: "custom/tokenizer".to_string(),
        },
    );

    let result = registry.resolve_source("my-custom-model");
    assert!(result.is_ok());

    if let Ok(TokenizerSource::HuggingFace { repo_id }) = result {
        assert_eq!(repo_id, "custom/tokenizer");
    } else {
        panic!("Expected HuggingFace source");
    }
}

#[test]
fn test_tokenizer_registry_alias() {
    let mut registry = TokenizerRegistry::new();

    // Add an alias
    registry.register(
        "my-alias",
        TokenizerSource::Alias {
            target: "claude".to_string(),
        },
    );

    let result = registry.resolve_source("my-alias");
    assert!(result.is_ok());

    // Should resolve to claude's tokenizer
    if let Ok(source) = result {
        match source {
            TokenizerSource::HuggingFace { repo_id } => {
                assert_eq!(repo_id, "Xenova/claude-tokenizer");
            }
            _ => panic!("Expected HuggingFace source after alias resolution"),
        }
    }
}

#[test]
fn test_tokenizer_registry_longest_match() {
    let mut registry = TokenizerRegistry::new();

    // Add mappings with different prefix lengths
    registry.register(
        "model",
        TokenizerSource::HuggingFace {
            repo_id: "short/match".to_string(),
        },
    );
    registry.register(
        "model/family",
        TokenizerSource::HuggingFace {
            repo_id: "long/match".to_string(),
        },
    );

    // Should match the longer prefix
    let result = registry.resolve_source("model/family/specific-v1");
    assert!(result.is_ok());

    if let Ok(TokenizerSource::HuggingFace { repo_id }) = result {
        assert_eq!(repo_id, "long/match");
    } else {
        panic!("Expected HuggingFace source");
    }
}

#[test]
fn test_tokenizer_registry_no_match() {
    let registry = TokenizerRegistry::new();

    let result = registry.resolve_source("unknown-model-family");
    assert!(result.is_err());
}

#[test]
fn test_tokenizer_alias_cycle_detected() {
    let mut registry = TokenizerRegistry::new();

    // Create a cycle: a → b → a
    registry.register(
        "alias-a",
        TokenizerSource::Alias {
            target: "alias-b".to_string(),
        },
    );
    registry.register(
        "alias-b",
        TokenizerSource::Alias {
            target: "alias-a".to_string(),
        },
    );

    let result = registry.resolve_source("alias-a");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("cycle") || err.contains("Alias"),
        "expected cycle error, got: {err}"
    );
}

#[test]
fn test_tokenizer_prefix_match_over_exact() {
    let registry = TokenizerRegistry::new();

    // "gpt-4-turbo-preview" should match "gpt-4" prefix from defaults
    let result = registry.resolve_source("gpt-4-turbo-preview");
    assert!(result.is_ok());
}

// Note: This test is commented out as it requires network access to HuggingFace Hub.
// Enable it for manual testing if needed.
//
// #[test]
// #[ignore]
// fn test_hf_tokenizer_from_hub() {
//     use ratatoskr::tokenizer::HfTokenizer;
//
//     let tokenizer = HfTokenizer::from_hub("Xenova/gpt-4").unwrap();
//     let text = "Hello, world!";
//     let count = tokenizer.count_tokens(text).unwrap();
//     assert!(count > 0);
//
//     let tokens = tokenizer.tokenize(text).unwrap();
//     assert_eq!(tokens.len(), count);
// }
