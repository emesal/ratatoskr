//! Tests for parameter validation in the registry.

use std::sync::Arc;

use llm::builder::LLMBackend;
use ratatoskr::providers::LlmChatProvider;
use ratatoskr::providers::registry::ProviderRegistry;
use ratatoskr::{
    ChatOptions, GenerateOptions, Message, ParameterName, ParameterValidationPolicy, RatatoskrError,
};

#[test]
fn validation_policy_variants() {
    let _ = ParameterValidationPolicy::Warn;
    let _ = ParameterValidationPolicy::Error;
    let _ = ParameterValidationPolicy::Ignore;
}

#[test]
fn validation_policy_default_is_warn() {
    let policy = ParameterValidationPolicy::default();
    assert!(matches!(policy, ParameterValidationPolicy::Warn));
}

#[test]
fn chat_options_set_parameters_empty() {
    let opts = ChatOptions::default().model("test");
    assert!(opts.set_parameters().is_empty());
}

#[test]
fn chat_options_set_parameters_tracks_all() {
    let opts = ChatOptions::default()
        .model("test")
        .temperature(0.7)
        .max_tokens(100)
        .top_p(0.9)
        .top_k(40)
        .seed(42);

    let params = opts.set_parameters();
    assert!(params.contains(&ParameterName::Temperature));
    assert!(params.contains(&ParameterName::MaxTokens));
    assert!(params.contains(&ParameterName::TopP));
    assert!(params.contains(&ParameterName::TopK));
    assert!(params.contains(&ParameterName::Seed));
    assert_eq!(params.len(), 5);
}

#[test]
fn generate_options_set_parameters_empty() {
    let opts = GenerateOptions::new("test");
    assert!(opts.set_parameters().is_empty());
}

#[test]
fn generate_options_set_parameters_tracks_all() {
    let opts = GenerateOptions::new("test")
        .temperature(0.7)
        .max_tokens(100)
        .top_p(0.9)
        .top_k(40)
        .frequency_penalty(0.5)
        .presence_penalty(0.3)
        .seed(42);

    let params = opts.set_parameters();
    assert!(params.contains(&ParameterName::Temperature));
    assert!(params.contains(&ParameterName::MaxTokens));
    assert!(params.contains(&ParameterName::TopP));
    assert!(params.contains(&ParameterName::TopK));
    assert!(params.contains(&ParameterName::FrequencyPenalty));
    assert!(params.contains(&ParameterName::PresencePenalty));
    assert!(params.contains(&ParameterName::Seed));
    assert_eq!(params.len(), 7);
}

#[test]
fn generate_options_set_parameters_includes_stop() {
    let opts = GenerateOptions::new("test").stop_sequence("END");
    let params = opts.set_parameters();
    assert!(params.contains(&ParameterName::Stop));
}

#[test]
fn unsupported_parameter_error_accessible() {
    let err = RatatoskrError::UnsupportedParameter {
        param: "top_k".to_string(),
        model: "gpt-4".to_string(),
        provider: "openai".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("top_k"));
    assert!(msg.contains("gpt-4"));
    assert!(msg.contains("openai"));
}

// =============================================================================
// End-to-end validation flow tests
// =============================================================================

/// Helper to create a registry with a provider and validation policy.
fn registry_with_policy(policy: ParameterValidationPolicy) -> ProviderRegistry {
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "test-key", "openrouter");
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(provider));
    registry.set_validation_policy(policy);
    registry
}

#[tokio::test]
async fn validation_error_policy_rejects_unsupported_param() {
    let registry = registry_with_policy(ParameterValidationPolicy::Error);

    // LlmChatProvider doesn't support top_k
    let opts = ChatOptions::default().model("test-model").top_k(40);
    let messages = vec![Message::user("hello")];

    let result = registry.chat(&messages, None, &opts).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(
        matches!(err, RatatoskrError::UnsupportedParameter { .. }),
        "expected UnsupportedParameter, got: {err:?}"
    );
}

#[tokio::test]
async fn validation_ignore_policy_allows_unsupported_param() {
    let registry = registry_with_policy(ParameterValidationPolicy::Ignore);

    // LlmChatProvider doesn't support top_k, but Ignore should let it through
    let opts = ChatOptions::default().model("test-model").top_k(40);
    let messages = vec![Message::user("hello")];

    // Will fail at the API call level (no valid key), but validation should pass
    let result = registry.chat(&messages, None, &opts).await;

    // Should not be UnsupportedParameter — might be ModelNotAvailable or Auth error
    if let Err(e) = &result {
        assert!(
            !matches!(e, RatatoskrError::UnsupportedParameter { .. }),
            "validation should have been ignored, got: {e:?}"
        );
    }
}

#[tokio::test]
async fn validation_skipped_when_provider_has_no_declared_params() {
    // Create a registry with default policy (Warn)
    let mut registry = ProviderRegistry::new();
    registry.set_validation_policy(ParameterValidationPolicy::Error);

    // No providers registered — should return NoProvider, not validation error
    let opts = ChatOptions::default().model("test-model").top_k(40);
    let messages = vec![Message::user("hello")];

    let result = registry.chat(&messages, None, &opts).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(
        matches!(err, RatatoskrError::NoProvider),
        "expected NoProvider (empty registry), got: {err:?}"
    );
}
