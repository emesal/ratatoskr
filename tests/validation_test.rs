//! Tests for parameter validation in the registry.

use ratatoskr::{
    ChatOptions, GenerateOptions, ParameterName, ParameterValidationPolicy, RatatoskrError,
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
