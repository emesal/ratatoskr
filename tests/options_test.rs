use ratatoskr::{ChatOptions, ReasoningConfig, ReasoningEffort, ResponseFormat};

#[test]
fn test_chat_options_top_k() {
    let opts = ChatOptions::default().model("gpt-4").top_k(40);
    assert_eq!(opts.top_k, Some(40));
}

#[test]
fn test_chat_options_top_k_serde() {
    let opts = ChatOptions::default().model("test").top_k(50);
    let json = serde_json::to_string(&opts).unwrap();
    let parsed: ChatOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.top_k, Some(50));
}

#[test]
fn test_chat_options_default() {
    let opts = ChatOptions::default();
    assert!(opts.model.is_empty());
    assert!(opts.temperature.is_none());
}

#[test]
fn test_chat_options_builder_style() {
    let opts = ChatOptions::default()
        .model("gpt-4")
        .temperature(0.7)
        .max_tokens(1000);

    assert_eq!(opts.model, "gpt-4");
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.max_tokens, Some(1000));
}

#[test]
fn test_reasoning_config() {
    let cfg = ReasoningConfig {
        effort: Some(ReasoningEffort::High),
        max_tokens: Some(8000),
        exclude_from_output: Some(false),
    };
    assert!(matches!(cfg.effort, Some(ReasoningEffort::High)));
}

#[test]
fn test_response_format_variants() {
    let text = ResponseFormat::Text;
    let json_obj = ResponseFormat::JsonObject;
    assert!(matches!(text, ResponseFormat::Text));
    assert!(matches!(json_obj, ResponseFormat::JsonObject));
}
