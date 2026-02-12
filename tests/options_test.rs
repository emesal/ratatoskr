use ratatoskr::{ChatOptions, ParameterName, ReasoningConfig, ReasoningEffort, ResponseFormat};

#[test]
fn test_chat_options_top_k() {
    let opts = ChatOptions::new("gpt-4").top_k(40);
    assert_eq!(opts.top_k, Some(40));
}

#[test]
fn test_chat_options_top_k_serde() {
    let opts = ChatOptions::new("test").top_k(50);
    let json = serde_json::to_string(&opts).unwrap();
    let parsed: ChatOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.top_k, Some(50));
}

#[test]
fn test_chat_options_new() {
    let opts = ChatOptions::new("test-model");
    assert_eq!(opts.model, "test-model");
    assert!(opts.temperature.is_none());
}

#[test]
fn test_chat_options_builder_style() {
    let opts = ChatOptions::new("gpt-4").temperature(0.7).max_tokens(1000);

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

#[test]
fn test_chat_options_parallel_tool_calls_builder() {
    let opts = ChatOptions::new("gpt-4o").parallel_tool_calls(false);
    assert_eq!(opts.parallel_tool_calls, Some(false));
}

#[test]
fn test_chat_options_parallel_tool_calls_serde() {
    let opts = ChatOptions::new("gpt-4o").parallel_tool_calls(true);
    let json = serde_json::to_string(&opts).unwrap();
    assert!(json.contains("\"parallel_tool_calls\":true"));
    let parsed: ChatOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.parallel_tool_calls, Some(true));
}

#[test]
fn test_chat_options_parallel_tool_calls_in_set_parameters() {
    let opts = ChatOptions::new("gpt-4o").parallel_tool_calls(true);
    let params = opts.set_parameters();
    assert!(params.contains(&ParameterName::ParallelToolCalls));

    // absent when not set
    let opts_none = ChatOptions::new("gpt-4o");
    let params_none = opts_none.set_parameters();
    assert!(!params_none.contains(&ParameterName::ParallelToolCalls));
}
