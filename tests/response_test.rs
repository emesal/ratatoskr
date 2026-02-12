use ratatoskr::{ChatEvent, ChatResponse, FinishReason, Usage};

#[test]
fn test_chat_response_default() {
    let resp = ChatResponse::default();
    assert!(resp.content.is_empty());
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn test_usage_total() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        reasoning_tokens: Some(20),
    };
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn test_chat_event_variants() {
    let content = ChatEvent::Content("hello".into());
    assert!(matches!(content, ChatEvent::Content(_)));

    let tool_call_end = ChatEvent::ToolCallEnd { index: 0 };
    assert!(matches!(tool_call_end, ChatEvent::ToolCallEnd { index: 0 }));

    let done = ChatEvent::Done;
    assert!(matches!(done, ChatEvent::Done));
}

#[test]
fn test_finish_reason() {
    let stop = FinishReason::Stop;
    let tools = FinishReason::ToolCalls;
    assert!(matches!(stop, FinishReason::Stop));
    assert!(matches!(tools, FinishReason::ToolCalls));
}

// =============================================================================
// Serde roundtrip tests
// =============================================================================

#[test]
fn test_usage_serde_roundtrip() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        reasoning_tokens: Some(20),
    };
    let json = serde_json::to_string(&usage).unwrap();
    let deserialized: Usage = serde_json::from_str(&json).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_usage_serde_omits_none_reasoning() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        reasoning_tokens: None,
    };
    let json = serde_json::to_string(&usage).unwrap();
    assert!(!json.contains("reasoning_tokens"));
    let deserialized: Usage = serde_json::from_str(&json).unwrap();
    assert_eq!(usage, deserialized);
}

#[test]
fn test_chat_event_content_serde_roundtrip() {
    let event = ChatEvent::Content("hello world".into());
    let json = serde_json::to_string(&event).unwrap();
    let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, deserialized);
}

#[test]
fn test_chat_event_usage_serde_roundtrip() {
    let event = ChatEvent::Usage(Usage {
        prompt_tokens: 42,
        completion_tokens: 13,
        total_tokens: 55,
        reasoning_tokens: None,
    });
    let json = serde_json::to_string(&event).unwrap();
    let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, deserialized);
}

#[test]
fn test_chat_event_done_serde_roundtrip() {
    let event = ChatEvent::Done;
    let json = serde_json::to_string(&event).unwrap();
    let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, deserialized);
}

#[test]
fn test_chat_response_serde_roundtrip() {
    let resp = ChatResponse {
        content: "test response".into(),
        reasoning: Some("because".into()),
        tool_calls: vec![],
        usage: Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            reasoning_tokens: None,
        }),
        model: Some("test-model".into()),
        finish_reason: FinishReason::Stop,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: ChatResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp, deserialized);
}
