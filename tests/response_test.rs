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
