use ratatoskr::{ToolCall, ToolChoice, ToolDefinition};
use serde_json::json;

#[test]
fn test_tool_definition_new() {
    let tool = ToolDefinition::new(
        "get_weather",
        "Get current weather",
        json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }),
    );
    assert_eq!(tool.name, "get_weather");
}

#[test]
fn test_tool_definition_from_openai_format() {
    let json_tool = json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }
        }
    });

    let tool = ToolDefinition::try_from(&json_tool).unwrap();
    assert_eq!(tool.name, "search");
    assert_eq!(tool.description, "Search the web");
}

#[test]
fn test_tool_call_default() {
    let call = ToolCall::default();
    assert!(call.id.is_empty());
    assert!(call.name.is_empty());
    assert!(call.arguments.is_empty());
}

#[test]
fn test_tool_choice_default() {
    let choice = ToolChoice::default();
    assert!(matches!(choice, ToolChoice::Auto));
}
