use ratatoskr::{ChatOptions, Message, Role, ToolDefinition};
use serde_json::json;

// These are internal conversions, but we can test via round-trip behavior
// For unit tests, we'll test the public interface behavior

#[test]
fn test_message_system_role() {
    let msg = Message::system("You are helpful");
    assert!(matches!(msg.role, Role::System));
}

#[test]
fn test_tool_definition_roundtrip() {
    let tool = ToolDefinition::new(
        "test_tool",
        "A test tool",
        json!({"type": "object", "properties": {}}),
    );

    // Serialize and deserialize
    let json = serde_json::to_string(&tool).unwrap();
    let parsed: ToolDefinition = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.name, "test_tool");
    assert_eq!(parsed.description, "A test tool");
}

#[test]
fn test_chat_options_roundtrip() {
    let opts = ChatOptions::default().model("gpt-4").temperature(0.7);

    let json = serde_json::to_string(&opts).unwrap();
    let parsed: ChatOptions = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.temperature, Some(0.7));
}
