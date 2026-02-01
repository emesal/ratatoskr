use ratatoskr::{Message, MessageContent, Role};

#[test]
fn test_message_constructors() {
    let sys = Message::system("You are helpful");
    assert!(matches!(sys.role, Role::System));

    let user = Message::user("Hello");
    assert!(matches!(user.role, Role::User));

    let asst = Message::assistant("Hi there!");
    assert!(matches!(asst.role, Role::Assistant));
}

#[test]
fn test_tool_result_message() {
    let tool = Message::tool_result("call_123", "result data");
    assert!(matches!(tool.role, Role::Tool { tool_call_id } if tool_call_id == "call_123"));
}

#[test]
fn test_message_content_text() {
    let msg = Message::user("test content");
    match msg.content {
        MessageContent::Text(s) => assert_eq!(s, "test content"),
    }
}
