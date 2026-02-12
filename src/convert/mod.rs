//! Conversions between ratatoskr types and llm crate types.
//!
//! This module is internal and handles the translation layer between
//! our stable public types and the llm crate's internal types.

use llm::chat::ChatMessage as LlmMessage;

use crate::{Message, MessageContent, Role, ToolCall, Usage as RataUsage};

/// Extract text content from a message, returning an error for non-text variants.
///
/// `MessageContent` is `#[non_exhaustive]` and will gain variants (e.g. `Image`,
/// `MultiPart`). The wildcard arm ensures graceful failure when that happens.
fn extract_text(content: &MessageContent, role: &str) -> crate::Result<String> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        #[allow(unreachable_patterns)]
        other => Err(crate::RatatoskrError::InvalidInput(format!(
            "{role} message content type not supported for llm conversion: {other:?}"
        ))),
    }
}

/// Convert our messages to llm crate messages
pub fn to_llm_messages(messages: &[Message]) -> crate::Result<(Option<String>, Vec<LlmMessage>)> {
    let mut system_prompt = None;
    let mut llm_messages = Vec::with_capacity(messages.len());

    for msg in messages {
        match &msg.role {
            Role::System => {
                let text = extract_text(&msg.content, "system")?;
                system_prompt = Some(text);
            }
            Role::User => {
                let text = extract_text(&msg.content, "user")?;
                llm_messages.push(LlmMessage::user().content(text).build());
            }
            Role::Assistant => {
                if let Some(tool_calls) = &msg.tool_calls {
                    // Assistant message with tool calls
                    let llm_tool_calls: Vec<llm::ToolCall> = tool_calls
                        .iter()
                        .map(|tc| llm::ToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: llm::FunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            },
                        })
                        .collect();

                    let content = msg.content.as_text().unwrap_or_default();
                    llm_messages.push(
                        LlmMessage::assistant()
                            .tool_use(llm_tool_calls)
                            .content(content)
                            .build(),
                    );
                } else {
                    let text = extract_text(&msg.content, "assistant")?;
                    llm_messages.push(LlmMessage::assistant().content(text).build());
                }
            }
            Role::Tool { tool_call_id } => {
                let text = extract_text(&msg.content, "tool")?;
                llm_messages.push(
                    LlmMessage::user()
                        .tool_result(vec![llm::ToolCall {
                            id: tool_call_id.clone(),
                            call_type: "function".to_string(),
                            function: llm::FunctionCall {
                                name: String::new(), // Not needed for result
                                arguments: text,
                            },
                        }])
                        .build(),
                );
            }
        }
    }

    Ok((system_prompt, llm_messages))
}

/// Convert llm crate tool calls to our format
pub fn from_llm_tool_calls(calls: &[llm::ToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .map(|c| ToolCall {
            id: c.id.clone(),
            name: c.function.name.clone(),
            arguments: c.function.arguments.clone(),
        })
        .collect()
}

/// Convert llm crate usage to our format
pub fn from_llm_usage(usage: &llm::chat::Usage) -> RataUsage {
    RataUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.prompt_tokens + usage.completion_tokens,
        reasoning_tokens: None, // llm crate doesn't expose this yet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_message_extracted_as_system_prompt() {
        let messages = vec![Message::system("you are helpful"), Message::user("hello")];
        let (system, llm_msgs) = to_llm_messages(&messages).unwrap();
        assert_eq!(system, Some("you are helpful".to_string()));
        assert_eq!(llm_msgs.len(), 1);
    }

    #[test]
    fn no_system_message_returns_none() {
        let messages = vec![Message::user("hello")];
        let (system, llm_msgs) = to_llm_messages(&messages).unwrap();
        assert!(system.is_none());
        assert_eq!(llm_msgs.len(), 1);
    }

    #[test]
    fn multi_role_conversation() {
        let messages = vec![
            Message::system("be helpful"),
            Message::user("hi"),
            Message::assistant("hello!"),
            Message::user("how are you?"),
        ];
        let (system, llm_msgs) = to_llm_messages(&messages).unwrap();
        assert_eq!(system, Some("be helpful".to_string()));
        assert_eq!(llm_msgs.len(), 3); // user, assistant, user (system extracted)
    }

    #[test]
    fn tool_call_conversion() {
        let tool_calls = vec![ToolCall {
            id: "call_123".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"city":"london"}"#.to_string(),
        }];
        let messages = vec![Message::assistant_with_tool_calls(None, tool_calls)];
        let (_, llm_msgs) = to_llm_messages(&messages).unwrap();
        assert_eq!(llm_msgs.len(), 1);
    }

    #[test]
    fn tool_result_message() {
        let messages = vec![Message::tool_result("call_123", r#"{"temp": 15}"#)];
        let (_, llm_msgs) = to_llm_messages(&messages).unwrap();
        assert_eq!(llm_msgs.len(), 1);
    }

    #[test]
    fn from_llm_tool_calls_roundtrip() {
        let llm_calls = vec![llm::ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: llm::FunctionCall {
                name: "search".to_string(),
                arguments: r#"{"q":"rust"}"#.to_string(),
            },
        }];

        let ours = from_llm_tool_calls(&llm_calls);
        assert_eq!(ours.len(), 1);
        assert_eq!(ours[0].id, "call_1");
        assert_eq!(ours[0].name, "search");
        assert_eq!(ours[0].arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn from_llm_usage_sums_total() {
        let llm_usage = llm::chat::Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };
        let usage = from_llm_usage(&llm_usage);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert!(usage.reasoning_tokens.is_none());
    }

    #[test]
    fn empty_message_list() {
        let (system, llm_msgs) = to_llm_messages(&[]).unwrap();
        assert!(system.is_none());
        assert!(llm_msgs.is_empty());
    }

    #[test]
    fn from_llm_tool_calls_empty() {
        let result = from_llm_tool_calls(&[]);
        assert!(result.is_empty());
    }
}
