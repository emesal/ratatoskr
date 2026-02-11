//! Conversions between ratatoskr types and llm crate types.
//!
//! This module is internal and handles the translation layer between
//! our stable public types and the llm crate's internal types.

use llm::chat::ChatMessage as LlmMessage;

use crate::{Message, MessageContent, Role, ToolCall, Usage as RataUsage};

/// Convert our messages to llm crate messages
pub fn to_llm_messages(messages: &[Message]) -> (Option<String>, Vec<LlmMessage>) {
    let mut system_prompt = None;
    let mut llm_messages = Vec::with_capacity(messages.len());

    for msg in messages {
        match &msg.role {
            Role::System => {
                // llm crate handles system separately via builder
                let MessageContent::Text(text) = &msg.content;
                system_prompt = Some(text.clone());
            }
            Role::User => {
                let MessageContent::Text(text) = &msg.content;
                llm_messages.push(LlmMessage::user().content(text.clone()).build());
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
                    let MessageContent::Text(text) = &msg.content;
                    llm_messages.push(LlmMessage::assistant().content(text.clone()).build());
                }
            }
            Role::Tool { tool_call_id } => {
                let MessageContent::Text(text) = &msg.content;
                llm_messages.push(
                    LlmMessage::user()
                        .tool_result(vec![llm::ToolCall {
                            id: tool_call_id.clone(),
                            call_type: "function".to_string(),
                            function: llm::FunctionCall {
                                name: String::new(), // Not needed for result
                                arguments: text.clone(),
                            },
                        }])
                        .build(),
                );
            }
        }
    }

    (system_prompt, llm_messages)
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
