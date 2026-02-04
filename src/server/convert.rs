//! Conversions between ratatoskr native types and protobuf types.
//!
//! Server-side conversions: proto → native for requests, native → proto for responses.
//! Client-side conversions live in `crate::client::service_client`.

use crate::{
    ChatEvent, ChatOptions, ChatResponse, ClassifyResult, Embedding, FinishReason, GenerateEvent,
    GenerateOptions, GenerateResponse, Message, MessageContent, ModelCapability, ModelInfo,
    ModelStatus, NliLabel, NliResult, ReasoningConfig, ReasoningEffort, ResponseFormat, Role,
    StanceLabel, StanceResult, Token, ToolCall, ToolChoice, ToolDefinition, Usage,
};

use super::proto;

// =============================================================================
// From Proto → Native (incoming requests)
// =============================================================================

impl From<proto::Message> for Message {
    fn from(p: proto::Message) -> Self {
        let role = match proto::Role::try_from(p.role).unwrap_or(proto::Role::Unspecified) {
            proto::Role::System => Role::System,
            proto::Role::User => Role::User,
            proto::Role::Assistant => Role::Assistant,
            proto::Role::Tool => Role::Tool {
                tool_call_id: p.tool_call_id.unwrap_or_default(),
            },
            proto::Role::Unspecified => Role::User,
        };

        Message {
            role,
            content: MessageContent::Text(p.content),
            tool_calls: if p.tool_calls.is_empty() {
                None
            } else {
                Some(p.tool_calls.into_iter().map(Into::into).collect())
            },
            name: p.name,
        }
    }
}

impl From<proto::ToolCall> for ToolCall {
    fn from(p: proto::ToolCall) -> Self {
        ToolCall {
            id: p.id,
            name: p.name,
            arguments: p.arguments,
        }
    }
}

impl From<proto::ToolDefinition> for ToolDefinition {
    fn from(p: proto::ToolDefinition) -> Self {
        ToolDefinition {
            name: p.name,
            description: p.description,
            parameters: serde_json::from_str(&p.parameters_json).unwrap_or_default(),
        }
    }
}

impl From<proto::ChatOptions> for ChatOptions {
    fn from(p: proto::ChatOptions) -> Self {
        ChatOptions {
            model: p.model,
            temperature: p.temperature,
            max_tokens: p.max_tokens.map(|t| t as usize),
            top_p: p.top_p,
            stop: if p.stop.is_empty() { None } else { Some(p.stop) },
            frequency_penalty: p.frequency_penalty,
            presence_penalty: p.presence_penalty,
            seed: p.seed,
            tool_choice: p.tool_choice.map(Into::into),
            response_format: p.response_format.map(Into::into),
            cache_prompt: p.cache_prompt,
            reasoning: p.reasoning.map(Into::into),
            raw_provider_options: None,
        }
    }
}

impl From<proto::ToolChoice> for ToolChoice {
    fn from(p: proto::ToolChoice) -> Self {
        match p.choice {
            Some(proto::tool_choice::Choice::Auto(true)) => ToolChoice::Auto,
            Some(proto::tool_choice::Choice::None(true)) => ToolChoice::None,
            Some(proto::tool_choice::Choice::Required(true)) => ToolChoice::Required,
            Some(proto::tool_choice::Choice::Function(name)) => ToolChoice::Function { name },
            _ => ToolChoice::Auto,
        }
    }
}

impl From<proto::ResponseFormat> for ResponseFormat {
    fn from(p: proto::ResponseFormat) -> Self {
        match p.format {
            Some(proto::response_format::Format::Text(true)) => ResponseFormat::Text,
            Some(proto::response_format::Format::JsonObject(true)) => ResponseFormat::JsonObject,
            Some(proto::response_format::Format::JsonSchema(s)) => ResponseFormat::JsonSchema {
                schema: serde_json::from_str(&s).unwrap_or_default(),
            },
            _ => ResponseFormat::Text,
        }
    }
}

impl From<proto::ReasoningConfig> for ReasoningConfig {
    fn from(p: proto::ReasoningConfig) -> Self {
        ReasoningConfig {
            effort: p.effort.and_then(|e| {
                match proto::ReasoningEffort::try_from(e).ok()? {
                    proto::ReasoningEffort::Low => Some(ReasoningEffort::Low),
                    proto::ReasoningEffort::Medium => Some(ReasoningEffort::Medium),
                    proto::ReasoningEffort::High => Some(ReasoningEffort::High),
                    proto::ReasoningEffort::Unspecified => None,
                }
            }),
            max_tokens: p.max_tokens.map(|t| t as usize),
            exclude_from_output: p.exclude_from_output,
        }
    }
}

impl From<proto::GenerateOptions> for GenerateOptions {
    fn from(p: proto::GenerateOptions) -> Self {
        GenerateOptions {
            model: p.model,
            max_tokens: p.max_tokens.map(|t| t as usize),
            temperature: p.temperature,
            top_p: p.top_p,
            stop_sequences: p.stop_sequences,
        }
    }
}

// =============================================================================
// Native → Proto (outgoing responses)
// =============================================================================

impl From<Message> for proto::Message {
    fn from(m: Message) -> Self {
        let (role, tool_call_id) = match m.role {
            Role::System => (proto::Role::System as i32, None),
            Role::User => (proto::Role::User as i32, None),
            Role::Assistant => (proto::Role::Assistant as i32, None),
            Role::Tool { tool_call_id } => (proto::Role::Tool as i32, Some(tool_call_id)),
        };

        proto::Message {
            role,
            content: m.content.as_text().unwrap_or_default().to_string(),
            tool_calls: m
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
            name: m.name,
            tool_call_id,
        }
    }
}

impl From<ToolCall> for proto::ToolCall {
    fn from(t: ToolCall) -> Self {
        proto::ToolCall {
            id: t.id,
            name: t.name,
            arguments: t.arguments,
        }
    }
}

impl From<Usage> for proto::Usage {
    fn from(u: Usage) -> Self {
        proto::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u.reasoning_tokens,
        }
    }
}

impl From<FinishReason> for proto::FinishReason {
    fn from(f: FinishReason) -> Self {
        match f {
            FinishReason::Stop => proto::FinishReason::Stop,
            FinishReason::Length => proto::FinishReason::Length,
            FinishReason::ToolCalls => proto::FinishReason::ToolCalls,
            FinishReason::ContentFilter => proto::FinishReason::ContentFilter,
        }
    }
}

impl From<ChatResponse> for proto::ChatResponse {
    fn from(r: ChatResponse) -> Self {
        proto::ChatResponse {
            content: r.content,
            reasoning: r.reasoning,
            tool_calls: r.tool_calls.into_iter().map(Into::into).collect(),
            usage: r.usage.map(Into::into),
            model: r.model,
            finish_reason: proto::FinishReason::from(r.finish_reason) as i32,
        }
    }
}

impl From<ChatEvent> for proto::ChatEvent {
    fn from(e: ChatEvent) -> Self {
        let event = match e {
            ChatEvent::Content(s) => proto::chat_event::Event::Content(s),
            ChatEvent::Reasoning(s) => proto::chat_event::Event::Reasoning(s),
            ChatEvent::ToolCallStart { index, id, name } => {
                proto::chat_event::Event::ToolCallStart(proto::ToolCallStart {
                    index: index as u32,
                    id,
                    name,
                })
            }
            ChatEvent::ToolCallDelta { index, arguments } => {
                proto::chat_event::Event::ToolCallDelta(proto::ToolCallDelta {
                    index: index as u32,
                    arguments,
                })
            }
            ChatEvent::Usage(u) => proto::chat_event::Event::Usage(u.into()),
            ChatEvent::Done => proto::chat_event::Event::Done(true),
        };
        proto::ChatEvent { event: Some(event) }
    }
}

impl From<GenerateResponse> for proto::GenerateResponse {
    fn from(r: GenerateResponse) -> Self {
        proto::GenerateResponse {
            text: r.text,
            usage: r.usage.map(Into::into),
            model: r.model,
            finish_reason: proto::FinishReason::from(r.finish_reason) as i32,
        }
    }
}

impl From<GenerateEvent> for proto::GenerateEvent {
    fn from(e: GenerateEvent) -> Self {
        let event = match e {
            GenerateEvent::Text(s) => proto::generate_event::Event::Text(s),
            GenerateEvent::Done => proto::generate_event::Event::Done(true),
        };
        proto::GenerateEvent { event: Some(event) }
    }
}

impl From<Embedding> for proto::Embedding {
    fn from(e: Embedding) -> Self {
        proto::Embedding {
            values: e.values,
            model: e.model,
            dimensions: e.dimensions as u32,
        }
    }
}

impl From<Embedding> for proto::EmbedResponse {
    fn from(e: Embedding) -> Self {
        proto::EmbedResponse {
            values: e.values,
            model: e.model,
            dimensions: e.dimensions as u32,
        }
    }
}

impl From<NliResult> for proto::NliResponse {
    fn from(r: NliResult) -> Self {
        proto::NliResponse {
            entailment: r.entailment,
            contradiction: r.contradiction,
            neutral: r.neutral,
            label: match r.label {
                NliLabel::Entailment => proto::NliLabel::Entailment as i32,
                NliLabel::Contradiction => proto::NliLabel::Contradiction as i32,
                NliLabel::Neutral => proto::NliLabel::Neutral as i32,
            },
        }
    }
}

impl From<ClassifyResult> for proto::ClassifyResponse {
    fn from(r: ClassifyResult) -> Self {
        proto::ClassifyResponse {
            scores: r.scores,
            top_label: r.top_label,
            confidence: r.confidence,
        }
    }
}

impl From<StanceResult> for proto::StanceResponse {
    fn from(r: StanceResult) -> Self {
        proto::StanceResponse {
            favor: r.favor,
            against: r.against,
            neutral: r.neutral,
            label: match r.label {
                StanceLabel::Favor => proto::StanceLabel::Favor as i32,
                StanceLabel::Against => proto::StanceLabel::Against as i32,
                StanceLabel::Neutral => proto::StanceLabel::Neutral as i32,
            },
            target: r.target,
        }
    }
}

impl From<Token> for proto::Token {
    fn from(t: Token) -> Self {
        proto::Token {
            id: t.id,
            text: t.text,
            start: t.start as u32,
            end: t.end as u32,
        }
    }
}

impl From<ModelInfo> for proto::ModelInfo {
    fn from(m: ModelInfo) -> Self {
        proto::ModelInfo {
            id: m.id,
            provider: m.provider,
            capabilities: m
                .capabilities
                .into_iter()
                .map(|c| match c {
                    ModelCapability::Chat => proto::ModelCapability::Chat as i32,
                    ModelCapability::Generate => proto::ModelCapability::Generate as i32,
                    ModelCapability::Embed => proto::ModelCapability::Embed as i32,
                    ModelCapability::Nli => proto::ModelCapability::Nli as i32,
                    ModelCapability::Classify | ModelCapability::Stance => {
                        proto::ModelCapability::Classify as i32
                    }
                })
                .collect(),
            context_window: m.context_window.map(|w| w as u32),
            dimensions: m.dimensions.map(|d| d as u32),
        }
    }
}

impl From<ModelStatus> for proto::ModelStatusResponse {
    fn from(s: ModelStatus) -> Self {
        match s {
            ModelStatus::Available => proto::ModelStatusResponse {
                status: proto::ModelStatus::Available as i32,
                reason: None,
            },
            ModelStatus::Loading => proto::ModelStatusResponse {
                status: proto::ModelStatus::Loading as i32,
                reason: None,
            },
            ModelStatus::Ready => proto::ModelStatusResponse {
                status: proto::ModelStatus::Ready as i32,
                reason: None,
            },
            ModelStatus::Unavailable { reason } => proto::ModelStatusResponse {
                status: proto::ModelStatus::Unavailable as i32,
                reason: Some(reason),
            },
        }
    }
}
