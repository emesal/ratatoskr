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

// ===== Phase 6: ModelMetadata proto roundtrip =====

#[test]
#[cfg(any(feature = "server", feature = "client"))]
fn test_model_metadata_proto_roundtrip() {
    use ratatoskr::server::proto;
    use ratatoskr::{
        ModelCapability, ModelInfo, ModelMetadata, ParameterAvailability, ParameterName,
        ParameterRange, PricingInfo,
    };

    // build a rich ModelMetadata
    let original = ModelMetadata::from_info(
        ModelInfo::new("openai/gpt-4o", "openrouter")
            .with_capability(ModelCapability::Chat)
            .with_context_window(128_000),
    )
    .with_parameter(
        ParameterName::Temperature,
        ParameterAvailability::Mutable {
            range: ParameterRange::new().min(0.0).max(2.0).default_value(1.0),
        },
    )
    .with_parameter(ParameterName::TopK, ParameterAvailability::Unsupported)
    .with_parameter(ParameterName::Seed, ParameterAvailability::Opaque)
    .with_parameter(
        ParameterName::Custom("logit_bias".into()),
        ParameterAvailability::ReadOnly { value: json!(0.5) },
    )
    .with_pricing(PricingInfo {
        prompt_cost_per_mtok: Some(2.50),
        completion_cost_per_mtok: Some(10.0),
    })
    .with_max_output_tokens(16384);

    // native → proto → native
    let proto_meta: proto::ProtoModelMetadata = original.clone().into();
    let roundtripped: ModelMetadata = proto_meta.into();

    assert_eq!(roundtripped.info.id, "openai/gpt-4o");
    assert_eq!(roundtripped.info.provider, "openrouter");
    assert_eq!(roundtripped.info.context_window, Some(128_000));
    assert_eq!(roundtripped.max_output_tokens, Some(16384));
    assert_eq!(roundtripped.parameters.len(), 4);

    // check temperature mutable range survived
    match &roundtripped.parameters[&ParameterName::Temperature] {
        ParameterAvailability::Mutable { range } => {
            assert_eq!(range.min, Some(0.0));
            assert_eq!(range.max, Some(2.0));
            assert_eq!(range.default, Some(1.0));
        }
        other => panic!("expected Mutable, got {other:?}"),
    }

    // check unsupported
    assert!(matches!(
        roundtripped.parameters[&ParameterName::TopK],
        ParameterAvailability::Unsupported
    ));

    // check opaque
    assert!(matches!(
        roundtripped.parameters[&ParameterName::Seed],
        ParameterAvailability::Opaque
    ));

    // check read-only custom param
    match &roundtripped.parameters[&ParameterName::Custom("logit_bias".into())] {
        ParameterAvailability::ReadOnly { value } => {
            assert_eq!(*value, json!(0.5));
        }
        other => panic!("expected ReadOnly, got {other:?}"),
    }

    // check pricing
    let pricing = roundtripped.pricing.unwrap();
    assert_eq!(pricing.prompt_cost_per_mtok, Some(2.50));
    assert_eq!(pricing.completion_cost_per_mtok, Some(10.0));
}
