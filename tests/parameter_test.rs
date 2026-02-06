use ratatoskr::{ParameterAvailability, ParameterName, ParameterRange};

#[test]
fn well_known_parameter_names() {
    assert_eq!(ParameterName::Temperature.as_str(), "temperature");
    assert_eq!(ParameterName::TopP.as_str(), "top_p");
    assert_eq!(ParameterName::TopK.as_str(), "top_k");
    assert_eq!(ParameterName::MaxTokens.as_str(), "max_tokens");
    assert_eq!(
        ParameterName::FrequencyPenalty.as_str(),
        "frequency_penalty"
    );
    assert_eq!(ParameterName::PresencePenalty.as_str(), "presence_penalty");
    assert_eq!(ParameterName::Seed.as_str(), "seed");
    assert_eq!(ParameterName::Stop.as_str(), "stop");
    assert_eq!(ParameterName::Reasoning.as_str(), "reasoning");
    assert_eq!(ParameterName::CachePrompt.as_str(), "cache_prompt");
    assert_eq!(ParameterName::ResponseFormat.as_str(), "response_format");
    assert_eq!(ParameterName::ToolChoice.as_str(), "tool_choice");
    assert_eq!(
        ParameterName::ParallelToolCalls.as_str(),
        "parallel_tool_calls"
    );
}

#[test]
fn custom_parameter_name() {
    let custom = ParameterName::Custom("logit_bias".into());
    assert_eq!(custom.as_str(), "logit_bias");
}

#[test]
fn parameter_name_from_str() {
    assert_eq!(
        "temperature".parse::<ParameterName>().unwrap(),
        ParameterName::Temperature
    );
    assert_eq!(
        "top_k".parse::<ParameterName>().unwrap(),
        ParameterName::TopK
    );
    assert_eq!(
        "logit_bias".parse::<ParameterName>().unwrap(),
        ParameterName::Custom("logit_bias".into())
    );
    assert_eq!(
        "parallel_tool_calls".parse::<ParameterName>().unwrap(),
        ParameterName::ParallelToolCalls
    );
}

#[test]
fn parameter_name_serde_roundtrip() {
    let names = vec![
        ParameterName::Temperature,
        ParameterName::ParallelToolCalls,
        ParameterName::Custom("logit_bias".into()),
    ];
    let json = serde_json::to_string(&names).unwrap();
    let parsed: Vec<ParameterName> = serde_json::from_str(&json).unwrap();
    assert_eq!(names, parsed);
}

#[test]
fn parameter_range_defaults() {
    let range = ParameterRange::default();
    assert!(range.min.is_none());
    assert!(range.max.is_none());
    assert!(range.default.is_none());
}

#[test]
fn parameter_range_builder() {
    let range = ParameterRange::new().min(0.0).max(2.0).default_value(1.0);
    assert_eq!(range.min, Some(0.0));
    assert_eq!(range.max, Some(2.0));
    assert_eq!(range.default, Some(1.0));
}

#[test]
fn parameter_availability_variants() {
    let mutable = ParameterAvailability::Mutable {
        range: ParameterRange::new().min(0.0).max(2.0),
    };
    let read_only = ParameterAvailability::ReadOnly {
        value: serde_json::json!(0.7),
    };
    let opaque = ParameterAvailability::Opaque;
    let unsupported = ParameterAvailability::Unsupported;

    assert!(matches!(mutable, ParameterAvailability::Mutable { .. }));
    assert!(matches!(read_only, ParameterAvailability::ReadOnly { .. }));
    assert!(matches!(opaque, ParameterAvailability::Opaque));
    assert!(matches!(unsupported, ParameterAvailability::Unsupported));
}

#[test]
fn parameter_availability_serde_roundtrip() {
    let avail = ParameterAvailability::Mutable {
        range: ParameterRange::new().min(0.0).max(2.0).default_value(1.0),
    };
    let json = serde_json::to_string(&avail).unwrap();
    let parsed: ParameterAvailability = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, ParameterAvailability::Mutable { .. }));
}

#[test]
fn parameter_availability_is_supported() {
    assert!(
        ParameterAvailability::Mutable {
            range: ParameterRange::default()
        }
        .is_supported()
    );
    assert!(
        ParameterAvailability::ReadOnly {
            value: serde_json::json!(1)
        }
        .is_supported()
    );
    assert!(ParameterAvailability::Opaque.is_supported());
    assert!(!ParameterAvailability::Unsupported.is_supported());
}
