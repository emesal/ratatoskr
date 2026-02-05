//! Tests for text generation types and methods.
//!
//! Live tests require API keys:
//! ```bash
//! OPENROUTER_API_KEY=sk-or-xxx cargo test --test generate_test -- --ignored
//! ```

use ratatoskr::{
    FinishReason, GenerateEvent, GenerateOptions, GenerateResponse, ReasoningConfig,
    ReasoningEffort,
};

// ============================================================================
// GenerateOptions Tests
// ============================================================================

#[test]
fn test_generate_options_new() {
    let opts = GenerateOptions::new("test-model");
    assert_eq!(opts.model, "test-model");
    assert!(opts.max_tokens.is_none());
    assert!(opts.temperature.is_none());
    assert!(opts.top_p.is_none());
    assert!(opts.stop_sequences.is_empty());
}

#[test]
fn test_generate_options_builder() {
    let opts = GenerateOptions::new("test-model")
        .max_tokens(100)
        .temperature(0.7)
        .top_p(0.9)
        .stop_sequence("END");

    assert_eq!(opts.model, "test-model");
    assert_eq!(opts.max_tokens, Some(100));
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.top_p, Some(0.9));
    assert_eq!(opts.stop_sequences, vec!["END".to_string()]);
}

#[test]
fn test_generate_options_multiple_stop_sequences() {
    let opts = GenerateOptions::new("model")
        .stop_sequence("STOP1")
        .stop_sequence("STOP2")
        .stop_sequence("STOP3");

    assert_eq!(opts.stop_sequences.len(), 3);
}

#[test]
fn test_generate_options_stop_sequences_vec() {
    let opts = GenerateOptions::new("model")
        .stop_sequence("A")
        .stop_sequence("B")
        .stop_sequence("C");

    assert_eq!(opts.stop_sequences, vec!["A", "B", "C"]);
}

#[test]
fn test_generate_options_serialization() {
    let opts = GenerateOptions::new("model")
        .max_tokens(50)
        .temperature(0.5);

    let json = serde_json::to_string(&opts).unwrap();
    assert!(json.contains("\"model\":\"model\""));
    assert!(json.contains("\"max_tokens\":50"));
    assert!(json.contains("\"temperature\":0.5"));
}

#[test]
fn test_generate_options_new_fields() {
    let opts = GenerateOptions::new("llama3")
        .top_k(40)
        .frequency_penalty(0.5)
        .presence_penalty(0.3)
        .seed(42)
        .reasoning(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude_from_output: None,
        });

    assert_eq!(opts.top_k, Some(40));
    assert_eq!(opts.frequency_penalty, Some(0.5));
    assert_eq!(opts.presence_penalty, Some(0.3));
    assert_eq!(opts.seed, Some(42));
    assert!(opts.reasoning.is_some());
}

#[test]
fn test_generate_options_serde_with_new_fields() {
    let opts = GenerateOptions::new("test")
        .top_k(50)
        .frequency_penalty(0.1)
        .seed(123);

    let json = serde_json::to_string(&opts).unwrap();
    let parsed: GenerateOptions = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.top_k, Some(50));
    assert_eq!(parsed.frequency_penalty, Some(0.1));
    assert_eq!(parsed.seed, Some(123));
}

#[test]
fn test_generate_options_backwards_compatible() {
    // existing fields still work
    let opts = GenerateOptions::new("model")
        .max_tokens(100)
        .temperature(0.7)
        .top_p(0.9)
        .stop_sequence("END");

    assert_eq!(opts.max_tokens, Some(100));
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.top_p, Some(0.9));
    assert_eq!(opts.stop_sequences, vec!["END".to_string()]);

    // new fields default to None
    assert!(opts.top_k.is_none());
    assert!(opts.frequency_penalty.is_none());
    assert!(opts.presence_penalty.is_none());
    assert!(opts.seed.is_none());
    assert!(opts.reasoning.is_none());
}

// ============================================================================
// GenerateResponse Tests
// ============================================================================

#[test]
fn test_generate_response_fields() {
    let response = GenerateResponse {
        text: "Hello, world!".to_string(),
        usage: None,
        model: Some("test-model".to_string()),
        finish_reason: FinishReason::Stop,
    };

    assert_eq!(response.text, "Hello, world!");
    assert!(response.usage.is_none());
    assert_eq!(response.model, Some("test-model".to_string()));
    assert!(matches!(response.finish_reason, FinishReason::Stop));
}

#[test]
fn test_generate_response_serialization() {
    let response = GenerateResponse {
        text: "Test output".to_string(),
        usage: None,
        model: Some("model".to_string()),
        finish_reason: FinishReason::Stop,
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"text\":\"Test output\""));
}

// ============================================================================
// GenerateEvent Tests
// ============================================================================

#[test]
fn test_generate_event_text() {
    let event = GenerateEvent::Text("chunk".to_string());
    match event {
        GenerateEvent::Text(t) => assert_eq!(t, "chunk"),
        _ => panic!("Expected Text variant"),
    }
}

#[test]
fn test_generate_event_done() {
    let event = GenerateEvent::Done;
    assert!(matches!(event, GenerateEvent::Done));
}

#[test]
fn test_generate_event_serialization() {
    let event = GenerateEvent::Text("hello".to_string());
    let json = serde_json::to_string(&event).unwrap();
    // Check tagged enum format
    assert!(json.contains("\"type\":\"text\""));
}

// ============================================================================
// Live Tests (ignored by default)
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_live_generate_ollama() {
    use ratatoskr::{ModelGateway, Ratatoskr};

    // Ollama actually implements complete(), unlike OpenRouter
    let gateway = Ratatoskr::builder()
        .ollama("http://localhost:11434")
        .build()
        .expect("Failed to build gateway");

    let response = gateway
        .generate(
            "What is 2 + 2? Answer with just the number.",
            &GenerateOptions::new("llama3.2:1b").max_tokens(10),
        )
        .await
        .expect("Generate failed");

    assert!(!response.text.is_empty());
    assert!(
        response.text.contains('4'),
        "Expected '4' in response: {}",
        response.text
    );
}

#[tokio::test]
#[ignore]
async fn test_live_generate_stream_openrouter() {
    use futures_util::StreamExt;
    use ratatoskr::{ModelGateway, Ratatoskr};

    let api_key =
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set for live tests");

    let gateway = Ratatoskr::builder()
        .openrouter(api_key)
        .build()
        .expect("Failed to build gateway");

    let mut stream = gateway
        .generate_stream(
            "Say only 'hello' and nothing else.",
            &GenerateOptions::new("anthropic/claude-3-haiku-20240307").max_tokens(10),
        )
        .await
        .expect("Generate stream failed");

    let mut full_response = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(GenerateEvent::Text(text)) => {
                full_response.push_str(&text);
            }
            Ok(GenerateEvent::Done) => break,
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    assert!(
        full_response.to_lowercase().contains("hello"),
        "Expected 'hello' in response: {}",
        full_response
    );
}
