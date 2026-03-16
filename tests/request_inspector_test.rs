//! Integration tests: request inspector callback fires with serialized JSON.

use std::sync::{Arc, Mutex};

use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal OpenAI-compatible chat response for the stub backend.
fn chat_response_json() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    })
}

#[tokio::test]
async fn inspector_fires_on_chat() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_json()))
        .mount(&server)
        .await;

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    let inspector = Arc::new(move |body: &str| {
        captured_clone.lock().unwrap().push(body.to_string());
    });

    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .request_inspector(inspector)
        .build()
        .unwrap();

    let _response = gateway
        .chat(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await
        .unwrap();

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "inspector should fire exactly once");

    // Verify the captured JSON is valid and contains expected fields
    let parsed: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert_eq!(parsed["model"], "test-model");
    assert!(parsed["messages"].is_array());
}

#[tokio::test]
async fn inspector_not_set_no_overhead() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_json()))
        .mount(&server)
        .await;

    // No inspector set — should work fine
    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .build()
        .unwrap();

    let response = gateway
        .chat(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await;

    assert!(response.is_ok());
}

#[tokio::test]
async fn inspector_fires_on_chat_stream() {
    use futures_util::StreamExt;

    let server = MockServer::start().await;

    // SSE streaming response
    let sse_body = "\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    let inspector = Arc::new(move |body: &str| {
        captured_clone.lock().unwrap().push(body.to_string());
    });

    let gateway = Ratatoskr::builder()
        .stub(&server.uri())
        .request_inspector(inspector)
        .build()
        .unwrap();

    let mut stream = gateway
        .chat_stream(
            &[Message::user("hi")],
            None,
            &ChatOptions::new("test-model"),
        )
        .await
        .unwrap();

    // Drain the stream
    while let Some(_event) = stream.next().await {}

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "inspector should fire once for stream");
    let parsed: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert!(
        parsed["stream"].as_bool().unwrap_or(false),
        "stream should be true"
    );
}
