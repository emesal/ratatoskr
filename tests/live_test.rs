//! Live integration tests - ignored by default, run with:
//! `OPENROUTER_API_KEY=sk-or-xxx cargo test --test integration -- --ignored`

use futures_util::StreamExt;
use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};

#[tokio::test]
#[ignore]
async fn test_live_chat_openrouter() {
    let api_key =
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set for live tests");

    let gateway = Ratatoskr::builder()
        .openrouter(Some(api_key))
        .build()
        .expect("Failed to build gateway");

    let response = gateway
        .chat(
            &[
                Message::system("You are a helpful assistant. Be brief."),
                Message::user("What is 2 + 2?"),
            ],
            None,
            &ChatOptions::new("ratatoskr:free/agentic"),
        )
        .await
        .expect("Chat failed");

    assert!(!response.content.is_empty());
    assert!(
        response.content.contains('4') || response.content.to_lowercase().contains("four"),
        "Expected answer to contain '4' or 'four', got: {}",
        response.content
    );
}

#[tokio::test]
#[ignore]
async fn test_live_streaming_openrouter() {
    let api_key =
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set for live tests");

    let gateway = Ratatoskr::builder()
        .openrouter(Some(api_key))
        .build()
        .expect("Failed to build gateway");

    let mut stream = gateway
        .chat_stream(
            &[Message::user("Say 'hello world' and nothing else.")],
            None,
            &ChatOptions::new("ratatoskr:free/agentic"),
        )
        .await
        .expect("Stream failed");

    let mut full_response = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(ratatoskr::ChatEvent::Content(text)) => {
                full_response.push_str(&text);
            }
            Ok(ratatoskr::ChatEvent::Done) => break,
            Ok(_) => {}
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    assert!(
        full_response.to_lowercase().contains("hello"),
        "Expected response to contain 'hello', got: {}",
        full_response
    );
}
