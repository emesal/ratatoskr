//! Integration tests for gRPC service mode.
//!
//! Starts an in-process ratd server and connects with a [`ServiceClient`],
//! validating the full round-trip through proto conversions.
//!
//! Sync trait methods ([`list_models`], [`model_status`]) use `block_on`
//! internally, so they must run inside [`tokio::task::spawn_blocking`] to
//! avoid panicking within the tokio runtime.

#![cfg(all(feature = "server", feature = "client"))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use ratatoskr::client::ServiceClient;
use ratatoskr::server::RatatoskrService;
use ratatoskr::server::proto::ratatoskr_server::RatatoskrServer;
use ratatoskr::{ChatOptions, Message, ModelGateway, Ratatoskr};
use tokio::net::TcpListener;
use tonic::transport::Server;

/// Find an available port for testing.
async fn find_available_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap()
}

/// Start a test server on a random port and return the address string.
async fn start_test_server() -> String {
    let addr = find_available_port().await;
    let addr_str = format!("http://{addr}");

    // Minimal gateway with a fake API key — won't make real calls
    // but is enough for health, models, and capabilities tests.
    let gateway = Ratatoskr::builder()
        .openrouter("test-key")
        .build()
        .expect("failed to build test gateway");

    let service = RatatoskrService::new(Arc::new(gateway));
    let server = RatatoskrServer::new(service);

    tokio::spawn(async move {
        Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    });

    // Give the server a moment to bind.
    tokio::time::sleep(Duration::from_millis(100)).await;

    addr_str
}

#[tokio::test]
async fn test_client_connect() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await;
    assert!(client.is_ok(), "failed to connect: {:?}", client.err());
}

#[tokio::test]
async fn test_client_capabilities() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await.unwrap();

    let caps = client.capabilities();
    assert!(caps.chat, "ServiceClient should report chat capability");
    assert!(
        caps.chat_streaming,
        "ServiceClient should report streaming capability"
    );
}

#[tokio::test]
async fn test_client_list_models() {
    let addr = start_test_server().await;
    let client = Arc::new(ServiceClient::connect(&addr).await.unwrap());

    // list_models() uses block_on internally — must escape the async context
    let models = tokio::task::spawn_blocking(move || client.list_models())
        .await
        .unwrap();

    // Should have at least one model from the openrouter provider.
    assert!(!models.is_empty(), "expected at least one model");
}

#[tokio::test]
async fn test_client_model_status() {
    let addr = start_test_server().await;
    let client = Arc::new(ServiceClient::connect(&addr).await.unwrap());

    let status = tokio::task::spawn_blocking(move || client.model_status("nonexistent-model"))
        .await
        .unwrap();

    // Should return some status variant without panicking.
    match status {
        ratatoskr::ModelStatus::Ready
        | ratatoskr::ModelStatus::Available
        | ratatoskr::ModelStatus::Loading
        | ratatoskr::ModelStatus::Unavailable { .. } => {}
        _ => panic!("unexpected model status variant"),
    }
}

#[tokio::test]
async fn test_health_rpc() {
    use ratatoskr::server::proto::ratatoskr_client::RatatoskrClient;

    let addr = start_test_server().await;
    let mut grpc_client = RatatoskrClient::connect(addr).await.unwrap();

    let response = grpc_client
        .health(ratatoskr::server::proto::HealthRequest {})
        .await
        .unwrap()
        .into_inner();

    assert!(response.healthy, "server should report healthy");
    assert!(!response.version.is_empty(), "version should be present");
}

// =============================================================================
// FetchModelMetadata RPC
// =============================================================================

#[tokio::test]
async fn test_fetch_model_metadata_rpc() {
    let addr = start_test_server().await;
    let client = ServiceClient::connect(&addr).await.unwrap();

    // fetch_model_metadata makes a real HTTP call to OpenRouter (with fake key),
    // so it should fail — but the RPC roundtrip itself should work without panicking.
    let result = client.fetch_model_metadata("some-model").await;
    assert!(result.is_err(), "should fail with fake API key");
}

#[tokio::test]
async fn test_get_model_metadata_rpc() {
    let addr = start_test_server().await;
    let client = Arc::new(ServiceClient::connect(&addr).await.unwrap());

    // get_model_metadata (sync) should return seed data for known models
    let metadata =
        tokio::task::spawn_blocking(move || client.model_metadata("anthropic/claude-sonnet-4"))
            .await
            .unwrap();

    assert!(
        metadata.is_some(),
        "should find claude-sonnet-4 in registry via RPC"
    );
    let m = metadata.unwrap();
    assert_eq!(m.info.id, "anthropic/claude-sonnet-4");
}

// =============================================================================
// Live tests — require ratd running with valid API keys
// =============================================================================

#[tokio::test]
#[ignore = "requires ratd running with valid API keys"]
async fn test_live_chat() {
    let client = ServiceClient::connect("http://127.0.0.1:9741")
        .await
        .expect("connect to ratd (is it running?)");

    let response = client
        .chat(
            &[Message::user("Say hello in exactly 3 words.")],
            None,
            &ChatOptions::new("anthropic/claude-sonnet-4"),
        )
        .await
        .expect("chat request failed");

    assert!(!response.content.is_empty());
    println!("response: {}", response.content);
}
