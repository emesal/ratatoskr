use ratatoskr::{Capabilities, ModelGateway, RatatoskrError};

// Test that the trait can be implemented
struct MockGateway;

#[async_trait::async_trait]
impl ModelGateway for MockGateway {
    async fn chat_stream(
        &self,
        _messages: &[ratatoskr::Message],
        _tools: Option<&[ratatoskr::ToolDefinition]>,
        _options: &ratatoskr::ChatOptions,
    ) -> ratatoskr::Result<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = ratatoskr::Result<ratatoskr::ChatEvent>> + Send>,
        >,
    > {
        Err(RatatoskrError::NotImplemented("chat_stream"))
    }

    async fn chat(
        &self,
        _messages: &[ratatoskr::Message],
        _tools: Option<&[ratatoskr::ToolDefinition]>,
        _options: &ratatoskr::ChatOptions,
    ) -> ratatoskr::Result<ratatoskr::ChatResponse> {
        Err(RatatoskrError::NotImplemented("chat"))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
}

#[test]
fn test_mock_gateway_capabilities() {
    let gateway = MockGateway;
    let caps = gateway.capabilities();
    assert!(!caps.chat);
}

#[tokio::test]
async fn test_default_embed_not_implemented() {
    let gateway = MockGateway;
    let result = gateway.embed("test", "model").await;
    assert!(matches!(result, Err(RatatoskrError::NotImplemented(_))));
}

#[tokio::test]
async fn test_default_infer_nli_batch_fallback() {
    let gateway = MockGateway;
    let pairs = [("premise1", "hypothesis1"), ("premise2", "hypothesis2")];
    let result = gateway.infer_nli_batch(&pairs, "model").await;
    // Should fail because the single infer_nli is not implemented
    assert!(result.is_err());
}

#[tokio::test]
async fn test_default_generate_not_implemented() {
    let gateway = MockGateway;
    let options = ratatoskr::GenerateOptions::new("model");
    let result = gateway.generate("test prompt", &options).await;
    assert!(matches!(result, Err(RatatoskrError::NotImplemented(_))));
}

#[tokio::test]
async fn test_default_generate_stream_not_implemented() {
    let gateway = MockGateway;
    let options = ratatoskr::GenerateOptions::new("model");
    let result = gateway.generate_stream("test prompt", &options).await;
    assert!(matches!(result, Err(RatatoskrError::NotImplemented(_))));
}
