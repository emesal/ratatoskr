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
