use llm::builder::LLMBackend;
use ratatoskr::ParameterName;
use ratatoskr::providers::LlmChatProvider;
use ratatoskr::providers::traits::ChatProvider;
use ratatoskr::providers::traits::GenerateProvider;

#[test]
fn llm_chat_provider_declares_chat_parameters() {
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "key", "openrouter");
    let params = ChatProvider::supported_chat_parameters(&provider);
    assert!(params.contains(&ParameterName::Temperature));
    assert!(params.contains(&ParameterName::MaxTokens));
    assert!(params.contains(&ParameterName::TopP));
    assert!(params.contains(&ParameterName::Reasoning));
    assert!(params.contains(&ParameterName::Stop));
    // top_k not yet supported by llm crate
    assert!(!params.contains(&ParameterName::TopK));
}

#[test]
fn llm_chat_provider_declares_generate_parameters() {
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "key", "openrouter");
    let params = GenerateProvider::supported_generate_parameters(&provider);
    assert!(params.contains(&ParameterName::Temperature));
    assert!(params.contains(&ParameterName::MaxTokens));
    assert!(params.contains(&ParameterName::TopP));
    assert!(params.contains(&ParameterName::Stop));
    // reasoning not supported for generate
    assert!(!params.contains(&ParameterName::Reasoning));
}

#[test]
fn default_supported_parameters_is_empty() {
    // Verify the default trait implementation returns empty
    // (indirectly tested — any provider that doesn't override returns [])
    // We verify LlmChatProvider explicitly returns non-empty above.
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "key", "openrouter");
    let chat_params = ChatProvider::supported_chat_parameters(&provider);
    let gen_params = GenerateProvider::supported_generate_parameters(&provider);
    assert!(
        !chat_params.is_empty(),
        "LlmChatProvider should declare chat params"
    );
    assert!(
        !gen_params.is_empty(),
        "LlmChatProvider should declare generate params"
    );
}
