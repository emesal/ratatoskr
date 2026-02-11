//! Tests for routing configuration, latency tracking, and cost-aware routing.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ratatoskr::providers::registry::ProviderRegistry;
use ratatoskr::providers::traits::{ChatProvider, EmbeddingProvider};
use ratatoskr::{
    ChatEvent, ChatOptions, ChatResponse, Embedding, FinishReason, Message, ModelCapability,
    ModelInfo, ModelMetadata, PricingInfo, ProviderCostInfo, ProviderLatency, RatatoskrError,
    Result, RoutingConfig, ToolDefinition,
};

// ============================================================================
// Mock providers
// ============================================================================

struct MockChat {
    name: &'static str,
}

#[async_trait]
impl ChatProvider for MockChat {
    fn name(&self) -> &str {
        self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse {
            content: format!("from {}", self.name),
            model: Some("test".into()),
            tool_calls: vec![],
            usage: None,
            reasoning: None,
            finish_reason: FinishReason::default(),
        })
    }

    async fn chat_stream(
        &self,
        _messages: &[Message],
        _tools: Option<&[ToolDefinition]>,
        _options: &ChatOptions,
    ) -> Result<std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<ChatEvent>> + Send>>> {
        Err(RatatoskrError::NotImplemented("stream"))
    }
}

struct MockEmbed {
    name: &'static str,
    dims: usize,
}

#[async_trait]
impl EmbeddingProvider for MockEmbed {
    fn name(&self) -> &str {
        self.name
    }

    async fn embed(&self, _text: &str, _model: &str) -> Result<Embedding> {
        Ok(Embedding {
            values: vec![0.1; self.dims],
            model: "test".into(),
            dimensions: self.dims,
        })
    }
}

// ============================================================================
// RoutingConfig tests
// ============================================================================

#[test]
fn routing_config_builder_api() {
    let config = RoutingConfig::new()
        .chat("anthropic")
        .generate("openrouter")
        .embed("local")
        .nli("huggingface")
        .classify("local");

    assert_eq!(config.chat.as_deref(), Some("anthropic"));
    assert_eq!(config.generate.as_deref(), Some("openrouter"));
    assert_eq!(config.embed.as_deref(), Some("local"));
    assert_eq!(config.nli.as_deref(), Some("huggingface"));
    assert_eq!(config.classify.as_deref(), Some("local"));
}

#[test]
fn routing_config_defaults_are_none() {
    let config = RoutingConfig::default();
    assert!(config.chat.is_none());
    assert!(config.generate.is_none());
    assert!(config.embed.is_none());
    assert!(config.nli.is_none());
    assert!(config.classify.is_none());
}

// ============================================================================
// Preferred provider reordering via ProviderRegistry
// ============================================================================

#[tokio::test]
async fn preferred_chat_provider_tried_first() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "openrouter" }));
    registry.add_chat(Arc::new(MockChat { name: "anthropic" }));
    registry.add_chat(Arc::new(MockChat { name: "ollama" }));

    // Before routing: openrouter is first
    let names = registry.provider_names();
    assert_eq!(names.chat, vec!["openrouter", "anthropic", "ollama"]);

    // Apply routing: prefer anthropic
    registry.apply_routing(&RoutingConfig::new().chat("anthropic"));

    let names = registry.provider_names();
    assert_eq!(names.chat, vec!["anthropic", "openrouter", "ollama"]);

    // Verify it actually dispatches to anthropic first
    let result = registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await
        .unwrap();
    assert_eq!(result.content, "from anthropic");
}

#[tokio::test]
async fn preferred_embed_provider_tried_first() {
    let mut registry = ProviderRegistry::new();
    registry.add_embedding(Arc::new(MockEmbed {
        name: "huggingface",
        dims: 384,
    }));
    registry.add_embedding(Arc::new(MockEmbed {
        name: "local",
        dims: 768,
    }));

    registry.apply_routing(&RoutingConfig::new().embed("local"));

    let names = registry.provider_names();
    assert_eq!(names.embedding, vec!["local", "huggingface"]);

    let result = registry.embed("hello", "any-model").await.unwrap();
    assert_eq!(result.dimensions, 768); // local's dimensions
}

#[test]
fn routing_unknown_provider_is_noop() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "openrouter" }));
    registry.add_chat(Arc::new(MockChat { name: "anthropic" }));

    // "nonexistent" doesn't match any provider — order unchanged
    registry.apply_routing(&RoutingConfig::new().chat("nonexistent"));

    let names = registry.provider_names();
    assert_eq!(names.chat, vec!["openrouter", "anthropic"]);
}

#[test]
fn routing_already_first_is_noop() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "openrouter" }));
    registry.add_chat(Arc::new(MockChat { name: "anthropic" }));

    registry.apply_routing(&RoutingConfig::new().chat("openrouter"));

    let names = registry.provider_names();
    assert_eq!(names.chat, vec!["openrouter", "anthropic"]);
}

#[test]
fn routing_multiple_capabilities() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "openrouter" }));
    registry.add_chat(Arc::new(MockChat { name: "anthropic" }));
    registry.add_embedding(Arc::new(MockEmbed {
        name: "huggingface",
        dims: 384,
    }));
    registry.add_embedding(Arc::new(MockEmbed {
        name: "local",
        dims: 768,
    }));

    registry.apply_routing(&RoutingConfig::new().chat("anthropic").embed("local"));

    let names = registry.provider_names();
    assert_eq!(names.chat, vec!["anthropic", "openrouter"]);
    assert_eq!(names.embedding, vec!["local", "huggingface"]);
}

// ============================================================================
// Latency tracking (EWMA)
// ============================================================================

#[test]
fn latency_tracker_basic() {
    let tracker = ProviderLatency::with_default_alpha();
    assert!(tracker.average().is_none());

    tracker.record(Duration::from_millis(100));
    assert_eq!(tracker.average().unwrap().as_millis(), 100);
    assert_eq!(tracker.observation_count(), 1);

    tracker.record(Duration::from_millis(200));
    assert_eq!(tracker.observation_count(), 2);
    // EWMA with alpha=0.2: 0.2*200 + 0.8*100 = 120
    let avg = tracker.average().unwrap().as_millis();
    assert_eq!(avg, 120);
}

#[tokio::test]
async fn registry_tracks_latency_on_dispatch() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat {
        name: "test-provider",
    }));

    // Dispatch a request
    registry
        .chat(&[], None, &ChatOptions::default().model("test"))
        .await
        .unwrap();

    // Latency should have been recorded
    let latency = registry.provider_latency("test-provider");
    assert!(latency.is_some());
    assert_eq!(latency.unwrap().observation_count(), 1);
    assert!(latency.unwrap().average().is_some());
}

#[tokio::test]
async fn registry_tracks_latency_for_embed() {
    let mut registry = ProviderRegistry::new();
    registry.add_embedding(Arc::new(MockEmbed {
        name: "embed-provider",
        dims: 128,
    }));

    registry.embed("hello", "model").await.unwrap();

    let latency = registry.provider_latency("embed-provider");
    assert!(latency.is_some());
    assert_eq!(latency.unwrap().observation_count(), 1);
}

// ============================================================================
// Cost-aware routing
// ============================================================================

#[test]
fn providers_by_cost_sorted_cheapest_first() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "expensive" }));
    registry.add_chat(Arc::new(MockChat { name: "cheap" }));

    // Simulate metadata with pricing — all providers see the same model metadata
    let metadata = ModelMetadata {
        info: ModelInfo {
            id: "test-model".into(),
            provider: "test".into(),
            context_window: Some(4096),
            capabilities: vec![ModelCapability::Chat],
            dimensions: None,
        },
        parameters: Default::default(),
        pricing: Some(PricingInfo {
            prompt_cost_per_mtok: Some(3.0),
            completion_cost_per_mtok: Some(15.0),
        }),
        max_output_tokens: None,
    };

    let sorted = registry.providers_by_cost(Some(&metadata));
    assert_eq!(sorted.len(), 2);
    // Both have same pricing from the same metadata, so order is preserved
    assert_eq!(sorted[0].provider, "expensive");
    assert_eq!(sorted[1].provider, "cheap");
    assert_eq!(sorted[0].prompt_cost_per_mtok, Some(3.0));
}

#[test]
fn providers_by_cost_unknown_pricing_sorts_last() {
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(MockChat { name: "priced" }));
    registry.add_chat(Arc::new(MockChat { name: "unpriced" }));

    // No metadata → all costs unknown
    let sorted = registry.providers_by_cost(None);
    assert_eq!(sorted.len(), 2);
    // All infinity, order preserved
    assert!(sorted[0].combined_cost().is_infinite());
}

#[test]
fn providers_by_cost_empty_registry() {
    let registry = ProviderRegistry::new();
    let sorted = registry.providers_by_cost(None);
    assert!(sorted.is_empty());
}

#[test]
fn provider_cost_info_combined_cost() {
    let info = ProviderCostInfo {
        provider: "test".into(),
        prompt_cost_per_mtok: Some(3.0),
        completion_cost_per_mtok: Some(15.0),
    };
    assert_eq!(info.combined_cost(), 18.0);

    let unknown = ProviderCostInfo {
        provider: "test".into(),
        prompt_cost_per_mtok: None,
        completion_cost_per_mtok: None,
    };
    assert!(unknown.combined_cost().is_infinite());
}
