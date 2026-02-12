//! Integration tests for [`ParameterDiscoveryCache`] — runtime parameter
//! rejection caching and its integration with builder and validation.

use std::sync::Arc;
use std::time::Duration;

use llm::builder::LLMBackend;
use ratatoskr::cache::{DiscoveryConfig, DiscoveryRecord, ParameterDiscoveryCache};
use ratatoskr::providers::LlmChatProvider;
use ratatoskr::providers::registry::ProviderRegistry;
use ratatoskr::{
    ChatOptions, Message, ParameterName, ParameterValidationPolicy, Ratatoskr, RatatoskrError,
};

// =============================================================================
// Builder integration
// =============================================================================

#[test]
fn builder_enables_discovery_by_default() {
    let gw = Ratatoskr::builder().openrouter("test-key").build();
    assert!(gw.is_ok());
}

#[test]
fn builder_accepts_custom_discovery_config() {
    let config = DiscoveryConfig::new()
        .max_entries(500)
        .ttl(Duration::from_secs(3600));

    let gw = Ratatoskr::builder()
        .openrouter("test-key")
        .discovery(config)
        .build();
    assert!(gw.is_ok());
}

#[test]
fn builder_disable_parameter_discovery() {
    let gw = Ratatoskr::builder()
        .openrouter("test-key")
        .disable_parameter_discovery()
        .build();
    assert!(gw.is_ok());
}

// =============================================================================
// Discovery cache consultation during validation
// =============================================================================

/// Create a registry with an OpenRouter provider, Error validation, and a
/// wired discovery cache.
fn registry_with_discovery() -> (ProviderRegistry, Arc<ParameterDiscoveryCache>) {
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "test-key", "openrouter");
    let cache = Arc::new(ParameterDiscoveryCache::new(&DiscoveryConfig::new()));

    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(provider));
    registry.set_validation_policy(ParameterValidationPolicy::Error);
    registry.set_discovery_cache(Arc::clone(&cache));

    (registry, cache)
}

#[tokio::test]
async fn discovery_cache_consulted_during_validation() {
    let (registry, cache) = registry_with_discovery();

    // Pre-seed a runtime rejection for `temperature` — simulating a provider
    // having previously rejected this parameter at runtime. Temperature IS
    // statically supported by LlmChatProvider, so without the discovery cache
    // entry it would pass validation. This tests that the discovery cache
    // overrides the static declaration.
    cache.record(DiscoveryRecord {
        parameter: ParameterName::Temperature,
        provider: "openrouter".to_string(),
        model: "test-model".to_string(),
        discovered_at: std::time::Instant::now(),
        reason: "rejected by provider".to_string(),
    });

    let opts = ChatOptions::new("test-model").temperature(0.5);
    let messages = vec![Message::user("hello")];

    let result = registry.chat(&messages, None, &opts).await;
    assert!(
        matches!(result, Err(RatatoskrError::UnsupportedParameter { .. })),
        "expected UnsupportedParameter from discovery cache, got: {result:?}"
    );
}

#[tokio::test]
async fn discovery_cache_not_consulted_when_disabled() {
    // Registry without discovery cache — temperature IS statically supported by
    // LlmChatProvider. Without a discovery cache, it should pass validation
    // (and fail later at the API level).
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "test-key", "openrouter");
    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(provider));
    registry.set_validation_policy(ParameterValidationPolicy::Error);
    // no set_discovery_cache — simulates disable_parameter_discovery()

    let opts = ChatOptions::new("test-model").temperature(0.5);
    let messages = vec![Message::user("hello")];

    let result = registry.chat(&messages, None, &opts).await;
    // Should NOT be UnsupportedParameter — it'll be some other error (API auth fail)
    if let Err(ref e) = result {
        assert!(
            !matches!(e, RatatoskrError::UnsupportedParameter { .. }),
            "temperature is statically supported, should not be rejected, got: {e:?}"
        );
    }
}

#[tokio::test]
async fn discovery_cache_scoped_to_model() {
    let (registry, cache) = registry_with_discovery();

    // Record a runtime rejection for temperature on model-a only.
    // Temperature IS statically supported, so this is purely a runtime discovery.
    cache.record(DiscoveryRecord {
        parameter: ParameterName::Temperature,
        provider: "openrouter".to_string(),
        model: "model-a".to_string(),
        discovered_at: std::time::Instant::now(),
        reason: "rejected by provider at runtime".to_string(),
    });

    // model-a with temperature → should fail (discovery cache rejects it)
    let opts_a = ChatOptions::new("model-a").temperature(0.5);
    let messages = vec![Message::user("hello")];
    let result_a = registry.chat(&messages, None, &opts_a).await;
    assert!(
        matches!(result_a, Err(RatatoskrError::UnsupportedParameter { .. })),
        "model-a should be rejected via discovery cache"
    );

    // model-b with temperature → discovery cache has no entry, should pass validation
    let opts_b = ChatOptions::new("model-b").temperature(0.5);
    let result_b = registry.chat(&messages, None, &opts_b).await;
    if let Err(ref e) = result_b {
        assert!(
            !matches!(e, RatatoskrError::UnsupportedParameter { .. }),
            "model-b should not be rejected (not in discovery cache), got: {e:?}"
        );
    }
}

// =============================================================================
// Runtime recording via direct cache API
// =============================================================================

#[test]
fn record_and_query_round_trip() {
    let cache = ParameterDiscoveryCache::new(&DiscoveryConfig::new());

    cache.record(DiscoveryRecord {
        parameter: ParameterName::TopK,
        provider: "openrouter".to_string(),
        model: "gpt-4".to_string(),
        discovered_at: std::time::Instant::now(),
        reason: "not supported by this model".to_string(),
    });

    assert!(cache.is_known_unsupported("openrouter", "gpt-4", &ParameterName::TopK));
    assert!(!cache.is_known_unsupported("openrouter", "gpt-4", &ParameterName::Temperature));
    assert!(!cache.is_known_unsupported("anthropic", "gpt-4", &ParameterName::TopK));
}

#[test]
fn batch_query_returns_matching_subset() {
    let cache = ParameterDiscoveryCache::new(&DiscoveryConfig::new());

    for param in [ParameterName::TopK, ParameterName::Seed] {
        cache.record(DiscoveryRecord {
            parameter: param,
            provider: "openrouter".to_string(),
            model: "gpt-4".to_string(),
            discovered_at: std::time::Instant::now(),
            reason: "rejected".to_string(),
        });
    }

    let query = vec![
        ParameterName::TopK,
        ParameterName::Temperature, // not recorded
        ParameterName::Seed,
    ];
    let unsupported = cache.known_unsupported_params("openrouter", "gpt-4", &query);
    assert_eq!(unsupported.len(), 2);
    assert!(unsupported.contains(&ParameterName::TopK));
    assert!(unsupported.contains(&ParameterName::Seed));
    assert!(!unsupported.contains(&ParameterName::Temperature));
}

#[test]
fn ttl_expiry() {
    let config = DiscoveryConfig::new()
        .max_entries(100)
        .ttl(Duration::from_millis(1));
    let cache = ParameterDiscoveryCache::new(&config);

    cache.record(DiscoveryRecord {
        parameter: ParameterName::TopK,
        provider: "openrouter".to_string(),
        model: "test-model".to_string(),
        discovered_at: std::time::Instant::now(),
        reason: "rejected".to_string(),
    });

    std::thread::sleep(Duration::from_millis(50));
    assert!(!cache.is_known_unsupported("openrouter", "test-model", &ParameterName::TopK));
}

#[test]
fn list_discoveries_returns_all_active() {
    let cache = ParameterDiscoveryCache::new(&DiscoveryConfig::new());

    for (provider, model, param) in [
        ("openrouter", "gpt-4", ParameterName::TopK),
        ("anthropic", "claude-3", ParameterName::Seed),
    ] {
        cache.record(DiscoveryRecord {
            parameter: param,
            provider: provider.to_string(),
            model: model.to_string(),
            discovered_at: std::time::Instant::now(),
            reason: "rejected".to_string(),
        });
    }

    assert_eq!(cache.list_discoveries().len(), 2);
}

#[test]
fn static_validation_still_works_with_discovery_cache() {
    // Sanity check: statically unsupported params (top_k for LlmChatProvider)
    // are still rejected even when discovery cache is empty.
    let provider = LlmChatProvider::new(LLMBackend::OpenRouter, "test-key", "openrouter");
    let cache = Arc::new(ParameterDiscoveryCache::new(&DiscoveryConfig::new()));

    let mut registry = ProviderRegistry::new();
    registry.add_chat(Arc::new(provider));
    registry.set_validation_policy(ParameterValidationPolicy::Error);
    registry.set_discovery_cache(cache);

    // top_k is statically unsupported — should fail even with empty discovery cache
    let opts = ChatOptions::new("test-model").top_k(40);
    let messages = vec![Message::user("hello")];

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = rt.block_on(registry.chat(&messages, None, &opts));
    assert!(matches!(
        result,
        Err(RatatoskrError::UnsupportedParameter { .. })
    ));
}
