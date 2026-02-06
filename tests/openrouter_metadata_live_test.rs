//! Live tests for OpenRouter metadata fetch.
//!
//! Requires `OPENROUTER_API_KEY` environment variable.
//! Run with: `cargo test --test openrouter_metadata_live_test -- --ignored`

use ratatoskr::{ModelGateway, ParameterName, Ratatoskr};

fn build_gateway() -> impl ModelGateway {
    let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");
    Ratatoskr::builder()
        .openrouter(key)
        .build()
        .expect("failed to build gateway")
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn fetch_known_model() {
    let gateway = build_gateway();
    let metadata = gateway
        .fetch_model_metadata("anthropic/claude-sonnet-4")
        .await
        .expect("fetch should succeed for known model");

    assert_eq!(metadata.info.id, "anthropic/claude-sonnet-4");
    assert_eq!(metadata.info.provider, "openrouter");
    assert!(metadata.info.context_window.is_some());
    assert!(metadata.max_output_tokens.is_some());
    assert!(metadata.pricing.is_some());
    assert!(
        metadata
            .parameters
            .contains_key(&ParameterName::Temperature)
    );

    println!("context window: {:?}", metadata.info.context_window);
    println!("max output: {:?}", metadata.max_output_tokens);
    println!("pricing: {:?}", metadata.pricing);
    println!(
        "parameters: {:?}",
        metadata.parameters.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn fetch_unknown_model() {
    let gateway = build_gateway();
    let result = gateway
        .fetch_model_metadata("totally-fake/nonexistent-model-xyz")
        .await;

    assert!(result.is_err(), "should fail for unknown model");
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn fetch_populates_cache() {
    let gateway = build_gateway();
    let model = "anthropic/claude-sonnet-4";

    // Before fetch, model might be in registry (seed) but not in cache.
    // After fetch, model_metadata should definitely return data.
    let _fetched = gateway
        .fetch_model_metadata(model)
        .await
        .expect("fetch should succeed");

    let cached = gateway
        .model_metadata(model)
        .expect("should find in registry or cache after fetch");

    assert_eq!(cached.info.id, model);
}
