//! Routing configuration, latency tracking, and cost-aware provider selection.
//!
//! This module provides:
//! - [`RoutingConfig`] — preferred provider per capability (reorders fallback chain)
//! - [`ProviderLatency`] — EWMA-based per-provider latency tracking
//! - [`ProviderCostInfo`] — cost information for provider ranking
//!
//! # Preferred provider routing
//!
//! When a [`RoutingConfig`] is set on the builder, the named provider is moved
//! to position 0 in the fallback chain for that capability. Other providers
//! remain in their original registration order as fallbacks.
//!
//! ```rust,ignore
//! Ratatoskr::builder()
//!     .openrouter(key)
//!     .anthropic(anthropic_key)
//!     .routing(RoutingConfig::new().chat("anthropic"))
//!     .build()?
//! ```
//!
//! # Latency tracking
//!
//! [`ProviderLatency`] tracks a per-provider exponentially weighted moving
//! average (EWMA) of request durations. This is the foundation for future
//! latency-based routing — currently exposed for observability.
//!
//! # Cost-aware routing
//!
//! [`ProviderCostInfo`] pairs a provider name with pricing data from
//! [`ModelMetadata`](crate::ModelMetadata). The registry's `providers_by_cost()`
//! method returns providers sorted cheapest-first for a given model.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Deserialize;

/// Preferred provider per capability.
///
/// Each field names a provider (e.g. `"openrouter"`, `"anthropic"`, `"ollama"`)
/// that should be tried first for that capability. Unset fields leave the
/// default registration order unchanged.
///
/// Used by both the builder (programmatic) and ratd config (TOML):
///
/// ```toml
/// [routing]
/// chat = "anthropic"
/// embed = "local"
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RoutingConfig {
    /// Preferred chat provider name.
    #[serde(default)]
    pub chat: Option<String>,
    /// Preferred text generation provider name.
    #[serde(default)]
    pub generate: Option<String>,
    /// Preferred embedding provider name.
    #[serde(default)]
    pub embed: Option<String>,
    /// Preferred NLI provider name.
    #[serde(default)]
    pub nli: Option<String>,
    /// Preferred classification provider name.
    #[serde(default)]
    pub classify: Option<String>,
    /// Preferred stance detection provider name.
    #[serde(default)]
    pub stance: Option<String>,
}

impl RoutingConfig {
    /// Create an empty routing config (no preferences).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the preferred chat provider.
    pub fn chat(mut self, provider: impl Into<String>) -> Self {
        self.chat = Some(provider.into());
        self
    }

    /// Set the preferred text generation provider.
    pub fn generate(mut self, provider: impl Into<String>) -> Self {
        self.generate = Some(provider.into());
        self
    }

    /// Set the preferred embedding provider.
    pub fn embed(mut self, provider: impl Into<String>) -> Self {
        self.embed = Some(provider.into());
        self
    }

    /// Set the preferred NLI provider.
    pub fn nli(mut self, provider: impl Into<String>) -> Self {
        self.nli = Some(provider.into());
        self
    }

    /// Set the preferred classification provider.
    pub fn classify(mut self, provider: impl Into<String>) -> Self {
        self.classify = Some(provider.into());
        self
    }

    /// Set the preferred stance detection provider.
    pub fn stance(mut self, provider: impl Into<String>) -> Self {
        self.stance = Some(provider.into());
        self
    }
}

/// Reorder a provider vec so the named provider is at index 0.
///
/// If no provider matches `preferred`, the vec is left unchanged.
/// This is the core mechanism for preferred-provider routing.
pub(crate) fn promote_preferred<T: HasName>(providers: &mut [T], preferred: &str) {
    if let Some(idx) = providers.iter().position(|p| p.name() == preferred)
        && idx > 0
    {
        // Rotate the preferred provider to position 0, preserving
        // relative order of the others.
        providers[..=idx].rotate_right(1);
    }
}

/// Trait for types that have a provider name. Implemented for Arc<dyn Provider>
/// wrappers so `promote_preferred` can work generically.
pub(crate) trait HasName {
    fn name(&self) -> &str;
}

// ============================================================================
// EWMA latency tracking
// ============================================================================

/// Per-provider latency tracker using exponential weighted moving average.
///
/// Thread-safe via atomics — no locks needed. The EWMA smoothing factor
/// `alpha` controls how quickly the average responds to new observations:
/// - Higher alpha (e.g. 0.3) = more responsive, noisier
/// - Lower alpha (e.g. 0.1) = smoother, slower to adapt
///
/// Default alpha: 0.2 (balances responsiveness and stability).
pub struct ProviderLatency {
    /// EWMA of request duration in microseconds, stored as AtomicU64
    /// (f64 bits reinterpreted). Initialized to 0 = no data yet.
    ewma_micros: AtomicU64,
    /// Smoothing factor (0.0–1.0).
    alpha: f64,
    /// Total number of observations recorded.
    count: AtomicU64,
}

impl ProviderLatency {
    /// Create a new latency tracker with the given EWMA smoothing factor.
    pub fn new(alpha: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0.0, 1.0]");
        Self {
            ewma_micros: AtomicU64::new(0_f64.to_bits()),
            alpha,
            count: AtomicU64::new(0),
        }
    }

    /// Create a latency tracker with the default smoothing factor (0.2).
    pub fn with_default_alpha() -> Self {
        Self::new(0.2)
    }

    /// Record a request duration observation.
    ///
    /// # Concurrency note
    ///
    /// There is a benign race on the first observation: two threads can both
    /// see `count == 0` and each initialise the EWMA independently. Since this
    /// is observability-only data (used for latency-aware routing, not
    /// correctness), the impact is bounded to a slightly inaccurate initial
    /// EWMA value that converges after a few more observations.
    pub fn record(&self, duration: Duration) {
        let micros = duration.as_micros() as f64;
        loop {
            let current_bits = self.ewma_micros.load(Ordering::Relaxed);
            let current = f64::from_bits(current_bits);
            let new = if self.count.load(Ordering::Relaxed) == 0 {
                // First observation — initialise directly
                micros
            } else {
                // EWMA: new = alpha * observation + (1 - alpha) * old
                self.alpha * micros + (1.0 - self.alpha) * current
            };
            if self
                .ewma_micros
                .compare_exchange_weak(
                    current_bits,
                    new.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Current EWMA latency estimate, or `None` if no observations yet.
    pub fn average(&self) -> Option<Duration> {
        if self.count.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let micros = f64::from_bits(self.ewma_micros.load(Ordering::Relaxed));
        Some(Duration::from_micros(micros as u64))
    }

    /// Total number of observations recorded.
    pub fn observation_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for ProviderLatency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderLatency")
            .field("average", &self.average())
            .field("count", &self.observation_count())
            .field("alpha", &self.alpha)
            .finish()
    }
}

// ============================================================================
// Cost-aware routing
// ============================================================================

/// Cost information for a provider serving a specific model.
///
/// Returned by `ProviderRegistry::providers_by_cost()`, sorted cheapest-first
/// by combined prompt + completion cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderCostInfo {
    /// Provider name.
    pub provider: String,
    /// Cost per million prompt tokens (USD), if known.
    pub prompt_cost_per_mtok: Option<f64>,
    /// Cost per million completion tokens (USD), if known.
    pub completion_cost_per_mtok: Option<f64>,
}

impl ProviderCostInfo {
    /// Combined cost per million tokens (prompt + completion), for sorting.
    ///
    /// Returns `f64::INFINITY` if either cost is unknown, so providers
    /// with unknown pricing sort last.
    pub fn combined_cost(&self) -> f64 {
        match (self.prompt_cost_per_mtok, self.completion_cost_per_mtok) {
            (Some(p), Some(c)) => p + c,
            _ => f64::INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RoutingConfig tests
    // ========================================================================

    #[test]
    fn routing_config_defaults_are_none() {
        let config = RoutingConfig::new();
        assert!(config.chat.is_none());
        assert!(config.generate.is_none());
        assert!(config.embed.is_none());
        assert!(config.nli.is_none());
        assert!(config.classify.is_none());
    }

    #[test]
    fn routing_config_builder_methods() {
        let config = RoutingConfig::new()
            .chat("anthropic")
            .embed("local")
            .generate("openrouter");
        assert_eq!(config.chat.as_deref(), Some("anthropic"));
        assert_eq!(config.embed.as_deref(), Some("local"));
        assert_eq!(config.generate.as_deref(), Some("openrouter"));
        assert!(config.nli.is_none());
    }

    // ========================================================================
    // promote_preferred tests
    // ========================================================================

    /// Simple wrapper for testing promote_preferred.
    struct Named(&'static str);
    impl HasName for Named {
        fn name(&self) -> &str {
            self.0
        }
    }

    #[test]
    fn promote_moves_to_front() {
        let mut providers = [Named("a"), Named("b"), Named("c")];
        promote_preferred(&mut providers, "c");
        let names: Vec<_> = providers.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["c", "a", "b"]);
    }

    #[test]
    fn promote_preserves_order_of_others() {
        let mut providers = [Named("a"), Named("b"), Named("c"), Named("d")];
        promote_preferred(&mut providers, "c");
        let names: Vec<_> = providers.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["c", "a", "b", "d"]);
    }

    #[test]
    fn promote_noop_if_already_first() {
        let mut providers = [Named("a"), Named("b"), Named("c")];
        promote_preferred(&mut providers, "a");
        let names: Vec<_> = providers.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn promote_noop_if_not_found() {
        let mut providers = [Named("a"), Named("b")];
        promote_preferred(&mut providers, "nonexistent");
        let names: Vec<_> = providers.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["a", "b"]);
    }

    // ========================================================================
    // ProviderLatency (EWMA) tests
    // ========================================================================

    #[test]
    fn latency_no_observations_returns_none() {
        let tracker = ProviderLatency::with_default_alpha();
        assert!(tracker.average().is_none());
        assert_eq!(tracker.observation_count(), 0);
    }

    #[test]
    fn latency_first_observation_is_exact() {
        let tracker = ProviderLatency::with_default_alpha();
        tracker.record(Duration::from_millis(100));
        let avg = tracker.average().unwrap();
        assert_eq!(avg, Duration::from_millis(100));
        assert_eq!(tracker.observation_count(), 1);
    }

    #[test]
    fn latency_ewma_converges_toward_new_values() {
        let tracker = ProviderLatency::new(0.5); // high alpha for clear effect
        tracker.record(Duration::from_millis(100));
        tracker.record(Duration::from_millis(200));

        // EWMA: 0.5 * 200 + 0.5 * 100 = 150
        let avg = tracker.average().unwrap();
        assert_eq!(avg.as_millis(), 150);
    }

    #[test]
    fn latency_ewma_with_multiple_observations() {
        let tracker = ProviderLatency::new(0.2);
        // Simulate a series of observations
        tracker.record(Duration::from_millis(100));
        tracker.record(Duration::from_millis(100));
        tracker.record(Duration::from_millis(100));
        // After several identical observations, EWMA should be close to 100ms
        let avg = tracker.average().unwrap();
        assert!((avg.as_millis() as i64 - 100).abs() < 2);
    }

    #[test]
    fn latency_ewma_responds_to_spike() {
        let tracker = ProviderLatency::new(0.3);
        // Establish baseline
        for _ in 0..10 {
            tracker.record(Duration::from_millis(100));
        }
        // Spike
        tracker.record(Duration::from_millis(1000));
        let avg = tracker.average().unwrap();
        // Should be > 100ms (responding to spike) but < 1000ms (smoothed)
        assert!(avg.as_millis() > 100);
        assert!(avg.as_millis() < 1000);
    }

    // ========================================================================
    // ProviderCostInfo tests
    // ========================================================================

    #[test]
    fn cost_info_combined_cost() {
        let info = ProviderCostInfo {
            provider: "test".into(),
            prompt_cost_per_mtok: Some(3.0),
            completion_cost_per_mtok: Some(15.0),
        };
        assert_eq!(info.combined_cost(), 18.0);
    }

    #[test]
    fn cost_info_unknown_costs_sort_last() {
        let known = ProviderCostInfo {
            provider: "cheap".into(),
            prompt_cost_per_mtok: Some(1.0),
            completion_cost_per_mtok: Some(2.0),
        };
        let unknown = ProviderCostInfo {
            provider: "unknown".into(),
            prompt_cost_per_mtok: None,
            completion_cost_per_mtok: None,
        };
        assert!(known.combined_cost() < unknown.combined_cost());
    }

    #[test]
    fn cost_info_partial_unknown_sorts_last() {
        let partial = ProviderCostInfo {
            provider: "partial".into(),
            prompt_cost_per_mtok: Some(1.0),
            completion_cost_per_mtok: None,
        };
        assert!(partial.combined_cost().is_infinite());
    }
}
