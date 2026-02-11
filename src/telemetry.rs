//! Telemetry metric name constants.
//!
//! Centralised metric names for ratatoskr operations. Consumers install
//! their own `metrics` recorder (e.g. prometheus, statsd); without a
//! recorder installed, all metric calls are no-ops.
//!
//! # Metric naming conventions
//!
//! All metrics are prefixed with `ratatoskr_`. Counters end in `_total`,
//! histograms use meaningful units (e.g. `_seconds`).
//!
//! # Common labels
//!
//! - `provider` — provider name (e.g. "openrouter", "ollama")
//! - `operation` — capability invoked (e.g. "chat", "embed", "infer_nli")
//! - `status` — outcome: "ok" or "error"
//! - `direction` — token direction: "prompt" or "completion"

/// Total requests dispatched through the registry.
///
/// Labels: `provider`, `operation`, `status` ("ok" | "error").
pub const REQUESTS_TOTAL: &str = "ratatoskr_requests_total";

/// Request duration in seconds.
///
/// Labels: `provider`, `operation`.
pub const REQUEST_DURATION_SECONDS: &str = "ratatoskr_request_duration_seconds";

/// Total retry attempts (not counting the initial request).
///
/// Labels: `provider`, `operation`.
pub const RETRIES_TOTAL: &str = "ratatoskr_retries_total";

/// Total tokens consumed.
///
/// Labels: `provider`, `direction` ("prompt" | "completion").
pub const TOKENS_TOTAL: &str = "ratatoskr_tokens_total";

/// Total cache hits (for response cache, task 9).
///
/// Labels: `operation`.
pub const CACHE_HITS_TOTAL: &str = "ratatoskr_cache_hits_total";

/// Total cache misses (for response cache, task 9).
///
/// Labels: `operation`.
pub const CACHE_MISSES_TOTAL: &str = "ratatoskr_cache_misses_total";
