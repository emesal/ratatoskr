use std::time::Duration;

use ratatoskr::RetryConfig;

#[test]
fn retry_config_defaults() {
    let config = RetryConfig::default();
    assert_eq!(config.max_attempts, 3);
    assert_eq!(config.initial_delay, Duration::from_millis(500));
    assert_eq!(config.max_delay, Duration::from_secs(30));
    assert!(config.jitter);
}

#[test]
fn retry_config_builder() {
    let config = RetryConfig::new()
        .max_attempts(5)
        .initial_delay(Duration::from_millis(100))
        .max_delay(Duration::from_secs(10))
        .jitter(false);

    assert_eq!(config.max_attempts, 5);
    assert_eq!(config.initial_delay, Duration::from_millis(100));
    assert_eq!(config.max_delay, Duration::from_secs(10));
    assert!(!config.jitter);
}

#[test]
fn retry_config_disabled() {
    let config = RetryConfig::disabled();
    assert_eq!(config.max_attempts, 1);
}

#[test]
fn retry_config_delay_calculation() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_millis(100))
        .max_delay(Duration::from_secs(10))
        .jitter(false);

    // Exponential backoff: 100ms, 200ms, 400ms, 800ms, ...
    assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
    assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
    assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
    assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
}

#[test]
fn retry_config_delay_capped_at_max() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_secs(1))
        .max_delay(Duration::from_secs(5))
        .jitter(false);

    // attempt 3 = 1 * 2^3 = 8s, but capped at 5s
    assert_eq!(config.delay_for_attempt(3), Duration::from_secs(5));
}

#[test]
fn retry_config_respects_retry_after() {
    let config = RetryConfig::new()
        .initial_delay(Duration::from_millis(100))
        .jitter(false);

    // retry_after from provider overrides calculated delay
    let delay = config.effective_delay(0, Some(Duration::from_secs(5)));
    assert_eq!(delay, Duration::from_secs(5));

    // without retry_after, uses calculated delay
    let delay = config.effective_delay(0, None);
    assert_eq!(delay, Duration::from_millis(100));
}
