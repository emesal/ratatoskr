use std::time::Duration;

use ratatoskr::{RatatoskrError, Result};

#[test]
fn test_error_display() {
    let err = RatatoskrError::ModelNotFound("gpt-5".to_string());
    assert!(err.to_string().contains("gpt-5"));
}

#[test]
fn test_not_implemented() {
    let err = RatatoskrError::NotImplemented("embed");
    assert!(err.to_string().contains("not implemented"));
}

#[test]
fn test_result_alias() {
    fn returns_error() -> Result<()> {
        Err(RatatoskrError::NoProvider)
    }
    assert!(returns_error().is_err());
}

// ============================================================================
// Transient error classification
// ============================================================================

#[test]
fn transient_errors() {
    assert!(RatatoskrError::RateLimited { retry_after: None }.is_transient());
    assert!(
        RatatoskrError::RateLimited {
            retry_after: Some(Duration::from_secs(1))
        }
        .is_transient()
    );
    assert!(RatatoskrError::Http("connection reset".into()).is_transient());
    assert!(
        RatatoskrError::Api {
            status: 500,
            message: "internal".into()
        }
        .is_transient()
    );
    assert!(
        RatatoskrError::Api {
            status: 502,
            message: "bad gateway".into()
        }
        .is_transient()
    );
    assert!(
        RatatoskrError::Api {
            status: 503,
            message: "unavailable".into()
        }
        .is_transient()
    );
    assert!(
        RatatoskrError::Api {
            status: 504,
            message: "timeout".into()
        }
        .is_transient()
    );
    assert!(RatatoskrError::Stream("connection reset".into()).is_transient());
    assert!(RatatoskrError::EmptyResponse.is_transient());
}

#[test]
fn permanent_errors() {
    assert!(!RatatoskrError::AuthenticationFailed.is_transient());
    assert!(!RatatoskrError::ModelNotFound("x".into()).is_transient());
    assert!(!RatatoskrError::InvalidInput("x".into()).is_transient());
    assert!(!RatatoskrError::NoProvider.is_transient());
    assert!(!RatatoskrError::Configuration("x".into()).is_transient());
    assert!(!RatatoskrError::ContentFiltered { reason: "x".into() }.is_transient());
    assert!(!RatatoskrError::ContextLengthExceeded { limit: 4096 }.is_transient());
    assert!(
        !RatatoskrError::Api {
            status: 400,
            message: "bad request".into()
        }
        .is_transient()
    );
    assert!(
        !RatatoskrError::Api {
            status: 401,
            message: "unauth".into()
        }
        .is_transient()
    );
    assert!(
        !RatatoskrError::Api {
            status: 403,
            message: "forbidden".into()
        }
        .is_transient()
    );
    assert!(
        !RatatoskrError::Api {
            status: 404,
            message: "not found".into()
        }
        .is_transient()
    );
    assert!(
        !RatatoskrError::UnsupportedParameter {
            param: "x".into(),
            model: "y".into(),
            provider: "z".into(),
        }
        .is_transient()
    );
}

#[test]
fn llm_error_transient_heuristic() {
    // network-sounding errors are transient
    assert!(RatatoskrError::Llm("connection reset by peer".into()).is_transient());
    assert!(RatatoskrError::Llm("timeout".into()).is_transient());
    assert!(RatatoskrError::Llm("connection refused".into()).is_transient());
    // generic errors are not
    assert!(!RatatoskrError::Llm("invalid model format".into()).is_transient());
}

// ============================================================================
// retry_after extraction
// ============================================================================

#[test]
fn retry_after_from_rate_limited() {
    let duration = Duration::from_secs(5);
    let err = RatatoskrError::RateLimited {
        retry_after: Some(duration),
    };
    assert_eq!(err.retry_after(), Some(duration));
}

#[test]
fn retry_after_none_when_not_specified() {
    let err = RatatoskrError::RateLimited { retry_after: None };
    assert_eq!(err.retry_after(), None);
}

#[test]
fn retry_after_none_for_non_rate_limit_errors() {
    assert_eq!(RatatoskrError::Http("timeout".into()).retry_after(), None);
    assert_eq!(RatatoskrError::AuthenticationFailed.retry_after(), None);
}
