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
