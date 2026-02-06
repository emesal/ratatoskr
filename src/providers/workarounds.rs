//! Provider-specific translation layer for parameters that require
//! backend-specific handling.
//!
//! All "this provider needs it *this* way" logic lives here and nowhere else.
//! When the llm crate gains native support for these features, entries here
//! can be removed without touching the rest of the codebase.

use llm::builder::LLMBackend;
use serde_json::{Map, Value};

use crate::ChatOptions;
use crate::error::{RatatoskrError, Result};

/// Adjustments computed by the workarounds module for a specific
/// backend + options combination.
///
/// Keeps provider-specific translation logic isolated from the
/// main provider implementation.
#[derive(Debug, Default)]
pub(crate) struct ProviderAdjustments {
    /// Extra fields to merge into the request body via `LLMBuilder::extra_body`.
    /// Only works for backends with `#[serde(flatten)]` support (openai-compatible).
    pub extra_body: Option<Value>,

    /// Native parallel tool use flag (for backends like mistral where the
    /// llm crate's `enable_parallel_tool_use` actually works).
    pub native_parallel_tool_calls: Option<bool>,
}

/// Compute provider-specific adjustments for the given backend and options.
///
/// This is the single entry point for all workaround logic. The returned
/// adjustments should be applied to the `LLMBuilder` in `build_provider()`.
///
/// # Errors
///
/// Returns `UnsupportedParameter` if a requested parameter cannot be
/// supported by the target backend (e.g. `parallel_tool_calls` on
/// anthropic direct).
pub(crate) fn compute_adjustments(
    backend: &LLMBackend,
    options: &ChatOptions,
) -> Result<ProviderAdjustments> {
    let mut adj = ProviderAdjustments::default();

    // start with raw_provider_options as base extra_body (if any)
    let mut extra_map: Map<String, Value> = options
        .raw_provider_options
        .as_ref()
        .and_then(|v: &Value| v.as_object().cloned())
        .unwrap_or_default();

    // parallel_tool_calls — backend-specific routing
    if let Some(ptc) = options.parallel_tool_calls {
        match backend {
            LLMBackend::Mistral => {
                adj.native_parallel_tool_calls = Some(ptc);
            }
            LLMBackend::OpenRouter | LLMBackend::OpenAI => {
                extra_map.insert("parallel_tool_calls".into(), Value::Bool(ptc));
            }
            LLMBackend::Anthropic => {
                return Err(RatatoskrError::UnsupportedParameter {
                    param: "parallel_tool_calls".into(),
                    model: options.model.clone(),
                    provider: "anthropic".into(),
                });
            }
            // google, ollama, deepseek, xai, etc. — silently ignore
            _ => {}
        }
    }

    // only set extra_body if we actually have entries
    if !extra_map.is_empty() {
        adj.extra_body = Some(Value::Object(extra_map));
    }

    Ok(adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_openrouter_parallel_true() {
        let opts = ChatOptions::default()
            .model("openai/gpt-4o")
            .parallel_tool_calls(true);
        let adj = compute_adjustments(&LLMBackend::OpenRouter, &opts).unwrap();
        assert_eq!(
            adj.extra_body.unwrap()["parallel_tool_calls"],
            Value::Bool(true)
        );
        assert!(adj.native_parallel_tool_calls.is_none());
    }

    #[test]
    fn test_openrouter_parallel_false() {
        let opts = ChatOptions::default()
            .model("openai/gpt-4o")
            .parallel_tool_calls(false);
        let adj = compute_adjustments(&LLMBackend::OpenRouter, &opts).unwrap();
        assert_eq!(
            adj.extra_body.unwrap()["parallel_tool_calls"],
            Value::Bool(false)
        );
    }

    #[test]
    fn test_mistral_parallel_native() {
        let opts = ChatOptions::default()
            .model("mistral-large")
            .parallel_tool_calls(true);
        let adj = compute_adjustments(&LLMBackend::Mistral, &opts).unwrap();
        assert_eq!(adj.native_parallel_tool_calls, Some(true));
        assert!(adj.extra_body.is_none());
    }

    #[test]
    fn test_anthropic_direct_errors() {
        let opts = ChatOptions::default()
            .model("claude-3.5-sonnet")
            .parallel_tool_calls(false);
        let err = compute_adjustments(&LLMBackend::Anthropic, &opts).unwrap_err();
        assert!(
            matches!(err, RatatoskrError::UnsupportedParameter { ref param, .. } if param == "parallel_tool_calls")
        );
    }

    #[test]
    fn test_google_ollama_ignored() {
        for backend in [LLMBackend::Google, LLMBackend::Ollama] {
            let opts = ChatOptions::default()
                .model("test")
                .parallel_tool_calls(true);
            let adj = compute_adjustments(&backend, &opts).unwrap();
            assert!(adj.extra_body.is_none());
            assert!(adj.native_parallel_tool_calls.is_none());
        }
    }

    #[test]
    fn test_raw_provider_options_passthrough() {
        let mut opts = ChatOptions::default().model("openai/gpt-4o");
        opts.raw_provider_options = Some(json!({"custom_key": "custom_value"}));
        let adj = compute_adjustments(&LLMBackend::OpenRouter, &opts).unwrap();
        let extra = adj.extra_body.unwrap();
        assert_eq!(extra["custom_key"], "custom_value");
    }

    #[test]
    fn test_parallel_overrides_raw() {
        let mut opts = ChatOptions::default()
            .model("openai/gpt-4o")
            .parallel_tool_calls(true);
        opts.raw_provider_options = Some(json!({"parallel_tool_calls": false, "other": 42}));
        let adj = compute_adjustments(&LLMBackend::OpenRouter, &opts).unwrap();
        let extra = adj.extra_body.unwrap();
        // typed field wins over raw
        assert_eq!(extra["parallel_tool_calls"], Value::Bool(true));
        assert_eq!(extra["other"], 42);
    }

    #[test]
    fn test_no_options_no_adjustments() {
        let opts = ChatOptions::default().model("gpt-4o");
        let adj = compute_adjustments(&LLMBackend::OpenRouter, &opts).unwrap();
        assert!(adj.extra_body.is_none());
        assert!(adj.native_parallel_tool_calls.is_none());
    }
}
