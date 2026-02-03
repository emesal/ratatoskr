//! Local NLI inference via ONNX Runtime.
//!
//! Uses cross-encoder models for natural language inference.

use crate::error::{Result, RatatoskrError};
use crate::model::Device;
use crate::types::{NliLabel, NliResult};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::PathBuf;

/// Supported local NLI models.
#[derive(Debug, Clone)]
pub enum LocalNliModel {
    /// cross-encoder/nli-deberta-v3-base — good balance of speed/accuracy.
    NliDebertaV3Base,
    /// cross-encoder/nli-deberta-v3-small — faster, slightly less accurate.
    NliDebertaV3Small,
    /// Custom model from local paths.
    Custom {
        model_path: PathBuf,
        tokenizer_path: PathBuf,
    },
}

impl LocalNliModel {
    /// Get the HuggingFace repo ID for this model.
    pub fn repo_id(&self) -> Option<&'static str> {
        match self {
            Self::NliDebertaV3Base => Some("cross-encoder/nli-deberta-v3-base"),
            Self::NliDebertaV3Small => Some("cross-encoder/nli-deberta-v3-small"),
            Self::Custom { .. } => None,
        }
    }

    /// Get the model name for display.
    pub fn name(&self) -> &str {
        match self {
            Self::NliDebertaV3Base => "nli-deberta-v3-base",
            Self::NliDebertaV3Small => "nli-deberta-v3-small",
            Self::Custom { model_path, .. } => model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom"),
        }
    }

    /// Get cache key for model manager.
    pub(crate) fn cache_key(&self) -> String {
        format!("onnx-nli:{}", self.name())
    }

    /// Resolve model and tokenizer paths, downloading if needed.
    fn resolve_paths(&self, cache_dir: &std::path::Path) -> Result<(PathBuf, PathBuf)> {
        match self {
            Self::Custom {
                model_path,
                tokenizer_path,
            } => Ok((model_path.clone(), tokenizer_path.clone())),
            _ => {
                let repo_id = self.repo_id().unwrap();
                download_model(repo_id, cache_dir)
            }
        }
    }
}

/// Information about an NLI model.
#[derive(Debug, Clone)]
pub struct NliModelInfo {
    /// Model name.
    pub name: String,
    /// Label order (entailment, neutral, contradiction for most models).
    pub labels: Vec<String>,
}

impl From<&LocalNliModel> for NliModelInfo {
    fn from(model: &LocalNliModel) -> Self {
        Self {
            name: model.name().to_string(),
            // Standard cross-encoder NLI label order
            labels: vec![
                "contradiction".to_string(),
                "entailment".to_string(),
                "neutral".to_string(),
            ],
        }
    }
}

/// Local NLI provider using ONNX Runtime.
pub struct OnnxNliProvider {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    model_info: NliModelInfo,
    #[allow(dead_code)]
    device: Device,
}

impl OnnxNliProvider {
    /// Create a new provider with the specified model.
    ///
    /// Downloads the model if not cached locally.
    pub fn new(model: LocalNliModel, device: Device) -> Result<Self> {
        let cache_dir = get_cache_dir();
        let (model_path, tokenizer_path) = model.resolve_paths(&cache_dir)?;

        let session = build_session(&model_path, &device)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RatatoskrError::Configuration(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self {
            session,
            tokenizer,
            model_info: (&model).into(),
            device,
        })
    }

    /// Run NLI inference on a single premise-hypothesis pair.
    pub fn infer_nli(&mut self, premise: &str, hypothesis: &str) -> Result<NliResult> {
        let (input_ids, attention_mask, token_type_ids) =
            self.encode_pair(premise, hypothesis)?;

        let outputs = self.run_inference(&input_ids, &attention_mask, Some(&token_type_ids))?;
        scores_to_result(&outputs[0], &self.model_info)
    }

    /// Run NLI inference on multiple premise-hypothesis pairs.
    ///
    /// More efficient than calling `infer_nli` multiple times.
    pub fn infer_nli_batch(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<NliResult>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // For now, process sequentially — batch encoding requires more work
        // (padding, attention mask handling). Can optimize later if needed.
        pairs
            .iter()
            .map(|(premise, hypothesis)| self.infer_nli(premise, hypothesis))
            .collect()
    }

    /// Get model information.
    pub fn model_info(&self) -> &NliModelInfo {
        &self.model_info
    }

    /// Encode a premise-hypothesis pair for the model.
    fn encode_pair(
        &self,
        premise: &str,
        hypothesis: &str,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| RatatoskrError::DataError(format!("Tokenization failed: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| t as i64)
            .collect();

        Ok((input_ids, attention_mask, token_type_ids))
    }

    /// Run the ONNX session.
    fn run_inference(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: Option<&[i64]>,
    ) -> Result<Vec<Vec<f32>>> {
        use ort::value::TensorRef;

        let seq_len = input_ids.len();
        let shape = [1_usize, seq_len];

        // Create tensor views — ort v2 expects (shape, slice)
        let input_ids_tensor =
            TensorRef::from_array_view((shape, input_ids)).map_err(|e| {
                RatatoskrError::DataError(format!("Failed to create input_ids tensor: {}", e))
            })?;

        let attention_mask_tensor =
            TensorRef::from_array_view((shape, attention_mask)).map_err(|e| {
                RatatoskrError::DataError(format!("Failed to create attention_mask tensor: {}", e))
            })?;

        // Build inputs dynamically based on model requirements
        // ort::inputs! returns a Vec directly
        let outputs = if let Some(type_ids) = token_type_ids {
            let token_type_ids_tensor =
                TensorRef::from_array_view((shape, type_ids)).map_err(|e| {
                    RatatoskrError::DataError(format!(
                        "Failed to create token_type_ids tensor: {}",
                        e
                    ))
                })?;
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                ])
                .map_err(|e| RatatoskrError::DataError(format!("ONNX inference failed: {}", e)))?
        } else {
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ])
                .map_err(|e| RatatoskrError::DataError(format!("ONNX inference failed: {}", e)))?
        };

        // Extract logits from output
        let logits = outputs
            .get("logits")
            .ok_or_else(|| RatatoskrError::DataError("No logits output found".to_string()))?;

        // try_extract_tensor returns (&Shape, &[T])
        let (tensor_shape, logits_data) = logits.try_extract_tensor::<f32>().map_err(|e| {
            RatatoskrError::DataError(format!("Failed to extract logits: {}", e))
        })?;

        // Convert to Vec<Vec<f32>> (batch_size x num_labels)
        let batch_size = tensor_shape[0] as usize;
        let num_labels = tensor_shape[1] as usize;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * num_labels;
            let end = start + num_labels;
            results.push(logits_data[start..end].to_vec());
        }

        Ok(results)
    }
}

/// Convert logits to NliResult with softmax probabilities.
fn scores_to_result(logits: &[f32], _model_info: &NliModelInfo) -> Result<NliResult> {
    if logits.len() != 3 {
        return Err(RatatoskrError::DataError(format!(
            "Expected 3 logits, got {}",
            logits.len()
        )));
    }

    let probs = softmax(logits);

    // Cross-encoder models use order: contradiction (0), entailment (1), neutral (2)
    // Map to our struct fields
    let contradiction = probs[0];
    let entailment = probs[1];
    let neutral = probs[2];

    let label = if entailment >= contradiction && entailment >= neutral {
        NliLabel::Entailment
    } else if contradiction >= neutral {
        NliLabel::Contradiction
    } else {
        NliLabel::Neutral
    };

    Ok(NliResult {
        entailment,
        contradiction,
        neutral,
        label,
    })
}

/// Softmax function.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

/// Build an ONNX session with the appropriate execution provider.
fn build_session(model_path: &std::path::Path, device: &Device) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| RatatoskrError::Configuration(format!("Failed to create session builder: {}", e)))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RatatoskrError::Configuration(format!("Failed to set optimization level: {}", e)))?;

    // Configure execution provider based on device
    let builder = match device {
        Device::Cpu => builder,
        #[cfg(feature = "cuda")]
        Device::Cuda { device_id } => {
            use ort::execution_providers::CUDAExecutionProvider;
            builder
                .with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(*device_id as i32)
                        .build(),
                ])
                .map_err(|e| {
                    RatatoskrError::Configuration(format!("Failed to configure CUDA: {}", e))
                })?
        }
    };

    builder
        .commit_from_file(model_path)
        .map_err(|e| RatatoskrError::Configuration(format!("Failed to load ONNX model: {}", e)))
}

/// Get the cache directory for models.
fn get_cache_dir() -> PathBuf {
    std::env::var("RATATOSKR_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("ratatoskr")
                .join("models")
        })
}

/// Download model and tokenizer from HuggingFace Hub.
fn download_model(repo_id: &str, _cache_dir: &std::path::Path) -> Result<(PathBuf, PathBuf)> {
    use hf_hub::api::sync::Api;

    let api = Api::new().map_err(|e| {
        RatatoskrError::Configuration(format!("Failed to initialize HF Hub API: {}", e))
    })?;

    let repo = api.model(repo_id.to_string());

    // Download ONNX model
    let model_path = repo.get("onnx/model.onnx").map_err(|e| {
        RatatoskrError::Configuration(format!("Failed to download ONNX model: {}", e))
    })?;

    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
        RatatoskrError::Configuration(format!("Failed to download tokenizer: {}", e))
    })?;

    Ok((model_path, tokenizer_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check relative ordering preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_scores_to_result_entailment() {
        let model_info = NliModelInfo {
            name: "test".to_string(),
            labels: vec!["contradiction".into(), "entailment".into(), "neutral".into()],
        };

        // High entailment score (index 1)
        let logits = vec![-1.0, 3.0, 0.0];
        let result = scores_to_result(&logits, &model_info).unwrap();

        assert_eq!(result.label, NliLabel::Entailment);
        assert!(result.entailment > result.contradiction);
        assert!(result.entailment > result.neutral);
    }

    #[test]
    fn test_scores_to_result_contradiction() {
        let model_info = NliModelInfo {
            name: "test".to_string(),
            labels: vec!["contradiction".into(), "entailment".into(), "neutral".into()],
        };

        // High contradiction score (index 0)
        let logits = vec![3.0, -1.0, 0.0];
        let result = scores_to_result(&logits, &model_info).unwrap();

        assert_eq!(result.label, NliLabel::Contradiction);
        assert!(result.contradiction > result.entailment);
        assert!(result.contradiction > result.neutral);
    }

    #[test]
    fn test_local_nli_model_properties() {
        let base = LocalNliModel::NliDebertaV3Base;
        assert_eq!(base.name(), "nli-deberta-v3-base");
        assert_eq!(base.repo_id(), Some("cross-encoder/nli-deberta-v3-base"));
        assert_eq!(base.cache_key(), "onnx-nli:nli-deberta-v3-base");

        let small = LocalNliModel::NliDebertaV3Small;
        assert_eq!(small.name(), "nli-deberta-v3-small");
        assert_eq!(small.repo_id(), Some("cross-encoder/nli-deberta-v3-small"));

        let custom = LocalNliModel::Custom {
            model_path: PathBuf::from("/path/to/model.onnx"),
            tokenizer_path: PathBuf::from("/path/to/tokenizer.json"),
        };
        assert_eq!(custom.name(), "model");
        assert_eq!(custom.repo_id(), None);
    }
}
