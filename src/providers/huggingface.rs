//! HuggingFace Inference API client for embeddings, NLI, and classification.
//!
//! This client uses HuggingFace's serverless inference endpoints.
//! See: <https://huggingface.co/docs/api-inference/index>

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::traits::{ClassifyProvider, EmbeddingProvider, NliProvider};
use crate::{ClassifyResult, Embedding, NliLabel, NliResult, RatatoskrError, Result};

/// Default base URL for HuggingFace Inference API
const DEFAULT_BASE_URL: &str = "https://api-inference.huggingface.co";

/// Client for HuggingFace Inference API.
///
/// Supports:
/// - Feature extraction (embeddings)
/// - NLI via zero-shot classification models
/// - Zero-shot classification
#[derive(Clone)]
pub struct HuggingFaceClient {
    api_key: String,
    http: Client,
    base_url: String,
}

impl HuggingFaceClient {
    /// Create a new HuggingFace client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Create a client with a custom base URL (for testing with wiremock).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build HTTP client");

        Self {
            api_key: api_key.into(),
            http,
            base_url: base_url.into(),
        }
    }

    /// Generate embeddings for a single text.
    ///
    /// Uses the feature-extraction pipeline.
    ///
    /// # Arguments
    /// * `text` - Text to embed
    /// * `model` - Full HuggingFace model ID (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
    pub async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        let url = format!("{}/pipeline/feature-extraction/{}", self.base_url, model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&EmbedRequest { inputs: text })
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        self.handle_response_errors(&response, model)?;

        // Response is [[f32; dim]] for single input
        let values: Vec<Vec<f32>> = response
            .json()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        let embedding_values = values
            .into_iter()
            .next()
            .ok_or(RatatoskrError::EmptyResponse)?;

        Ok(Embedding {
            dimensions: embedding_values.len(),
            values: embedding_values,
            model: model.to_string(),
        })
    }

    /// Generate embeddings for multiple texts.
    ///
    /// # Arguments
    /// * `texts` - Texts to embed
    /// * `model` - Full HuggingFace model ID
    pub async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        let url = format!("{}/pipeline/feature-extraction/{}", self.base_url, model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&EmbedBatchRequest { inputs: texts })
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        self.handle_response_errors(&response, model)?;

        // Response is [[[f32; dim]]] for batch (extra nesting)
        let values: Vec<Vec<Vec<f32>>> = response
            .json()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        Ok(values
            .into_iter()
            .map(|v| {
                let embedding_values = v.into_iter().next().unwrap_or_default();
                Embedding {
                    dimensions: embedding_values.len(),
                    values: embedding_values,
                    model: model.to_string(),
                }
            })
            .collect())
    }

    /// Perform natural language inference.
    ///
    /// Uses zero-shot classification models that take premise + hypothesis
    /// and return entailment/neutral/contradiction scores.
    ///
    /// # Arguments
    /// * `premise` - The premise text
    /// * `hypothesis` - The hypothesis to evaluate against the premise
    /// * `model` - Full HuggingFace model ID (e.g., `facebook/bart-large-mnli`)
    pub async fn infer_nli(
        &self,
        premise: &str,
        hypothesis: &str,
        model: &str,
    ) -> Result<NliResult> {
        let url = format!("{}/models/{}", self.base_url, model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&ZeroShotRequest {
                inputs: premise,
                parameters: ZeroShotParameters {
                    candidate_labels: vec!["entailment", "neutral", "contradiction"],
                    hypothesis_template: Some(hypothesis),
                },
            })
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        self.handle_response_errors(&response, model)?;

        let result: ZeroShotResponse = response
            .json()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        // Map labels to scores
        let mut entailment = 0.0f32;
        let mut neutral = 0.0f32;
        let mut contradiction = 0.0f32;

        for (label, score) in result.labels.iter().zip(result.scores.iter()) {
            match label.to_lowercase().as_str() {
                "entailment" => entailment = *score,
                "neutral" => neutral = *score,
                "contradiction" => contradiction = *score,
                _ => {}
            }
        }

        let label = if entailment >= neutral && entailment >= contradiction {
            NliLabel::Entailment
        } else if contradiction >= neutral {
            NliLabel::Contradiction
        } else {
            NliLabel::Neutral
        };

        Ok(NliResult {
            entailment,
            neutral,
            contradiction,
            label,
        })
    }

    /// Perform zero-shot classification.
    ///
    /// # Arguments
    /// * `text` - Text to classify
    /// * `labels` - Candidate labels
    /// * `model` - Full HuggingFace model ID (e.g., `facebook/bart-large-mnli`)
    pub async fn classify(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult> {
        let url = format!("{}/models/{}", self.base_url, model);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&ZeroShotRequest {
                inputs: text,
                parameters: ZeroShotParameters {
                    candidate_labels: labels.to_vec(),
                    hypothesis_template: None,
                },
            })
            .send()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        self.handle_response_errors(&response, model)?;

        let result: ZeroShotResponse = response
            .json()
            .await
            .map_err(|e| RatatoskrError::Http(e.to_string()))?;

        let mut scores = std::collections::HashMap::new();
        for (label, score) in result.labels.iter().zip(result.scores.iter()) {
            scores.insert(label.clone(), *score);
        }

        let (top_label, confidence) = result
            .labels
            .into_iter()
            .zip(result.scores.into_iter())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| (String::new(), 0.0));

        Ok(ClassifyResult {
            scores,
            top_label,
            confidence,
        })
    }

    /// Check response status and map to appropriate error.
    fn handle_response_errors(&self, response: &reqwest::Response, model: &str) -> Result<()> {
        let status = response.status();

        if status.is_success() {
            return Ok(());
        }

        match status.as_u16() {
            401 => Err(RatatoskrError::AuthenticationFailed),
            404 => Err(RatatoskrError::ModelNotFound(model.to_string())),
            429 => {
                // Try to parse retry-after header
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs);
                Err(RatatoskrError::RateLimited { retry_after })
            }
            503 => Err(RatatoskrError::Api {
                status: 503,
                message: "Model is loading, please retry".to_string(),
            }),
            code => Err(RatatoskrError::Api {
                status: code,
                message: format!("HuggingFace API error: {}", status),
            }),
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    inputs: &'a str,
}

#[derive(Serialize)]
struct EmbedBatchRequest<'a> {
    inputs: &'a [&'a str],
}

#[derive(Serialize)]
struct ZeroShotRequest<'a> {
    inputs: &'a str,
    parameters: ZeroShotParameters<'a>,
}

#[derive(Serialize)]
struct ZeroShotParameters<'a> {
    candidate_labels: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hypothesis_template: Option<&'a str>,
}

#[derive(Deserialize)]
struct ZeroShotResponse {
    labels: Vec<String>,
    scores: Vec<f32>,
}

// ============================================================================
// Provider Trait Implementations
// ============================================================================

/// HuggingFace accepts any model string and forwards it to the API.
/// It never returns `ModelNotAvailable` — it's a universal fallback.
#[async_trait]
impl EmbeddingProvider for HuggingFaceClient {
    fn name(&self) -> &str {
        "huggingface"
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Embedding> {
        // Delegate to the existing method
        HuggingFaceClient::embed(self, text, model).await
    }

    async fn embed_batch(&self, texts: &[&str], model: &str) -> Result<Vec<Embedding>> {
        // Delegate to the existing method
        HuggingFaceClient::embed_batch(self, texts, model).await
    }
}

#[async_trait]
impl NliProvider for HuggingFaceClient {
    fn name(&self) -> &str {
        "huggingface"
    }

    async fn infer_nli(&self, premise: &str, hypothesis: &str, model: &str) -> Result<NliResult> {
        // Delegate to the existing method
        HuggingFaceClient::infer_nli(self, premise, hypothesis, model).await
    }
}

#[async_trait]
impl ClassifyProvider for HuggingFaceClient {
    fn name(&self) -> &str {
        "huggingface"
    }

    async fn classify_zero_shot(
        &self,
        text: &str,
        labels: &[&str],
        model: &str,
    ) -> Result<ClassifyResult> {
        // Delegate to the existing `classify` method
        HuggingFaceClient::classify(self, text, labels, model).await
    }
}
