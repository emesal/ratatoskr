//! Stub types for future capabilities (Phase 2+)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding result (Phase 2+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub model: String,
    pub dimensions: usize,
}

/// NLI result (Phase 2+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NliResult {
    pub entailment: f32,
    pub contradiction: f32,
    pub neutral: f32,
    pub label: NliLabel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NliLabel {
    Entailment,
    Contradiction,
    Neutral,
}

/// Zero-shot classification result (Phase 2+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResult {
    pub scores: HashMap<String, f32>,
    pub top_label: String,
    pub confidence: f32,
}
