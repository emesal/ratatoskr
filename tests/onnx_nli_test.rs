//! Tests for ONNX NLI provider.

#![cfg(feature = "local-inference")]

use ratatoskr::providers::onnx_nli::{LocalNliModel, NliModelInfo, OnnxNliProvider};
use ratatoskr::{Device, NliLabel};
use std::path::PathBuf;

#[test]
fn test_local_nli_model_properties() {
    let base = LocalNliModel::NliDebertaV3Base;
    assert_eq!(base.name(), "nli-deberta-v3-base");
    assert_eq!(base.repo_id(), Some("cross-encoder/nli-deberta-v3-base"));

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

#[test]
fn test_nli_model_info_from_model() {
    let base = LocalNliModel::NliDebertaV3Base;
    let info: NliModelInfo = (&base).into();

    assert_eq!(info.name, "nli-deberta-v3-base");
    assert_eq!(info.labels.len(), 3);
    assert!(info.labels.contains(&"entailment".to_string()));
    assert!(info.labels.contains(&"neutral".to_string()));
    assert!(info.labels.contains(&"contradiction".to_string()));
}

#[test]
fn test_device_default() {
    let device = Device::default();
    assert_eq!(device, Device::Cpu);
    assert_eq!(device.name(), "CPU");
}

// Live test - requires model download, run with --ignored
#[test]
#[ignore]
fn test_onnx_nli_inference() {
    // This test downloads the model on first run
    let mut provider =
        OnnxNliProvider::new(LocalNliModel::NliDebertaV3Small, Device::Cpu).unwrap();

    // Test entailment
    let result = provider
        .infer_nli("The cat is on the mat.", "There is a cat.")
        .unwrap();

    assert_eq!(result.label, NliLabel::Entailment);
    assert!(result.entailment > 0.5);

    // Test contradiction
    let result = provider
        .infer_nli("The cat is black.", "The cat is white.")
        .unwrap();

    assert_eq!(result.label, NliLabel::Contradiction);
    assert!(result.contradiction > 0.5);

    // Test neutral
    let result = provider
        .infer_nli("A person is walking.", "The person is happy.")
        .unwrap();

    assert_eq!(result.label, NliLabel::Neutral);
}

// Live test for batch inference
#[test]
#[ignore]
fn test_onnx_nli_batch_inference() {
    let mut provider =
        OnnxNliProvider::new(LocalNliModel::NliDebertaV3Small, Device::Cpu).unwrap();

    let pairs = [
        ("The sky is blue.", "The sky has a color."),
        ("It is raining.", "The sun is shining brightly."),
    ];

    let results = provider.infer_nli_batch(&pairs).unwrap();
    assert_eq!(results.len(), 2);

    // First should be entailment
    assert_eq!(results[0].label, NliLabel::Entailment);

    // Second should be contradiction
    assert_eq!(results[1].label, NliLabel::Contradiction);
}
