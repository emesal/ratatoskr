//! Stance detection types.
//!
//! Stance detection determines whether text expresses favor, opposition, or
//! neutrality toward a specific target topic.

use serde::{Deserialize, Serialize};

/// The detected stance label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StanceLabel {
    /// Text expresses support for the target.
    Favor,
    /// Text expresses opposition to the target.
    Against,
    /// Text is neutral or does not express a clear stance.
    Neutral,
}

/// Result of stance detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StanceResult {
    /// Score for favor stance (0.0 to 1.0).
    pub favor: f32,
    /// Score for against stance (0.0 to 1.0).
    pub against: f32,
    /// Score for neutral stance (0.0 to 1.0).
    pub neutral: f32,
    /// The determined stance label (highest score).
    pub label: StanceLabel,
    /// The target topic that stance was measured against.
    pub target: String,
}

impl StanceResult {
    /// Create a stance result from individual scores.
    ///
    /// The label is determined from the highest score. On ties, the priority
    /// order is: `Favor` > `Against` > `Neutral`.
    pub fn from_scores(favor: f32, against: f32, neutral: f32, target: impl Into<String>) -> Self {
        let label = if favor >= against && favor >= neutral {
            StanceLabel::Favor
        } else if against >= favor && against >= neutral {
            StanceLabel::Against
        } else {
            StanceLabel::Neutral
        };
        Self {
            favor,
            against,
            neutral,
            label,
            target: target.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stance_result_from_scores_favor() {
        let result = StanceResult::from_scores(0.8, 0.1, 0.1, "climate change");
        assert_eq!(result.label, StanceLabel::Favor);
        assert_eq!(result.target, "climate change");
    }

    #[test]
    fn stance_result_from_scores_against() {
        let result = StanceResult::from_scores(0.1, 0.7, 0.2, "new policy");
        assert_eq!(result.label, StanceLabel::Against);
    }

    #[test]
    fn stance_result_from_scores_neutral() {
        let result = StanceResult::from_scores(0.2, 0.2, 0.6, "topic");
        assert_eq!(result.label, StanceLabel::Neutral);
    }

    #[test]
    fn stance_result_from_scores_tie_favors_favor() {
        // When favor == against, favor wins (checked first)
        let result = StanceResult::from_scores(0.5, 0.5, 0.0, "tie");
        assert_eq!(result.label, StanceLabel::Favor);
    }
}
