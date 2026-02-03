//! Token types for tokenization results.

use serde::{Deserialize, Serialize};

/// A single token from tokenization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Token {
    /// Token ID in the vocabulary.
    pub id: u32,
    /// Text representation of the token.
    pub text: String,
    /// Start byte offset in the original text.
    pub start: usize,
    /// End byte offset in the original text (exclusive).
    pub end: usize,
}

impl Token {
    /// Create a new token.
    pub fn new(id: u32, text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            id,
            text: text.into(),
            start,
            end,
        }
    }

    /// Get the byte length of this token in the original text.
    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_new() {
        let token = Token::new(42, "hello", 0, 5);
        assert_eq!(token.id, 42);
        assert_eq!(token.text, "hello");
        assert_eq!(token.start, 0);
        assert_eq!(token.end, 5);
    }

    #[test]
    fn token_byte_len() {
        let token = Token::new(1, "test", 10, 14);
        assert_eq!(token.byte_len(), 4);
    }
}
