//! Gateway implementations

mod builder;
mod embedded;

pub use builder::{Ratatoskr, RatatoskrBuilder};
pub use embedded::EmbeddedGateway;
