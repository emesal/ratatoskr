//! Gateway implementations

mod builder;
mod embedded;
pub mod routing;

pub use builder::{Ratatoskr, RatatoskrBuilder};
pub use embedded::EmbeddedGateway;
