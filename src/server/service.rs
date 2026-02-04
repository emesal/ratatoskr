//! gRPC service implementation.
//!
//! Placeholder — full implementation in Task 4.

use std::sync::Arc;

use crate::ModelGateway;

/// gRPC service that wraps a ModelGateway implementation.
pub struct RatatoskrService<G: ModelGateway> {
    gateway: Arc<G>,
}

impl<G: ModelGateway> RatatoskrService<G> {
    /// Create a new service wrapping the given gateway.
    pub fn new(gateway: Arc<G>) -> Self {
        Self { gateway }
    }
}
