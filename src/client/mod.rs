//! Client library for connecting to ratd.
//!
//! Provides [`ServiceClient`], which implements [`ModelGateway`](crate::ModelGateway)
//! by forwarding calls to a remote ratd instance over gRPC.

mod service_client;

pub use service_client::ServiceClient;
