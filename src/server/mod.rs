//! gRPC server implementation for ratd.
//!
//! This module provides the gRPC service that exposes the ModelGateway
//! interface over the network.
//!
//! # Transport Extensibility
//!
//! Currently only TCP transport is supported. The transport layer is
//! abstracted to allow future addition of Unix socket support for
//! tighter permission control on multi-user systems.

pub mod convert;
pub mod service;

/// Re-exported generated proto types.
pub mod proto {
    tonic::include_proto!("ratatoskr.v1");
}

pub use service::RatatoskrService;
