//! gRPC server and shared proto types.
//!
//! This module provides:
//! - Generated protobuf types (`proto`) used by both server and client
//! - Type conversions between native and proto types (`convert`)
//! - The gRPC service implementation (`service`, server-only)
//! - Configuration types (`config`, server-only)
//!
//! # Transport Extensibility
//!
//! Currently only TCP transport is supported. The transport layer is
//! abstracted to allow future addition of Unix socket support for
//! tighter permission control on multi-user systems.

#[cfg(feature = "server")]
pub mod config;
pub mod convert;
#[cfg(feature = "server")]
pub mod service;

/// Re-exported generated proto types.
pub mod proto {
    tonic::include_proto!("ratatoskr.v1");
}

#[cfg(feature = "server")]
pub use service::RatatoskrService;
