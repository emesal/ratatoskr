//! Model management and loading infrastructure.

#[cfg(feature = "local-inference")]
pub mod device;
#[cfg(feature = "local-inference")]
pub mod manager;
#[cfg(feature = "local-inference")]
pub mod source;

#[cfg(feature = "local-inference")]
pub use device::Device;
#[cfg(feature = "local-inference")]
pub use manager::{LoadedModels, ModelManager, ModelManagerConfig};
#[cfg(feature = "local-inference")]
pub use source::ModelSource;
