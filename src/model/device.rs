//! Device configuration for local inference.

/// Compute device for local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU execution.
    Cpu,

    /// CUDA GPU execution.
    #[cfg(feature = "cuda")]
    Cuda {
        /// GPU device ID (0-indexed).
        device_id: u32,
    },
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl Device {
    /// Create CPU device.
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create CUDA device with the given device ID.
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: u32) -> Self {
        Self::Cuda { device_id }
    }

    /// Get the device name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => "CUDA",
        }
    }
}

// Note: execution_provider helper will be implemented in Task 7 (ONNX NLI Provider)
// when we actually need to configure ONNX Runtime sessions.
