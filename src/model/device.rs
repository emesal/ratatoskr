//! Device configuration for local inference.

/// Compute device for local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// CPU execution (default).
    #[default]
    Cpu,

    /// CUDA GPU execution.
    #[cfg(feature = "cuda")]
    Cuda {
        /// GPU device ID (0-indexed).
        device_id: u32,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn cpu_constructor() {
        assert_eq!(Device::cpu(), Device::Cpu);
    }

    #[test]
    fn cpu_name() {
        assert_eq!(Device::Cpu.name(), "CPU");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_constructor() {
        let device = Device::cuda(0);
        assert_eq!(device, Device::Cuda { device_id: 0 });
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_name() {
        assert_eq!(Device::cuda(0).name(), "CUDA");
    }
}
