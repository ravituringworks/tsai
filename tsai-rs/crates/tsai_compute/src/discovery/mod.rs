//! Hardware discovery orchestration.
//!
//! This module provides unified device discovery across all backends.

mod cpu;
mod gpu;

pub use cpu::*;
pub use gpu::*;

use std::sync::Arc;

use crate::backend::cpu::CpuBackend;
use crate::backend::ComputeBackend;
use crate::device::{DevicePool, DeviceType};
use crate::error::ComputeResult;

/// Hardware discovery orchestrator.
///
/// Discovers all available compute devices across all backends
/// and registers them in a device pool.
pub struct HardwareDiscovery;

impl HardwareDiscovery {
    /// Perform full system discovery.
    ///
    /// This discovers all available compute devices and returns
    /// a pool containing them.
    pub fn discover_all() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();

        // Phase 1: CPU discovery (always available)
        Self::discover_cpu(&pool)?;

        // Phase 2: GPU discovery (feature-gated)
        #[cfg(target_os = "macos")]
        Self::discover_metal(&pool)?;

        #[cfg(feature = "cuda")]
        Self::discover_cuda(&pool)?;

        #[cfg(feature = "vulkan")]
        Self::discover_vulkan(&pool)?;

        #[cfg(feature = "opencl")]
        Self::discover_opencl(&pool)?;

        #[cfg(feature = "rocm")]
        Self::discover_rocm(&pool)?;

        // Phase 3: Select default device
        pool.auto_select_default()?;

        Ok(pool)
    }

    /// Discover CPU devices.
    pub fn discover_cpu(pool: &DevicePool) -> ComputeResult<()> {
        let devices = CpuBackend::enumerate_devices()?;
        pool.register_all(devices)?;
        tracing::info!("Discovered {} CPU device(s)", pool.devices_of_type(DeviceType::Cpu).len());
        Ok(())
    }

    /// Discover Metal devices (macOS).
    #[cfg(target_os = "macos")]
    pub fn discover_metal(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::metal::MetalBackend;

        match MetalBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} Metal device(s)", count);
            }
            Err(e) => {
                tracing::warn!("Metal discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover CUDA devices.
    #[cfg(feature = "cuda")]
    pub fn discover_cuda(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::cuda::CudaBackend;

        match CudaBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} CUDA device(s)", count);
            }
            Err(e) => {
                tracing::warn!("CUDA discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover Vulkan devices.
    #[cfg(feature = "vulkan")]
    pub fn discover_vulkan(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::vulkan::VulkanBackend;

        match VulkanBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} Vulkan device(s)", count);
            }
            Err(e) => {
                tracing::debug!("Vulkan discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover OpenCL devices.
    #[cfg(feature = "opencl")]
    pub fn discover_opencl(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::opencl::OpenClBackend;

        match OpenClBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} OpenCL device(s)", count);
            }
            Err(e) => {
                tracing::debug!("OpenCL discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover ROCm devices.
    #[cfg(feature = "rocm")]
    pub fn discover_rocm(pool: &DevicePool) -> ComputeResult<()> {
        use crate::backend::rocm::RocmBackend;

        match RocmBackend::enumerate_devices() {
            Ok(devices) => {
                let count = devices.len();
                pool.register_all(devices)?;
                tracing::info!("Discovered {} ROCm device(s)", count);
            }
            Err(e) => {
                tracing::debug!("ROCm discovery failed: {}", e);
            }
        }
        Ok(())
    }

    /// Discover only CPU devices.
    pub fn discover_cpu_only() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();
        Self::discover_cpu(&pool)?;
        pool.auto_select_default()?;
        Ok(pool)
    }

    /// Discover only GPU devices.
    pub fn discover_gpu_only() -> ComputeResult<DevicePool> {
        let pool = DevicePool::new();

        #[cfg(target_os = "macos")]
        Self::discover_metal(&pool)?;

        #[cfg(feature = "cuda")]
        Self::discover_cuda(&pool)?;

        #[cfg(feature = "vulkan")]
        Self::discover_vulkan(&pool)?;

        #[cfg(feature = "opencl")]
        Self::discover_opencl(&pool)?;

        #[cfg(feature = "rocm")]
        Self::discover_rocm(&pool)?;

        pool.auto_select_default()?;
        Ok(pool)
    }
}

/// Get a shared device pool with automatic discovery.
///
/// This is a convenience function that performs discovery once
/// and caches the result.
pub fn get_device_pool() -> ComputeResult<Arc<DevicePool>> {
    use std::sync::OnceLock;

    static POOL: OnceLock<Arc<DevicePool>> = OnceLock::new();

    // If already initialized, return it
    if let Some(pool) = POOL.get() {
        return Ok(pool.clone());
    }

    // Try to initialize
    let pool = Arc::new(HardwareDiscovery::discover_all()?);

    // Use get_or_init to handle race conditions (another thread may have initialized)
    Ok(POOL.get_or_init(|| pool).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_all() {
        let pool = HardwareDiscovery::discover_all().unwrap();
        assert!(pool.has_devices());
        pool.print_summary();
    }

    #[test]
    fn test_discovery_cpu() {
        let pool = HardwareDiscovery::discover_cpu_only().unwrap();
        assert!(!pool.cpu_devices().is_empty());
    }
}
