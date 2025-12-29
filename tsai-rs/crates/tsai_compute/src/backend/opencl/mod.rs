//! OpenCL compute backend.
//!
//! This backend provides cross-platform GPU/CPU computation via OpenCL,
//! supporting a wide range of hardware from various vendors.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, DeviceCapabilities, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// OpenCL device representation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpenClDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
}

impl ComputeDevice for OpenClDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::OpenClDevice
    }

    fn device_id(&self) -> DeviceId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

/// OpenCL buffer wrapper.
pub struct OpenClBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for OpenClBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn map(&self) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "OpenCL buffer mapping not yet implemented".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "OpenCL buffer mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }
}

/// OpenCL command encoder.
pub struct OpenClCommandEncoder {
    commands: Vec<()>,
}

impl CommandEncoder for OpenClCommandEncoder {
    type Buffer = OpenClBuffer;

    fn copy_host_to_device(&mut self, _src: &[u8], _dst: &Self::Buffer, _offset: usize) {}
    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], _offset: usize) {}
    fn copy_buffer_to_buffer(
        &mut self,
        _src: &Self::Buffer,
        _src_offset: usize,
        _dst: &Self::Buffer,
        _dst_offset: usize,
        _size: usize,
    ) {
    }
    fn fill_buffer(&mut self, _buffer: &Self::Buffer, _offset: usize, _size: usize, _value: u8) {}
    fn barrier(&mut self) {}
}

/// OpenCL fence.
pub struct OpenClFence {
    completed: AtomicBool,
}

impl Fence for OpenClFence {
    fn is_signaled(&self) -> bool {
        self.completed.load(Ordering::SeqCst)
    }

    fn wait(&self) {
        while !self.is_signaled() {
            std::thread::yield_now();
        }
    }

    fn wait_timeout(&self, timeout_ms: u64) -> bool {
        let start = std::time::Instant::now();
        while !self.is_signaled() {
            if start.elapsed() >= std::time::Duration::from_millis(timeout_ms) {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

/// OpenCL compute backend.
pub struct OpenClBackend {
    device: OpenClDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for OpenClBackend {
    type Device = OpenClDevice;
    type Buffer = OpenClBuffer;
    type CommandEncoder = OpenClCommandEncoder;
    type Fence = OpenClFence;

    fn name() -> &'static str {
        "OpenCL"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        #[cfg(feature = "opencl")]
        {
            // TODO: Implement using opencl3
            Err(ComputeError::DiscoveryFailed(
                "OpenCL device enumeration not yet implemented".to_string(),
            ))
        }

        #[cfg(not(feature = "opencl"))]
        {
            Err(ComputeError::BackendInitFailed(
                "OpenCL support not compiled".to_string(),
            ))
        }
    }

    fn new(_device: &Self::Device) -> ComputeResult<Self> {
        Err(ComputeError::BackendInitFailed(
            "OpenCL backend not yet implemented".to_string(),
        ))
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, _size: usize, _usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        Err(ComputeError::BackendInitFailed(
            "OpenCL backend not yet implemented".to_string(),
        ))
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Err(ComputeError::BackendInitFailed(
            "OpenCL backend not yet implemented".to_string(),
        ))
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        Err(ComputeError::BackendInitFailed(
            "OpenCL backend not yet implemented".to_string(),
        ))
    }

    fn wait(&self, _fence: &Self::Fence) -> ComputeResult<()> {
        Ok(())
    }

    fn synchronize(&self) -> ComputeResult<()> {
        Ok(())
    }

    fn seed(&self, seed: u64) {
        self.rng_seed.store(seed, Ordering::SeqCst);
    }
}
