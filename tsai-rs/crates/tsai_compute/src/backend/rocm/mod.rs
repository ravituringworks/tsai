//! ROCm compute backend for AMD GPUs.
//!
//! This backend provides GPU computation via AMD ROCm/HIP,
//! supporting AMD Radeon and Instinct GPUs.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{ComputeDevice, DeviceCapabilities, DeviceId, DeviceType};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// ROCm device representation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RocmDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
}

impl ComputeDevice for RocmDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::RocmGpu
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

/// ROCm buffer wrapper.
pub struct RocmBuffer {
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
}

impl Buffer for RocmBuffer {
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
            "ROCm buffer mapping not yet implemented".to_string(),
        ))
    }

    fn map_range(&self, _offset: usize, _size: usize) -> ComputeResult<BufferMapping<'_>> {
        Err(ComputeError::MappingFailed(
            "ROCm buffer mapping not yet implemented".to_string(),
        ))
    }

    fn is_mapped(&self) -> bool {
        false
    }
}

/// ROCm command encoder.
pub struct RocmCommandEncoder {
    commands: Vec<()>,
}

impl CommandEncoder for RocmCommandEncoder {
    type Buffer = RocmBuffer;

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

/// ROCm fence.
pub struct RocmFence {
    completed: AtomicBool,
}

impl Fence for RocmFence {
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

/// ROCm compute backend.
pub struct RocmBackend {
    device: RocmDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for RocmBackend {
    type Device = RocmDevice;
    type Buffer = RocmBuffer;
    type CommandEncoder = RocmCommandEncoder;
    type Fence = RocmFence;

    fn name() -> &'static str {
        "ROCm"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        Err(ComputeError::BackendInitFailed(
            "ROCm support not yet implemented".to_string(),
        ))
    }

    fn new(_device: &Self::Device) -> ComputeResult<Self> {
        Err(ComputeError::BackendInitFailed(
            "ROCm backend not yet implemented".to_string(),
        ))
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, _size: usize, _usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        Err(ComputeError::BackendInitFailed(
            "ROCm backend not yet implemented".to_string(),
        ))
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Err(ComputeError::BackendInitFailed(
            "ROCm backend not yet implemented".to_string(),
        ))
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        Err(ComputeError::BackendInitFailed(
            "ROCm backend not yet implemented".to_string(),
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
