//! Metal compute backend for Apple GPUs.
//!
//! This backend provides GPU computation via Apple's Metal framework,
//! supporting Apple Silicon (M1/M2/M3) and AMD GPUs on macOS.
//!
//! Note: Full Metal support requires objc2-metal API integration.
//! This is a stub implementation for compilation.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::backend::{CommandEncoder, ComputeBackend, Fence};
use crate::device::{
    ComputeDevice, ComputeVersion, DeviceCapabilities, DeviceFeature, DeviceId, DeviceType,
    MetalFamily, Precision,
};
use crate::error::{ComputeError, ComputeResult};
use crate::memory::{Buffer, BufferMapping, BufferUsage};

/// Metal device representation.
#[derive(Debug, Clone)]
pub struct MetalDevice {
    id: DeviceId,
    name: String,
    capabilities: DeviceCapabilities,
}

impl Hash for MetalDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for MetalDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MetalDevice {}

impl MetalDevice {
    /// Create a Metal device (placeholder for actual Metal device).
    pub fn new(index: u32) -> Self {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.compute_version = ComputeVersion::Metal {
            family: MetalFamily::Apple8,
        };
        capabilities.vendor = "Apple".to_string();
        capabilities.total_memory = sys_info::mem_info()
            .map(|m| m.total * 1024)
            .unwrap_or(8 * 1024 * 1024 * 1024);
        capabilities.compute_units = 8;
        capabilities.features = vec![
            DeviceFeature::Float16,
            DeviceFeature::SharedMemory,
            DeviceFeature::UnifiedMemory,
        ];
        capabilities.supported_precisions = vec![
            Precision::Float16,
            Precision::Float32,
        ];
        capabilities.is_integrated = true;

        Self {
            id: DeviceId::metal(index),
            name: "Apple GPU".to_string(),
            capabilities,
        }
    }
}

impl ComputeDevice for MetalDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::MetalGpu
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

/// Metal buffer wrapper.
pub struct MetalBuffer {
    data: Vec<u8>,
    size: usize,
    usage: BufferUsage,
    device_id: DeviceId,
    is_mapped: AtomicBool,
}

impl Buffer for MetalBuffer {
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
        self.map_range(0, self.size)
    }

    fn map_range(&self, offset: usize, size: usize) -> ComputeResult<BufferMapping<'_>> {
        if offset + size > self.data.len() {
            return Err(ComputeError::MappingFailed("Range out of bounds".to_string()));
        }

        if self.is_mapped.swap(true, Ordering::SeqCst) {
            return Err(ComputeError::MappingFailed("Buffer already mapped".to_string()));
        }

        let ptr = self.data.as_ptr().wrapping_add(offset) as *mut u8;
        let is_mapped = &self.is_mapped;

        Ok(unsafe {
            BufferMapping::new(
                ptr,
                size,
                offset,
                Some(Box::new(move || {
                    is_mapped.store(false, Ordering::SeqCst);
                })),
            )
        })
    }

    fn is_mapped(&self) -> bool {
        self.is_mapped.load(Ordering::SeqCst)
    }

    fn raw_ptr(&self) -> Option<*mut u8> {
        Some(self.data.as_ptr() as *mut u8)
    }
}

/// Metal command encoder.
pub struct MetalCommandEncoder {
    commands: Vec<MetalCommand>,
}

enum MetalCommand {
    CopyH2D { src: Vec<u8>, offset: usize },
    CopyD2H { offset: usize, size: usize },
    Fill { offset: usize, size: usize, value: u8 },
}

impl MetalCommandEncoder {
    fn new() -> Self {
        Self { commands: Vec::new() }
    }
}

impl CommandEncoder for MetalCommandEncoder {
    type Buffer = MetalBuffer;

    fn copy_host_to_device(&mut self, src: &[u8], _dst: &Self::Buffer, offset: usize) {
        self.commands.push(MetalCommand::CopyH2D {
            src: src.to_vec(),
            offset,
        });
    }

    fn copy_device_to_host(&mut self, _src: &Self::Buffer, _dst: &mut [u8], offset: usize) {
        self.commands.push(MetalCommand::CopyD2H {
            offset,
            size: 0,
        });
    }

    fn copy_buffer_to_buffer(
        &mut self,
        _src: &Self::Buffer,
        _src_offset: usize,
        _dst: &Self::Buffer,
        _dst_offset: usize,
        _size: usize,
    ) {
        // Stub
    }

    fn fill_buffer(&mut self, _buffer: &Self::Buffer, offset: usize, size: usize, value: u8) {
        self.commands.push(MetalCommand::Fill { offset, size, value });
    }

    fn barrier(&mut self) {
        // Metal handles barriers automatically
    }
}

/// Metal fence.
pub struct MetalFence {
    completed: Arc<AtomicBool>,
}

impl MetalFence {
    fn new_signaled() -> Self {
        Self {
            completed: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl Fence for MetalFence {
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
        let timeout = std::time::Duration::from_millis(timeout_ms);

        while !self.is_signaled() {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

/// Metal compute backend.
pub struct MetalBackend {
    device: MetalDevice,
    rng_seed: AtomicU64,
}

impl ComputeBackend for MetalBackend {
    type Device = MetalDevice;
    type Buffer = MetalBuffer;
    type CommandEncoder = MetalCommandEncoder;
    type Fence = MetalFence;

    fn name() -> &'static str {
        "Metal"
    }

    fn enumerate_devices() -> ComputeResult<Vec<Self::Device>> {
        // Return a single Metal device on macOS
        Ok(vec![MetalDevice::new(0)])
    }

    fn new(device: &Self::Device) -> ComputeResult<Self> {
        Ok(Self {
            device: device.clone(),
            rng_seed: AtomicU64::new(0),
        })
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> ComputeResult<Self::Buffer> {
        Ok(MetalBuffer {
            data: vec![0u8; size],
            size,
            usage,
            device_id: self.device.device_id(),
            is_mapped: AtomicBool::new(false),
        })
    }

    fn create_encoder(&self) -> ComputeResult<Self::CommandEncoder> {
        Ok(MetalCommandEncoder::new())
    }

    fn submit(&self, _encoder: Self::CommandEncoder) -> ComputeResult<Self::Fence> {
        // Stub: immediately complete
        Ok(MetalFence::new_signaled())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_discovery() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        assert!(!devices.is_empty());
        println!("Found {} Metal device(s)", devices.len());
    }

    #[test]
    fn test_metal_backend() {
        let devices = MetalBackend::enumerate_devices().unwrap();
        let backend = MetalBackend::new(&devices[0]).unwrap();

        let buffer = backend.allocate_buffer(1024, BufferUsage::HOST_VISIBLE).unwrap();
        assert_eq!(buffer.size(), 1024);
    }
}
