# tsai_compute Benchmark Report

This report summarizes the performance improvements provided by tsai_compute's heterogeneous compute abstraction layer.

## System Information

- **Platform**: macOS (Darwin)
- **Architecture**: AArch64 (Apple Silicon)
- **SIMD Level**: NEON (mandatory on AArch64)

## Key Improvements from tsai_compute

### 1. SIMD-Accelerated Operations

The `tsai_compute` crate provides portable SIMD operations via the `wide` crate, with runtime dispatch to the optimal implementation.

#### Dot Product Performance (Critical for ML)

| Array Size | SIMD Throughput | Scalar Throughput | Speedup |
|------------|-----------------|-------------------|---------|
| 1,024      | 1.87 Gelem/s    | 1.17 Gelem/s      | **1.6x** |
| 4,096      | 4.77 Gelem/s    | 0.77 Gelem/s      | **6.2x** |
| 16,384     | 4.29 Gelem/s    | 0.81 Gelem/s      | **5.3x** |
| 65,536     | 4.08 Gelem/s    | 0.74 Gelem/s      | **5.5x** |
| 262,144    | 3.45 Gelem/s    | 0.77 Gelem/s      | **4.5x** |

#### Sum Reduction Performance

| Array Size | SIMD Throughput | Scalar Throughput | Speedup |
|------------|-----------------|-------------------|---------|
| 1,024      | 1.89 Gelem/s    | 1.34 Gelem/s      | **1.4x** |
| 4,096      | 5.72 Gelem/s    | 0.98 Gelem/s      | **5.8x** |
| 16,384     | 4.77 Gelem/s    | 0.91 Gelem/s      | **5.2x** |
| 65,536     | 4.77 Gelem/s    | 0.85 Gelem/s      | **5.6x** |
| 262,144    | 3.92 Gelem/s    | 0.79 Gelem/s      | **5.0x** |

#### Element-wise Addition Performance

| Array Size | SIMD Throughput | Scalar Throughput | Speedup |
|------------|-----------------|-------------------|---------|
| 1,024      | 4.04 Gelem/s    | 2.12 Gelem/s      | **1.9x** |
| 4,096      | 7.09 Gelem/s    | 2.35 Gelem/s      | **3.0x** |
| 16,384     | 8.68 Gelem/s    | 2.37 Gelem/s      | **3.7x** |
| 65,536     | 8.51 Gelem/s    | 2.23 Gelem/s      | **3.8x** |
| 262,144    | 6.78 Gelem/s    | 2.16 Gelem/s      | **3.1x** |

#### Element-wise Multiplication Performance

| Array Size | SIMD Throughput | Scalar Throughput | Speedup |
|------------|-----------------|-------------------|---------|
| 1,024      | 3.52 Gelem/s    | 1.97 Gelem/s      | **1.8x** |
| 4,096      | 6.87 Gelem/s    | 2.44 Gelem/s      | **2.8x** |
| 16,384     | 8.88 Gelem/s    | 2.45 Gelem/s      | **3.6x** |
| 65,536     | 8.51 Gelem/s    | 2.22 Gelem/s      | **3.8x** |
| 262,144    | 6.63 Gelem/s    | 2.15 Gelem/s      | **3.1x** |

### 2. Fast Hardware Discovery

| Operation            | Time      | Notes                           |
|---------------------|-----------|----------------------------------|
| Full Discovery      | ~50 µs    | CPU + GPU enumeration            |
| CPU-only Discovery  | ~500 ns   | Just CPU backend                 |

### 3. Fast Device Selection

| Scheduler Type       | Selection Time | Use Case                        |
|---------------------|----------------|----------------------------------|
| Best Device         | ~40 ns         | Get the optimal device           |
| Simple Scheduler    | ~34 ns         | Always picks best device         |
| Workload Scheduler  | ~40 ns         | Considers FLOPS/memory needs     |
| Round-Robin         | ~30 ns         | Load balancing across devices    |

### 4. High-Performance Memory Operations

| Buffer Size | Write Throughput | Read Throughput |
|-------------|------------------|-----------------|
| 1 KB        | 6.0 GiB/s        | 1.9 GiB/s       |
| 16 KB       | 23.7 GiB/s       | 7.7 GiB/s       |
| 256 KB      | 29.2 GiB/s       | 8.1 GiB/s       |

Buffer allocation is very fast (~30 ns for 1KB, ~380 ns for 1MB).

## Summary of Improvements

1. **SIMD Operations**: 3-6x faster than scalar for common ML operations (dot product, sum, element-wise ops)

2. **Device Discovery**: Sub-microsecond CPU discovery, ~50µs full system discovery

3. **Scheduling Overhead**: <50ns to select optimal device for workload

4. **Memory Throughput**: Up to 29 GiB/s write, 8 GiB/s read for mapped buffers

## Supported Backends

| Backend | Status | Platform |
|---------|--------|----------|
| CPU (SIMD) | ✅ Full | All platforms |
| Metal | ✅ Ready | macOS |
| CUDA | ✅ Ready | Linux/Windows + NVIDIA |
| Vulkan | 🔧 Stub | Cross-platform |
| OpenCL | 🔧 Stub | Cross-platform |
| ROCm | 🔧 Stub | Linux + AMD |

## Running Benchmarks

```bash
# Run SIMD operation benchmarks
cargo bench -p tsai_compute --bench simd_ops

# Run device/memory benchmarks
cargo bench -p tsai_compute --bench device_ops

# View HTML reports
open target/criterion/report/index.html
```

## Benchmark Environment

- Criterion 0.5 with 100 samples per benchmark
- Release build with LTO enabled
- Warm-up period of 3 seconds
