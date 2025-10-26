# Metal GPU Benchmark Usage Guide

## Benchmarks

### Compute Benchmark
`build/compute_benchmark` - Tests GPU compute performance (FP32 TFLOPS)

**Usage:**
```bash
# List available devices
./build/compute_benchmark --list

# Run on specific device
./build/compute_benchmark --device 0
./build/compute_benchmark --device 1

# Run on all devices sequentially (one at a time)
./build/compute_benchmark --sequential

# Run on all devices in parallel (default)
./build/compute_benchmark --parallel
./build/compute_benchmark  # Same as --parallel
```

### Bandwidth Benchmark
`build/bandwidth_benchmark` - Tests HBM memory bandwidth (GB/s)

**Usage:**
```bash
# List available devices
./build/bandwidth_benchmark --list

# Run on specific device
./build/bandwidth_benchmark --device 0
./build/bandwidth_benchmark --device 1
```

## Use Cases

### Individual GPU Performance
Test each GPU independently:
```bash
# Compute performance
./build/compute_benchmark --device 0
./build/compute_benchmark --device 1

# Memory bandwidth
./build/bandwidth_benchmark --device 0
./build/bandwidth_benchmark --device 1
```

### Combined GPU Performance
Test all GPUs working together (compute only):
```bash
# Sequential - one at a time
./build/compute_benchmark --sequential

# Parallel - simultaneously
./build/compute_benchmark --parallel
```

## Example Workflow

```bash
# 1. Build the benchmarks
./build.sh

# 2. List available devices
./build/compute_benchmark --list

# 3. Test compute performance on GPU 0
./build/compute_benchmark --device 0

# 4. Test memory bandwidth on GPU 0
./build/bandwidth_benchmark --device 0

# 5. Test both GPUs in parallel (compute)
./build/compute_benchmark --parallel

# 6. Compare bandwidth across GPUs
./build/bandwidth_benchmark --device 0
./build/bandwidth_benchmark --device 1
```

## Interpreting Results

- **TFLOPS**: Tera floating-point operations per second (higher is better)
- **FP32**: Single-precision (32-bit) floating point
- **Average time**: Mean execution time across benchmark runs
- **Combined Performance**: Sum of all GPU performances (when running multiple GPUs)
- **Scaling Efficiency**: How well multiple GPUs work together

## About the Benchmarks

### Compute Benchmark
Tests FP32 (single-precision) floating-point performance using fused multiply-add (FMA) operations:
- 32 FMA operations per iteration = 64 FLOPs per iteration
- Each thread runs 2,000,000 iterations
- Uses 262,144 threads total
- Total workload: ~33.6 trillion FLOPs per run
- 10 runs with statistics (avg, min, max, stddev)

### Bandwidth Benchmark
Tests HBM2 memory bandwidth with different access patterns:
- Copy test: Read + Write bandwidth (both directions)
- Read-only test: Pure read bandwidth
- Write-only test: Pure write bandwidth
- Tests with 64MB, 256MB, and 1GB arrays
- Uses GPU-private memory (.storageModePrivate) for true HBM testing
- 5 runs per test
