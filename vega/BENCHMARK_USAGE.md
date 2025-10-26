# Metal FLOPS Benchmark Usage Guide

## Benchmark

`build/flops_benchmark` - Flexible benchmark with device selection and execution modes.

**Usage:**
```bash
# List available devices
./build/flops_benchmark --list

# Run on specific device
./build/flops_benchmark --device 0
./build/flops_benchmark --device 1

# Run on all devices sequentially (one at a time)
./build/flops_benchmark --sequential

# Run on all devices in parallel (default)
./build/flops_benchmark --parallel
./build/flops_benchmark  # Same as --parallel
```

## Use Cases

### Individual GPU Performance
To test each GPU independently and compare their performance:
```bash
./build/flops_benchmark --device 0
./build/flops_benchmark --device 1
```

### Combined Performance - Sequential
To see total performance when running workloads one GPU at a time:
```bash
./build/flops_benchmark --sequential
```

### Combined Performance - Parallel
To see total performance when both GPUs work simultaneously:
```bash
./build/flops_benchmark --parallel
```

## Example Workflow

```bash
# 1. Build the benchmark
./build.sh

# 2. List available devices
./build/flops_benchmark --list

# 3. Test GPU 0
./build/flops_benchmark --device 0

# 4. Test GPU 1
./build/flops_benchmark --device 1

# 5. Test both GPUs in parallel
./build/flops_benchmark --parallel

# 6. Test both GPUs sequentially
./build/flops_benchmark --sequential
```

## Interpreting Results

- **TFLOPS**: Tera floating-point operations per second (higher is better)
- **FP32**: Single-precision (32-bit) floating point
- **FP16**: Half-precision (16-bit) floating point
- **FP16/FP32 Ratio**: Speedup factor (Vega II should be ~2x in theory)
- **Combined Performance**: Sum of all GPU performances
- **Scaling Efficiency**: How well multiple GPUs work together
