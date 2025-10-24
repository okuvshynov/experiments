# Metal FLOPS Benchmark for Mac Pro 2019 with Radeon Vega II

This benchmark measures the theoretical FLOPS (floating-point operations per second) of your GPU using Metal compute shaders.

## What it does

The benchmark runs compute kernels that perform intensive FMA (fused multiply-add) operations in both FP32 (32-bit float) and FP16 (16-bit half-precision) modes. Each FMA counts as 2 FLOPs (one multiply + one add).

The kernel uses:
- Multiple accumulators to maximize instruction-level parallelism
- FMA operations (fused multiply-add) for efficiency
- Many iterations to get stable timing measurements

## Expected Performance

Your **Radeon Vega II (32GB)** specifications:
- **FP32 Performance**: ~14.2 TFLOPS (theoretical peak)
- **FP16 Performance**: ~28.4 TFLOPS (theoretical peak, 2x FP32)
- Architecture: Vega 20
- Compute Units: 60
- Stream Processors: 3840

The benchmark won't hit theoretical peak (no real-world code does), but you should see:
- FP32: 10-13 TFLOPS (70-90% of peak)
- FP16: 20-26 TFLOPS (similar efficiency)
- Ratio: ~2x (FP16/FP32)

## Building and Running

### Quick Start

```bash
# Make the build script executable
chmod +x build.sh

# Build
./build.sh

# Run
./flops_benchmark
```

### Manual Build

If the build script doesn't work, you can compile manually:

```bash
# Compile Metal shader to AIR
xcrun -sdk macosx metal -c flops_benchmark.metal -o flops_benchmark.air

# Create Metal library
xcrun -sdk macosx metallib flops_benchmark.air -o flops_benchmark.metallib

# Compile Swift program
swiftc -o flops_benchmark flops_benchmark.swift -framework Metal -framework Foundation

# Run
./flops_benchmark
```

## Understanding the Results

The benchmark outputs:
- **Threads**: Number of parallel GPU threads
- **Iterations**: Number of compute loops per thread
- **Total FLOPs**: Total floating-point operations performed
- **Time**: Execution time (averaged over 5 runs)
- **TFLOPS**: Tera-FLOPS (trillions of FLOPs per second)

Lower results than theoretical peak are normal due to:
- Memory bandwidth limitations
- Instruction scheduling overhead
- Cache effects
- Driver overhead

## Customization

You can modify the benchmark parameters in `flops_benchmark.swift`:
- `numThreads`: Number of parallel threads (default: 65536)
- `iterations`: Number of iterations per thread (default: 100000)
- `numRuns`: Number of timed runs to average (default: 5)

For longer benchmarks (more stable results), increase `iterations` or `numRuns`.

## Troubleshooting

If you get "Metal is not supported":
- Ensure you're running on macOS with Metal support
- Check that your GPU is properly recognized

If build fails:
- Ensure Xcode command line tools are installed: `xcode-select --install`
- Check that you have a recent macOS version

## Technical Details

- Uses compute shaders for maximum parallelism
- FMA (fused multiply-add) operations: `result = a * b + c`
- 8 FMA operations per inner loop = 16 FLOPs per iteration
- 4 accumulators for instruction-level parallelism
- Prevents compiler optimization by writing results to buffer

## References

- Radeon Vega II specs: https://www.amd.com/en/products/specifications/graphics/11726
- Metal Performance Shaders: https://developer.apple.com/metal/
- FMA operations: https://en.wikipedia.org/wiki/Multiply–accumulate_operation
