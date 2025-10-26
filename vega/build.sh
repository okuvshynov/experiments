#!/bin/bash

set -e

echo "Building Metal FLOPS Benchmarks..."

# Create build directory
mkdir -p build

# Compile Metal shaders
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c flops_benchmark.metal -o build/flops_benchmark.air
xcrun -sdk macosx metallib build/flops_benchmark.air -o build/flops_benchmark.metallib

xcrun -sdk macosx metal -c flops_benchmark_heavy.metal -o build/flops_benchmark_heavy.air
xcrun -sdk macosx metallib build/flops_benchmark_heavy.air -o build/flops_benchmark_heavy.metallib

xcrun -sdk macosx metal -c memory_bandwidth.metal -o build/memory_bandwidth.air
xcrun -sdk macosx metallib build/memory_bandwidth.air -o build/memory_bandwidth.metallib

# Compile Swift programs
echo "Compiling Swift programs..."
swiftc -O -o build/flops_benchmark flops_benchmark_flexible.swift -framework Metal -framework Foundation -framework QuartzCore
swiftc -O -o build/memory_bandwidth memory_bandwidth_benchmark.swift -framework Metal -framework Foundation -framework QuartzCore

echo "Build complete!"
echo ""
echo "Run FLOPS benchmark:     build/flops_benchmark [--device N | --sequential | --parallel | --list]"
echo "Run bandwidth benchmark: build/memory_bandwidth [--device N | --list]"
