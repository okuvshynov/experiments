#!/bin/bash

set -e

echo "Building Metal GPU Benchmarks..."

# Create build directory
mkdir -p build

# Compile Metal shaders
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c compute_benchmark.metal -o build/compute_benchmark.air
xcrun -sdk macosx metallib build/compute_benchmark.air -o build/compute_benchmark.metallib

xcrun -sdk macosx metal -c bandwidth_benchmark.metal -o build/bandwidth_benchmark.air
xcrun -sdk macosx metallib build/bandwidth_benchmark.air -o build/bandwidth_benchmark.metallib

# Compile Swift programs
echo "Compiling Swift programs..."
swiftc -O -o build/compute_benchmark compute_benchmark.swift -framework Metal -framework Foundation -framework QuartzCore
swiftc -O -o build/bandwidth_benchmark bandwidth_benchmark.swift -framework Metal -framework Foundation -framework QuartzCore

echo "Build complete!"
echo ""
echo "Run compute benchmark:   build/compute_benchmark [--device N | --sequential | --parallel | --list]"
echo "Run bandwidth benchmark: build/bandwidth_benchmark [--device N | --list]"
