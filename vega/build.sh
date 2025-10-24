#!/bin/bash

set -e

echo "Building Metal FLOPS Benchmark..."

# Compile Metal shader
echo "Compiling Metal shader..."
xcrun -sdk macosx metal -c flops_benchmark.metal -o flops_benchmark.air
xcrun -sdk macosx metallib flops_benchmark.air -o flops_benchmark.metallib

# Compile Swift program
echo "Compiling Swift program..."
swiftc -o flops_benchmark flops_benchmark.swift -framework Metal -framework Foundation

echo "Build complete!"
echo ""
echo "Run with: ./flops_benchmark"
