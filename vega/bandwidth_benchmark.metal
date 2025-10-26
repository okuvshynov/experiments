#include <metal_stdlib>
using namespace metal;

// Simple copy kernel - pure read/write bandwidth test
kernel void bandwidth_copy(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& array_length [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid < array_length) {
        output[gid] = input[gid];
    }
}

// Vector copy using float4 - tests vectorized memory access
kernel void bandwidth_copy_vec4(device const float4* input [[buffer(0)]],
                                device float4* output [[buffer(1)]],
                                constant uint& array_length [[buffer(2)]],
                                uint gid [[thread_position_in_grid]])
{
    if (gid < array_length) {
        output[gid] = input[gid];
    }
}

// Read-only bandwidth test - tests read bandwidth
kernel void bandwidth_read(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& array_length [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid < array_length) {
        // Read multiple values to maximize read bandwidth
        float sum = 0.0f;
        uint base = gid * 8;
        for (uint i = 0; i < 8 && (base + i) < array_length; i++) {
            sum += input[base + i];
        }
        output[gid] = sum;
    }
}

// Write-only bandwidth test - tests write bandwidth
kernel void bandwidth_write(device float* output [[buffer(0)]],
                           constant uint& array_length [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
    if (gid < array_length) {
        // Write multiple values to maximize write bandwidth
        uint base = gid * 8;
        float value = float(gid);
        for (uint i = 0; i < 8 && (base + i) < array_length; i++) {
            output[base + i] = value + float(i);
        }
    }
}

// Strided access pattern - tests cache and memory access patterns
kernel void bandwidth_strided(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& array_length [[buffer(2)]],
                             constant uint& stride [[buffer(3)]],
                             uint gid [[thread_position_in_grid]])
{
    uint index = gid * stride;
    if (index < array_length) {
        output[gid] = input[index];
    }
}

// Random access pattern - tests worst-case memory access
kernel void bandwidth_random(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& array_length [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    // Simple pseudo-random number generator
    uint hash = gid;
    hash = hash * 747796405u + 2891336453u;
    hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
    hash = (hash >> 22u) ^ hash;

    uint index = hash % array_length;
    output[gid] = input[index];
}
