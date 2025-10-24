#include <metal_stdlib>
using namespace metal;

// FP32 benchmark kernel - performs many FMA (fused multiply-add) operations
kernel void benchmark_fp32(device float* output [[buffer(0)]],
                           constant uint& iterations [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
    // Initialize multiple accumulators to maximize instruction-level parallelism
    float acc0 = 1.0f + float(gid) * 0.001f;
    float acc1 = 1.0f + float(gid) * 0.002f;
    float acc2 = 1.0f + float(gid) * 0.003f;
    float acc3 = 1.0f + float(gid) * 0.004f;
    
    float mult0 = 1.000001f;
    float mult1 = 1.000002f;
    float mult2 = 1.000003f;
    float mult3 = 1.000004f;
    
    // Perform many FMA operations
    // Each FMA is 2 FLOPs (multiply + add)
    for (uint i = 0; i < iterations; i++) {
        // Unroll to maximize FLOP count - 8 FMAs per iteration = 16 FLOPs
        acc0 = fma(acc0, mult0, acc1);
        acc1 = fma(acc1, mult1, acc2);
        acc2 = fma(acc2, mult2, acc3);
        acc3 = fma(acc3, mult3, acc0);
        
        acc0 = fma(acc0, mult1, acc2);
        acc1 = fma(acc1, mult2, acc3);
        acc2 = fma(acc2, mult3, acc0);
        acc3 = fma(acc3, mult0, acc1);
    }
    
    // Write result to prevent optimization
    output[gid] = acc0 + acc1 + acc2 + acc3;
}

// FP16 benchmark kernel - same logic but using half precision
kernel void benchmark_fp16(device half* output [[buffer(0)]],
                           constant uint& iterations [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
    half acc0 = half(1.0f + float(gid) * 0.001f);
    half acc1 = half(1.0f + float(gid) * 0.002f);
    half acc2 = half(1.0f + float(gid) * 0.003f);
    half acc3 = half(1.0f + float(gid) * 0.004f);
    
    half mult0 = half(1.001f);
    half mult1 = half(1.002f);
    half mult2 = half(1.003f);
    half mult3 = half(1.004f);
    
    for (uint i = 0; i < iterations; i++) {
        acc0 = fma(acc0, mult0, acc1);
        acc1 = fma(acc1, mult1, acc2);
        acc2 = fma(acc2, mult2, acc3);
        acc3 = fma(acc3, mult3, acc0);
        
        acc0 = fma(acc0, mult1, acc2);
        acc1 = fma(acc1, mult2, acc3);
        acc2 = fma(acc2, mult3, acc0);
        acc3 = fma(acc3, mult0, acc1);
    }
    
    output[gid] = acc0 + acc1 + acc2 + acc3;
}

