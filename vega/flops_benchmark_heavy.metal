#include <metal_stdlib>
using namespace metal;

// FP32 heavy benchmark kernel - more accumulators and operations
kernel void benchmark_fp32_heavy(device float* output [[buffer(0)]],
                                 constant uint& iterations [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]])
{
    // Initialize 8 accumulators for more instruction-level parallelism
    float acc0 = 1.0f + float(gid) * 0.001f;
    float acc1 = 1.0f + float(gid) * 0.002f;
    float acc2 = 1.0f + float(gid) * 0.003f;
    float acc3 = 1.0f + float(gid) * 0.004f;
    float acc4 = 1.0f + float(gid) * 0.005f;
    float acc5 = 1.0f + float(gid) * 0.006f;
    float acc6 = 1.0f + float(gid) * 0.007f;
    float acc7 = 1.0f + float(gid) * 0.008f;

    float mult0 = 1.000001f;
    float mult1 = 1.000002f;
    float mult2 = 1.000003f;
    float mult3 = 1.000004f;
    float mult4 = 1.000005f;
    float mult5 = 1.000006f;
    float mult6 = 1.000007f;
    float mult7 = 1.000008f;

    // Perform many FMA operations
    // Each FMA is 2 FLOPs (multiply + add)
    // 32 FMAs per iteration = 64 FLOPs per iteration
    for (uint i = 0; i < iterations; i++) {
        // First round - 8 FMAs
        acc0 = fma(acc0, mult0, acc1);
        acc1 = fma(acc1, mult1, acc2);
        acc2 = fma(acc2, mult2, acc3);
        acc3 = fma(acc3, mult3, acc4);
        acc4 = fma(acc4, mult4, acc5);
        acc5 = fma(acc5, mult5, acc6);
        acc6 = fma(acc6, mult6, acc7);
        acc7 = fma(acc7, mult7, acc0);

        // Second round - 8 FMAs
        acc0 = fma(acc0, mult1, acc2);
        acc1 = fma(acc1, mult2, acc3);
        acc2 = fma(acc2, mult3, acc4);
        acc3 = fma(acc3, mult4, acc5);
        acc4 = fma(acc4, mult5, acc6);
        acc5 = fma(acc5, mult6, acc7);
        acc6 = fma(acc6, mult7, acc0);
        acc7 = fma(acc7, mult0, acc1);

        // Third round - 8 FMAs
        acc0 = fma(acc0, mult2, acc3);
        acc1 = fma(acc1, mult3, acc4);
        acc2 = fma(acc2, mult4, acc5);
        acc3 = fma(acc3, mult5, acc6);
        acc4 = fma(acc4, mult6, acc7);
        acc5 = fma(acc5, mult7, acc0);
        acc6 = fma(acc6, mult0, acc1);
        acc7 = fma(acc7, mult1, acc2);

        // Fourth round - 8 FMAs
        acc0 = fma(acc0, mult3, acc4);
        acc1 = fma(acc1, mult4, acc5);
        acc2 = fma(acc2, mult5, acc6);
        acc3 = fma(acc3, mult6, acc7);
        acc4 = fma(acc4, mult7, acc0);
        acc5 = fma(acc5, mult0, acc1);
        acc6 = fma(acc6, mult1, acc2);
        acc7 = fma(acc7, mult2, acc3);
    }

    // Write result to prevent optimization
    output[gid] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

// FP16 heavy benchmark kernel - same logic but using half precision
kernel void benchmark_fp16_heavy(device half* output [[buffer(0)]],
                                 constant uint& iterations [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]])
{
    half acc0 = half(1.0f + float(gid) * 0.001f);
    half acc1 = half(1.0f + float(gid) * 0.002f);
    half acc2 = half(1.0f + float(gid) * 0.003f);
    half acc3 = half(1.0f + float(gid) * 0.004f);
    half acc4 = half(1.0f + float(gid) * 0.005f);
    half acc5 = half(1.0f + float(gid) * 0.006f);
    half acc6 = half(1.0f + float(gid) * 0.007f);
    half acc7 = half(1.0f + float(gid) * 0.008f);

    half mult0 = half(1.001f);
    half mult1 = half(1.002f);
    half mult2 = half(1.003f);
    half mult3 = half(1.004f);
    half mult4 = half(1.005f);
    half mult5 = half(1.006f);
    half mult6 = half(1.007f);
    half mult7 = half(1.008f);

    for (uint i = 0; i < iterations; i++) {
        acc0 = fma(acc0, mult0, acc1);
        acc1 = fma(acc1, mult1, acc2);
        acc2 = fma(acc2, mult2, acc3);
        acc3 = fma(acc3, mult3, acc4);
        acc4 = fma(acc4, mult4, acc5);
        acc5 = fma(acc5, mult5, acc6);
        acc6 = fma(acc6, mult6, acc7);
        acc7 = fma(acc7, mult7, acc0);

        acc0 = fma(acc0, mult1, acc2);
        acc1 = fma(acc1, mult2, acc3);
        acc2 = fma(acc2, mult3, acc4);
        acc3 = fma(acc3, mult4, acc5);
        acc4 = fma(acc4, mult5, acc6);
        acc5 = fma(acc5, mult6, acc7);
        acc6 = fma(acc6, mult7, acc0);
        acc7 = fma(acc7, mult0, acc1);

        acc0 = fma(acc0, mult2, acc3);
        acc1 = fma(acc1, mult3, acc4);
        acc2 = fma(acc2, mult4, acc5);
        acc3 = fma(acc3, mult5, acc6);
        acc4 = fma(acc4, mult6, acc7);
        acc5 = fma(acc5, mult7, acc0);
        acc6 = fma(acc6, mult0, acc1);
        acc7 = fma(acc7, mult1, acc2);

        acc0 = fma(acc0, mult3, acc4);
        acc1 = fma(acc1, mult4, acc5);
        acc2 = fma(acc2, mult5, acc6);
        acc3 = fma(acc3, mult6, acc7);
        acc4 = fma(acc4, mult7, acc0);
        acc5 = fma(acc5, mult0, acc1);
        acc6 = fma(acc6, mult1, acc2);
        acc7 = fma(acc7, mult2, acc3);
    }

    output[gid] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}
