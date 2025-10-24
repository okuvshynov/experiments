import Metal
import Foundation
import QuartzCore

class FLOPSBenchmark {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let fp32Pipeline: MTLComputePipelineState
    let fp16Pipeline: MTLComputePipelineState
    
    init?() {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return nil
        }
        self.device = device
        
        print("GPU: \(device.name)")
        print("Max threadgroup size: \(device.maxThreadsPerThreadgroup)")
        print("Max threads per grid: \(device.maxThreadsPerThreadgroup)")
        
        // Create command queue
        guard let queue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = queue
        
        // Load and compile Metal shader
        let libraryURL = URL(fileURLWithPath: "flops_benchmark.metallib")
        guard let library = try? device.makeLibrary(URL: libraryURL) else {
            print("Failed to load Metal library")
            return nil
        }
        
        // Create compute pipelines
        guard let fp32Function = library.makeFunction(name: "benchmark_fp32"),
              let fp16Function = library.makeFunction(name: "benchmark_fp16") else {
            print("Failed to load shader functions")
            return nil
        }
        
        guard let fp32 = try? device.makeComputePipelineState(function: fp32Function),
              let fp16 = try? device.makeComputePipelineState(function: fp16Function) else {
            print("Failed to create compute pipeline states")
            return nil
        }
        
        self.fp32Pipeline = fp32
        self.fp16Pipeline = fp16
    }
    
    func runBenchmark(precision: String, pipeline: MTLComputePipelineState, elementSize: Int) -> Double? {
        let numThreads = 65536  // Total number of threads
        let iterations: UInt32 = 1000000  // Iterations per thread (increased 10x)
        let flopsPerIteration = 16  // 8 FMA operations × 2 FLOPs each
        
        // Create output buffer
        guard let outputBuffer = device.makeBuffer(length: numThreads * elementSize, options: .storageModeShared) else {
            print("Failed to create output buffer")
            return nil
        }
        
        // Create iterations buffer
        var iterCount = iterations
        guard let iterBuffer = device.makeBuffer(bytes: &iterCount, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("Failed to create iterations buffer")
            return nil
        }
        
        // Calculate thread group configuration
        let threadGroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (numThreads + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
        
        // Warm-up runs (3 runs to ensure GPU is fully warmed up)
        print("  Warming up...")
        for _ in 0..<3 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(iterBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        print("  Running benchmark...")
        
        // Actual timed run - multiple iterations for better accuracy
        let numRuns = 10
        var totalTime = 0.0
        var minTime = Double.infinity
        var maxTime = 0.0
        var times: [Double] = []
        
        for run in 0..<numRuns {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(iterBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            
            let startTime = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let endTime = CACurrentMediaTime()
            
            let runTime = Double(endTime - startTime)
            times.append(runTime)
            totalTime += runTime
            minTime = min(minTime, runTime)
            maxTime = max(maxTime, runTime)
        }
        
        let avgTime = totalTime / Double(numRuns)
        
        // Calculate standard deviation
        let variance = times.map { pow($0 - avgTime, 2) }.reduce(0, +) / Double(numRuns)
        let stdDev = sqrt(variance)
        
        let totalFLOPs = Double(numThreads) * Double(iterations) * Double(flopsPerIteration)
        let avgTflops = (totalFLOPs / avgTime) / 1e12
        let minTflops = (totalFLOPs / maxTime) / 1e12  // Note: min time = max TFLOPS
        let maxTflops = (totalFLOPs / minTime) / 1e12  // Note: max time = min TFLOPS
        
        print("\n\(precision) Performance:")
        print("  Threads: \(numThreads)")
        print("  Iterations per thread: \(iterations)")
        print("  Total FLOPs per run: \(String(format: "%.2e", totalFLOPs))")
        print("  Time - Avg: \(String(format: "%.6f", avgTime))s, Min: \(String(format: "%.6f", minTime))s, Max: \(String(format: "%.6f", maxTime))s, StdDev: \(String(format: "%.6f", stdDev))s")
        print("  Performance - Avg: \(String(format: "%.3f", avgTflops)) TFLOPS, Range: [\(String(format: "%.3f", minTflops)) - \(String(format: "%.3f", maxTflops))] TFLOPS")
        
        return avgTflops
    }
}

// Main execution
print("=== Metal FLOPS Benchmark ===\n")

guard let benchmark = FLOPSBenchmark() else {
    print("Failed to initialize benchmark")
    exit(1)
}

// Run FP32 benchmark
let fp32TFLOPS = benchmark.runBenchmark(precision: "FP32", pipeline: benchmark.fp32Pipeline, elementSize: MemoryLayout<Float>.size)

// Run FP16 benchmark (half precision is 2 bytes)
let fp16TFLOPS = benchmark.runBenchmark(precision: "FP16", pipeline: benchmark.fp16Pipeline, elementSize: 2)

if let fp32 = fp32TFLOPS, let fp16 = fp16TFLOPS {
    print("\n=== Summary ===")
    print("FP32: \(String(format: "%.3f", fp32)) TFLOPS")
    print("FP16: \(String(format: "%.3f", fp16)) TFLOPS")
    print("FP16/FP32 Ratio: \(String(format: "%.2f", fp16/fp32))x")
    
    if fp16/fp32 < 1.5 {
        print("\n⚠️  Note: FP16 performance is lower than expected.")
        print("   Expected ~2x speedup for Vega II, but got \(String(format: "%.2f", fp16/fp32))x")
        print("   Possible causes:")
        print("   - Metal compiler may not be using packed FP16 instructions")
        print("   - Driver/macOS version may affect FP16 optimization")
        print("   - Vega architecture's FP16 support may require specific code patterns")
    }
}
