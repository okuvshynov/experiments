import Metal
import Foundation
import QuartzCore

class FLOPSBenchmark {
    let device: MTLDevice
    let deviceIndex: Int
    let commandQueue: MTLCommandQueue
    let fp32Pipeline: MTLComputePipelineState
    let fp16Pipeline: MTLComputePipelineState

    init?(device: MTLDevice, index: Int, libraryPath: String) {
        self.device = device
        self.deviceIndex = index

        print("GPU \(index): \(device.name)")
        print("  Max threadgroup size: \(device.maxThreadsPerThreadgroup)")
        print("  Registry ID: \(device.registryID)")

        // Create command queue
        guard let queue = device.makeCommandQueue() else {
            print("  Failed to create command queue")
            return nil
        }
        self.commandQueue = queue

        // Load and compile Metal shader
        let libraryURL = URL(fileURLWithPath: libraryPath)
        guard let library = try? device.makeLibrary(URL: libraryURL) else {
            print("  Failed to load Metal library")
            return nil
        }

        // Create compute pipelines
        guard let fp32Function = library.makeFunction(name: "benchmark_fp32_heavy"),
              let fp16Function = library.makeFunction(name: "benchmark_fp16_heavy") else {
            print("  Failed to load shader functions")
            return nil
        }

        guard let fp32 = try? device.makeComputePipelineState(function: fp32Function),
              let fp16 = try? device.makeComputePipelineState(function: fp16Function) else {
            print("  Failed to create compute pipeline states")
            return nil
        }

        self.fp32Pipeline = fp32
        self.fp16Pipeline = fp16
    }

    func runBenchmark(precision: String, pipeline: MTLComputePipelineState, elementSize: Int, numThreads: Int, iterations: UInt32) -> (tflops: Double, time: Double)? {
        let flopsPerIteration = 64  // 32 FMA operations × 2 FLOPs each

        // Create output buffer
        guard let outputBuffer = device.makeBuffer(length: numThreads * elementSize, options: .storageModeShared) else {
            print("  Failed to create output buffer")
            return nil
        }

        // Create iterations buffer
        var iterCount = iterations
        guard let iterBuffer = device.makeBuffer(bytes: &iterCount, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            print("  Failed to create iterations buffer")
            return nil
        }

        // Calculate thread group configuration
        let threadGroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (numThreads + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)

        print("  GPU \(deviceIndex): Warming up (2 runs)...")

        // Shorter warm-up
        for _ in 0..<2 {
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

        print("  GPU \(deviceIndex): Running timed benchmark (3 runs)...")

        // Actual timed run
        let numRuns = 3
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
            print("    Run \(run + 1): \(String(format: "%.4f", runTime))s")
            times.append(runTime)
        }

        let avgTime = times.reduce(0, +) / Double(numRuns)
        let minTime = times.min()!
        let maxTime = times.max()!

        let totalFLOPs = Double(numThreads) * Double(iterations) * Double(flopsPerIteration)
        let avgTflops = (totalFLOPs / avgTime) / 1e12

        print("    Average time: \(String(format: "%.4f", avgTime))s")
        print("    Performance: \(String(format: "%.3f", avgTflops)) TFLOPS")

        return (avgTflops, avgTime)
    }
}

// Main execution
print("=== Metal Dual-GPU Heavy FLOPS Benchmark ===\n")

let devices = MTLCopyAllDevices()
print("Found \(devices.count) Metal device(s):\n")

if devices.isEmpty {
    print("No Metal devices found!")
    exit(1)
}

// Compile Metal shader
print("Compiling Metal kernels...")
let compileResult = Process()
compileResult.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
compileResult.arguments = ["-sdk", "macosx", "metal", "-c", "flops_benchmark_heavy.metal", "-o", "flops_benchmark_heavy.air"]
try? compileResult.run()
compileResult.waitUntilExit()

let linkResult = Process()
linkResult.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
linkResult.arguments = ["-sdk", "macosx", "metallib", "flops_benchmark_heavy.air", "-o", "flops_benchmark_heavy.metallib"]
try? linkResult.run()
linkResult.waitUntilExit()
print("Done.\n")

// Create benchmarks
var benchmarks: [FLOPSBenchmark] = []
for (index, device) in devices.enumerated() {
    if let benchmark = FLOPSBenchmark(device: device, index: index, libraryPath: "flops_benchmark_heavy.metallib") {
        benchmarks.append(benchmark)
    }
    print()
}

if benchmarks.isEmpty {
    print("Failed to initialize benchmarks")
    exit(1)
}

// Configuration - tuned for ~2-5 second runs per GPU
let numThreads = 262144
let iterations: UInt32 = 2000000

print("Configuration:")
print("  Threads: \(numThreads)")
print("  Iterations: \(iterations)")
print("  FLOPs per iteration: 64")
let totalFlopsPerRun = Double(numThreads) * Double(iterations) * 64.0
print("  Total FLOPs per run: \(String(format: "%.2e", totalFlopsPerRun))")
print()

// Run both GPUs in parallel using dispatch groups
func runParallelBenchmarks(precision: String, getPipeline: @escaping (FLOPSBenchmark) -> MTLComputePipelineState, elementSize: Int) -> [(tflops: Double, time: Double)] {
    print("=== \(precision) Benchmark (Parallel Execution) ===\n")

    let group = DispatchGroup()
    let lock = NSLock()
    var results: [(tflops: Double, time: Double)?] = Array(repeating: nil, count: benchmarks.count)

    let overallStart = CACurrentMediaTime()

    for (index, benchmark) in benchmarks.enumerated() {
        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            print("Starting GPU \(index) - \(precision)")
            let result = benchmark.runBenchmark(
                precision: precision,
                pipeline: getPipeline(benchmark),
                elementSize: elementSize,
                numThreads: numThreads,
                iterations: iterations
            )
            lock.lock()
            results[index] = result
            lock.unlock()
            group.leave()
        }
    }

    group.wait()
    let overallEnd = CACurrentMediaTime()

    print("\nTotal parallel execution time: \(String(format: "%.3f", overallEnd - overallStart))s\n")

    return results.compactMap { $0 }
}

// Run benchmarks
let fp32Results = runParallelBenchmarks(precision: "FP32", getPipeline: { $0.fp32Pipeline }, elementSize: MemoryLayout<Float>.size)
let fp16Results = runParallelBenchmarks(precision: "FP16", getPipeline: { $0.fp16Pipeline }, elementSize: 2)

// Print summary
print("=== Summary ===\n")
print("Per-GPU Performance:")
for (index, benchmark) in benchmarks.enumerated() {
    print("\nGPU \(index): \(benchmark.device.name)")
    if index < fp32Results.count {
        print("  FP32: \(String(format: "%.3f", fp32Results[index].tflops)) TFLOPS (avg time: \(String(format: "%.3f", fp32Results[index].time))s)")
    }
    if index < fp16Results.count {
        print("  FP16: \(String(format: "%.3f", fp16Results[index].tflops)) TFLOPS (avg time: \(String(format: "%.3f", fp16Results[index].time))s)")
    }
    if index < fp32Results.count && index < fp16Results.count {
        print("  FP16/FP32 Ratio: \(String(format: "%.2f", fp16Results[index].tflops/fp32Results[index].tflops))x")
    }
}

print("\nCombined Performance (Both GPUs):")
let totalFP32 = fp32Results.map { $0.tflops }.reduce(0, +)
let totalFP16 = fp16Results.map { $0.tflops }.reduce(0, +)
print("  FP32: \(String(format: "%.3f", totalFP32)) TFLOPS")
print("  FP16: \(String(format: "%.3f", totalFP16)) TFLOPS")

if benchmarks.count > 1 {
    print("\nScaling Efficiency:")
    print("  Number of GPUs: \(benchmarks.count)")
    print("  FP32 per GPU avg: \(String(format: "%.3f", totalFP32/Double(benchmarks.count))) TFLOPS")
    print("  FP16 per GPU avg: \(String(format: "%.3f", totalFP16/Double(benchmarks.count))) TFLOPS")
}
