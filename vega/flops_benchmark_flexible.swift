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
            print("  Failed to load Metal library from \(libraryPath)")
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

    func runBenchmark(precision: String, pipeline: MTLComputePipelineState, elementSize: Int, numThreads: Int, iterations: UInt32, numRuns: Int = 3) -> (tflops: Double, time: Double)? {
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

        print("  GPU \(deviceIndex): Warming up...")

        // Warm-up runs
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

        print("  GPU \(deviceIndex): Running \(precision) benchmark (\(numRuns) runs)...")

        // Actual timed runs
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

enum RunMode {
    case single(Int)      // Run on a specific device index
    case sequential       // Run on all devices one at a time
    case parallel         // Run on all devices simultaneously
}

func printUsage() {
    print("""
    Usage: flops_benchmark [options]

    Options:
      --device <index>    Run benchmark on specific device (0, 1, etc.)
      --sequential        Run on all devices sequentially
      --parallel          Run on all devices in parallel (default)
      --list              List available devices and exit
      --help              Show this help message

    Examples:
      flops_benchmark --device 0       # Run on GPU 0 only
      flops_benchmark --device 1       # Run on GPU 1 only
      flops_benchmark --sequential     # Run on all GPUs one at a time
      flops_benchmark --parallel       # Run on all GPUs simultaneously
    """)
}

// Parse command line arguments
func parseArguments() -> RunMode? {
    let args = CommandLine.arguments

    if args.contains("--help") {
        printUsage()
        exit(0)
    }

    if args.contains("--list") {
        return nil  // Special case handled in main
    }

    if let deviceIndex = args.firstIndex(of: "--device") {
        guard deviceIndex + 1 < args.count,
              let deviceNum = Int(args[deviceIndex + 1]) else {
            print("Error: --device requires a device index number")
            printUsage()
            exit(1)
        }
        return .single(deviceNum)
    }

    if args.contains("--sequential") {
        return .sequential
    }

    // Default to parallel
    return .parallel
}

// Run benchmarks on selected devices in parallel
func runParallel(benchmarks: [FLOPSBenchmark], numThreads: Int, iterations: UInt32) -> (fp32: [(tflops: Double, time: Double)], fp16: [(tflops: Double, time: Double)]) {
    print("\n=== Running in PARALLEL mode ===\n")

    func runParallelBenchmarks(precision: String, getPipeline: @escaping (FLOPSBenchmark) -> MTLComputePipelineState, elementSize: Int) -> [(tflops: Double, time: Double)] {
        print("--- \(precision) Benchmark ---\n")

        let group = DispatchGroup()
        let lock = NSLock()
        var results: [(tflops: Double, time: Double)?] = Array(repeating: nil, count: benchmarks.count)

        let overallStart = CACurrentMediaTime()

        for (index, benchmark) in benchmarks.enumerated() {
            group.enter()
            DispatchQueue.global(qos: .userInteractive).async {
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

        print("\n  Total parallel execution time: \(String(format: "%.3f", overallEnd - overallStart))s\n")

        return results.compactMap { $0 }
    }

    let fp32Results = runParallelBenchmarks(precision: "FP32", getPipeline: { $0.fp32Pipeline }, elementSize: MemoryLayout<Float>.size)
    let fp16Results = runParallelBenchmarks(precision: "FP16", getPipeline: { $0.fp16Pipeline }, elementSize: 2)

    return (fp32Results, fp16Results)
}

// Run benchmarks on selected devices sequentially
func runSequential(benchmarks: [FLOPSBenchmark], numThreads: Int, iterations: UInt32) -> (fp32: [(tflops: Double, time: Double)], fp16: [(tflops: Double, time: Double)]) {
    print("\n=== Running in SEQUENTIAL mode ===\n")

    var fp32Results: [(tflops: Double, time: Double)] = []
    var fp16Results: [(tflops: Double, time: Double)] = []

    for (index, benchmark) in benchmarks.enumerated() {
        print("--- GPU \(index): \(benchmark.device.name) ---\n")

        print("Running FP32...")
        if let result = benchmark.runBenchmark(
            precision: "FP32",
            pipeline: benchmark.fp32Pipeline,
            elementSize: MemoryLayout<Float>.size,
            numThreads: numThreads,
            iterations: iterations
        ) {
            fp32Results.append(result)
        }

        print("\nRunning FP16...")
        if let result = benchmark.runBenchmark(
            precision: "FP16",
            pipeline: benchmark.fp16Pipeline,
            elementSize: 2,
            numThreads: numThreads,
            iterations: iterations
        ) {
            fp16Results.append(result)
        }

        print()
    }

    return (fp32Results, fp16Results)
}

// Print summary
func printSummary(benchmarks: [FLOPSBenchmark], fp32Results: [(tflops: Double, time: Double)], fp16Results: [(tflops: Double, time: Double)]) {
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

    if benchmarks.count > 1 {
        print("\nCombined Performance:")
        let totalFP32 = fp32Results.map { $0.tflops }.reduce(0, +)
        let totalFP16 = fp16Results.map { $0.tflops }.reduce(0, +)
        print("  FP32: \(String(format: "%.3f", totalFP32)) TFLOPS")
        print("  FP16: \(String(format: "%.3f", totalFP16)) TFLOPS")

        print("\nScaling Efficiency:")
        print("  Number of GPUs: \(benchmarks.count)")
        print("  FP32 per GPU avg: \(String(format: "%.3f", totalFP32/Double(benchmarks.count))) TFLOPS")
        print("  FP16 per GPU avg: \(String(format: "%.3f", totalFP16/Double(benchmarks.count))) TFLOPS")
    }
}

// Main execution
print("=== Metal FLOPS Benchmark ===\n")

let devices = MTLCopyAllDevices()

if devices.isEmpty {
    print("No Metal devices found!")
    exit(1)
}

// Handle --list option
if CommandLine.arguments.contains("--list") {
    print("Found \(devices.count) Metal device(s):\n")
    for (index, device) in devices.enumerated() {
        print("GPU \(index): \(device.name)")
        print("  Registry ID: \(device.registryID)")
        print("  Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
        print()
    }
    exit(0)
}

let mode = parseArguments() ?? .parallel

print("Found \(devices.count) Metal device(s)\n")

// Configuration
let libraryPath = "build/flops_benchmark_heavy.metallib"
let numThreads = 262144
let iterations: UInt32 = 2000000

// Create benchmarks based on mode
var benchmarks: [FLOPSBenchmark] = []

switch mode {
case .single(let deviceIndex):
    print("Running on single device: GPU \(deviceIndex)\n")
    guard deviceIndex < devices.count else {
        print("Error: Device index \(deviceIndex) out of range (0-\(devices.count - 1))")
        exit(1)
    }
    if let benchmark = FLOPSBenchmark(device: devices[deviceIndex], index: deviceIndex, libraryPath: libraryPath) {
        benchmarks.append(benchmark)
    }

case .sequential, .parallel:
    print("Running on all \(devices.count) device(s)\n")
    for (index, device) in devices.enumerated() {
        if let benchmark = FLOPSBenchmark(device: device, index: index, libraryPath: libraryPath) {
            benchmarks.append(benchmark)
        }
    }
}

if benchmarks.isEmpty {
    print("Failed to initialize any benchmarks")
    exit(1)
}

print("\nConfiguration:")
print("  Threads: \(numThreads)")
print("  Iterations: \(iterations)")
print("  FLOPs per iteration: 64")
let totalFlopsPerRun = Double(numThreads) * Double(iterations) * 64.0
print("  Total FLOPs per run: \(String(format: "%.2e", totalFlopsPerRun))")

// Run benchmarks based on mode
let (fp32Results, fp16Results): ([(tflops: Double, time: Double)], [(tflops: Double, time: Double)])

switch mode {
case .parallel:
    (fp32Results, fp16Results) = runParallel(benchmarks: benchmarks, numThreads: numThreads, iterations: iterations)
case .sequential, .single:
    (fp32Results, fp16Results) = runSequential(benchmarks: benchmarks, numThreads: numThreads, iterations: iterations)
}

// Print summary
printSummary(benchmarks: benchmarks, fp32Results: fp32Results, fp16Results: fp16Results)
