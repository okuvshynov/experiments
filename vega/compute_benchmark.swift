import Metal
import Foundation
import QuartzCore

class FLOPSBenchmark {
    let device: MTLDevice
    let deviceIndex: Int
    let commandQueue: MTLCommandQueue
    let fp32Pipeline: MTLComputePipelineState

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

        // Create FP32 compute pipeline
        guard let fp32Function = library.makeFunction(name: "compute_fp32") else {
            print("  Failed to load FP32 shader function")
            return nil
        }

        guard let fp32 = try? device.makeComputePipelineState(function: fp32Function) else {
            print("  Failed to create FP32 compute pipeline state")
            return nil
        }

        self.fp32Pipeline = fp32
    }

    func runBenchmark(numThreads: Int, iterations: UInt32, numRuns: Int = 10) -> (tflops: Double, time: Double)? {
        let flopsPerIteration = 64  // 32 FMA operations × 2 FLOPs each

        // Create output buffer
        guard let outputBuffer = device.makeBuffer(length: numThreads * MemoryLayout<Float>.size, options: .storageModeShared) else {
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
        let threadGroupSize = MTLSize(width: min(fp32Pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (numThreads + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)

        print("  GPU \(deviceIndex): Warming up...")

        // Warm-up runs
        for _ in 0..<2 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(fp32Pipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(iterBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        print("  GPU \(deviceIndex): Running FP32 benchmark (\(numRuns) runs)...")

        // Actual timed runs
        var times: [Double] = []

        for run in 0..<numRuns {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(fp32Pipeline)
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

        // Calculate standard deviation
        let variance = times.map { pow($0 - avgTime, 2) }.reduce(0, +) / Double(numRuns)
        let stdDev = sqrt(variance)

        let totalFLOPs = Double(numThreads) * Double(iterations) * Double(flopsPerIteration)
        let avgTflops = (totalFLOPs / avgTime) / 1e12
        let minTflops = (totalFLOPs / maxTime) / 1e12  // min time = max TFLOPS
        let maxTflops = (totalFLOPs / minTime) / 1e12  // max time = min TFLOPS

        print("    Time   - Avg: \(String(format: "%.4f", avgTime))s, Min: \(String(format: "%.4f", minTime))s, Max: \(String(format: "%.4f", maxTime))s, StdDev: \(String(format: "%.4f", stdDev))s")
        print("    TFLOPS - Avg: \(String(format: "%.3f", avgTflops)), Min: \(String(format: "%.3f", minTflops)), Max: \(String(format: "%.3f", maxTflops))")

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
func runParallel(benchmarks: [FLOPSBenchmark], numThreads: Int, iterations: UInt32) -> [(tflops: Double, time: Double)] {
    print("\n=== Running in PARALLEL mode ===\n")
    print("--- FP32 Benchmark ---\n")

    let group = DispatchGroup()
    let lock = NSLock()
    var results: [(tflops: Double, time: Double)?] = Array(repeating: nil, count: benchmarks.count)

    let overallStart = CACurrentMediaTime()

    for (index, benchmark) in benchmarks.enumerated() {
        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            let result = benchmark.runBenchmark(
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

// Run benchmarks on selected devices sequentially
func runSequential(benchmarks: [FLOPSBenchmark], numThreads: Int, iterations: UInt32) -> [(tflops: Double, time: Double)] {
    print("\n=== Running in SEQUENTIAL mode ===\n")

    var results: [(tflops: Double, time: Double)] = []

    for (index, benchmark) in benchmarks.enumerated() {
        print("--- GPU \(index): \(benchmark.device.name) ---\n")

        if let result = benchmark.runBenchmark(
            numThreads: numThreads,
            iterations: iterations
        ) {
            results.append(result)
        }

        print()
    }

    return results
}

// Print summary
func printSummary(benchmarks: [FLOPSBenchmark], results: [(tflops: Double, time: Double)]) {
    print("=== Summary ===\n")
    print("Per-GPU Performance:")
    for (index, benchmark) in benchmarks.enumerated() {
        print("\nGPU \(index): \(benchmark.device.name)")
        if index < results.count {
            print("  FP32: \(String(format: "%.3f", results[index].tflops)) TFLOPS")
            print("  Average time: \(String(format: "%.3f", results[index].time))s")
        }
    }

    if benchmarks.count > 1 {
        let totalTFLOPS = results.map { $0.tflops }.reduce(0, +)

        print("\nCombined Performance:")
        print("  Total FP32: \(String(format: "%.3f", totalTFLOPS)) TFLOPS")

        print("\nScaling Efficiency:")
        print("  Number of GPUs: \(benchmarks.count)")
        print("  Average per GPU: \(String(format: "%.3f", totalTFLOPS/Double(benchmarks.count))) TFLOPS")
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
let libraryPath = "build/compute_benchmark.metallib"
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
let results: [(tflops: Double, time: Double)]

switch mode {
case .parallel:
    results = runParallel(benchmarks: benchmarks, numThreads: numThreads, iterations: iterations)
case .sequential, .single:
    results = runSequential(benchmarks: benchmarks, numThreads: numThreads, iterations: iterations)
}

// Print summary
printSummary(benchmarks: benchmarks, results: results)
