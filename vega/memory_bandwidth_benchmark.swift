import Metal
import Foundation
import QuartzCore

class MemoryBandwidthBenchmark {
    let device: MTLDevice
    let deviceIndex: Int
    let commandQueue: MTLCommandQueue
    let copyPipeline: MTLComputePipelineState
    let copyVec4Pipeline: MTLComputePipelineState
    let readPipeline: MTLComputePipelineState
    let writePipeline: MTLComputePipelineState
    let stridedPipeline: MTLComputePipelineState
    let randomPipeline: MTLComputePipelineState

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

        // Load Metal shader library
        let libraryURL = URL(fileURLWithPath: libraryPath)
        guard let library = try? device.makeLibrary(URL: libraryURL) else {
            print("  Failed to load Metal library from \(libraryPath)")
            return nil
        }

        // Create compute pipelines for different bandwidth tests
        guard let copyFunc = library.makeFunction(name: "bandwidth_copy"),
              let copyVec4Func = library.makeFunction(name: "bandwidth_copy_vec4"),
              let readFunc = library.makeFunction(name: "bandwidth_read"),
              let writeFunc = library.makeFunction(name: "bandwidth_write"),
              let stridedFunc = library.makeFunction(name: "bandwidth_strided"),
              let randomFunc = library.makeFunction(name: "bandwidth_random") else {
            print("  Failed to load shader functions")
            return nil
        }

        guard let copy = try? device.makeComputePipelineState(function: copyFunc),
              let copyVec4 = try? device.makeComputePipelineState(function: copyVec4Func),
              let read = try? device.makeComputePipelineState(function: readFunc),
              let write = try? device.makeComputePipelineState(function: writeFunc),
              let strided = try? device.makeComputePipelineState(function: stridedFunc),
              let random = try? device.makeComputePipelineState(function: randomFunc) else {
            print("  Failed to create compute pipeline states")
            return nil
        }

        self.copyPipeline = copy
        self.copyVec4Pipeline = copyVec4
        self.readPipeline = read
        self.writePipeline = write
        self.stridedPipeline = strided
        self.randomPipeline = random
    }

    func runCopyBenchmark(arraySize: Int, numRuns: Int = 5) -> (bandwidth: Double, time: Double)? {
        let bufferSize = arraySize * MemoryLayout<Float>.size

        // Use .storageModePrivate for GPU-only memory (HBM) - maximum bandwidth
        guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModePrivate),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
            print("  Failed to create buffers")
            return nil
        }

        var arrayLength = UInt32(arraySize)
        guard let lengthBuffer = device.makeBuffer(bytes: &arrayLength, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return nil
        }

        let threadGroupSize = MTLSize(width: min(copyPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (arraySize + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)

        // Warm-up
        for _ in 0..<2 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(copyPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 2)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Timed runs
        var times: [Double] = []
        for _ in 0..<numRuns {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(copyPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 2)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()

            let startTime = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let endTime = CACurrentMediaTime()

            times.append(endTime - startTime)
        }

        let avgTime = times.reduce(0, +) / Double(numRuns)
        let minTime = times.min()!
        let maxTime = times.max()!

        // Copy involves reading and writing, so 2x the data size
        let totalBytes = Double(bufferSize * 2)
        let avgBandwidth = totalBytes / avgTime / 1e9  // GB/s
        let minBandwidth = totalBytes / maxTime / 1e9
        let maxBandwidth = totalBytes / minTime / 1e9

        return (avgBandwidth, avgTime)
    }

    func runReadBenchmark(arraySize: Int, numRuns: Int = 5) -> (bandwidth: Double, time: Double)? {
        let bufferSize = arraySize * MemoryLayout<Float>.size
        let outputSize = (arraySize / 8) * MemoryLayout<Float>.size

        guard let inputBuffer = device.makeBuffer(length: bufferSize, options: .storageModePrivate),
              let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModePrivate) else {
            print("  Failed to create buffers")
            return nil
        }

        var arrayLength = UInt32(arraySize)
        guard let lengthBuffer = device.makeBuffer(bytes: &arrayLength, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return nil
        }

        let numThreads = arraySize / 8
        let threadGroupSize = MTLSize(width: min(readPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (numThreads + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)

        // Warm-up
        for _ in 0..<2 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(readPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 2)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Timed runs
        var times: [Double] = []
        for _ in 0..<numRuns {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(readPipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 2)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()

            let startTime = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let endTime = CACurrentMediaTime()

            times.append(endTime - startTime)
        }

        let avgTime = times.reduce(0, +) / Double(numRuns)
        let minTime = times.min()!

        // Read test: reading 8 floats per thread, writing 1
        let totalBytesRead = Double(bufferSize)
        let avgBandwidth = totalBytesRead / avgTime / 1e9  // GB/s
        let maxBandwidth = totalBytesRead / minTime / 1e9

        return (avgBandwidth, avgTime)
    }

    func runWriteBenchmark(arraySize: Int, numRuns: Int = 5) -> (bandwidth: Double, time: Double)? {
        let bufferSize = arraySize * MemoryLayout<Float>.size

        guard let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
            print("  Failed to create buffer")
            return nil
        }

        var arrayLength = UInt32(arraySize)
        guard let lengthBuffer = device.makeBuffer(bytes: &arrayLength, length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return nil
        }

        let numThreads = arraySize / 8
        let threadGroupSize = MTLSize(width: min(writePipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (numThreads + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)

        // Warm-up
        for _ in 0..<2 {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Timed runs
        var times: [Double] = []
        for _ in 0..<numRuns {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }

            encoder.setComputePipelineState(writePipeline)
            encoder.setBuffer(outputBuffer, offset: 0, index: 0)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()

            let startTime = CACurrentMediaTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let endTime = CACurrentMediaTime()

            times.append(endTime - startTime)
        }

        let avgTime = times.reduce(0, +) / Double(numRuns)
        let minTime = times.min()!

        let totalBytesWritten = Double(bufferSize)
        let avgBandwidth = totalBytesWritten / avgTime / 1e9  // GB/s
        let maxBandwidth = totalBytesWritten / minTime / 1e9

        return (avgBandwidth, avgTime)
    }
}

// Main execution
print("=== Metal HBM Memory Bandwidth Benchmark ===\n")

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
        print()
    }
    exit(0)
}

let libraryPath = "build/memory_bandwidth.metallib"

// Parse device selection
var deviceIndex = 0
if let idx = CommandLine.arguments.firstIndex(of: "--device"),
   idx + 1 < CommandLine.arguments.count,
   let devNum = Int(CommandLine.arguments[idx + 1]) {
    deviceIndex = devNum
}

guard deviceIndex < devices.count else {
    print("Error: Device index \(deviceIndex) out of range (0-\(devices.count - 1))")
    exit(1)
}

print("Testing GPU \(deviceIndex)\n")

guard let benchmark = MemoryBandwidthBenchmark(device: devices[deviceIndex], index: deviceIndex, libraryPath: libraryPath) else {
    print("Failed to initialize benchmark")
    exit(1)
}

// Test with different array sizes to see how bandwidth scales
// Using large arrays to fill HBM and avoid cache effects
let arraySizes = [
    (name: "64 MB", size: 16 * 1024 * 1024),    // 64 MB (16M floats)
    (name: "256 MB", size: 64 * 1024 * 1024),   // 256 MB (64M floats)
    (name: "1 GB", size: 256 * 1024 * 1024),    // 1 GB (256M floats)
]

print("\nConfiguration:")
print("  Test types: Copy (R+W), Read-only, Write-only")
print("  Runs per test: 5")
print()

for test in arraySizes {
    print("=== Testing with \(test.name) arrays ===\n")

    print("Copy (Read + Write):")
    if let result = benchmark.runCopyBenchmark(arraySize: test.size, numRuns: 5) {
        print("  Bandwidth: \(String(format: "%.2f", result.bandwidth)) GB/s")
        print("  Time: \(String(format: "%.4f", result.time))s")
    }

    print("\nRead-only:")
    if let result = benchmark.runReadBenchmark(arraySize: test.size, numRuns: 5) {
        print("  Bandwidth: \(String(format: "%.2f", result.bandwidth)) GB/s")
        print("  Time: \(String(format: "%.4f", result.time))s")
    }

    print("\nWrite-only:")
    if let result = benchmark.runWriteBenchmark(arraySize: test.size, numRuns: 5) {
        print("  Bandwidth: \(String(format: "%.2f", result.bandwidth)) GB/s")
        print("  Time: \(String(format: "%.4f", result.time))s")
    }

    print("\n" + String(repeating: "-", count: 50) + "\n")
}

print("=== Summary ===")
print("AMD Radeon Pro Vega II specifications:")
print("  HBM2 memory bandwidth: 1024 GB/s (theoretical)")
print("  Memory: 32 GB HBM2")
print("\nNote: Real-world bandwidth is typically 70-90% of theoretical peak")
