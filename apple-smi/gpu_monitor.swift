#!/usr/bin/env swift

import Foundation
import IOKit

func fetchIOService(_ name: String) -> [NSDictionary]? {
    var iterator: io_iterator_t = io_iterator_t()
    var obj: io_registry_entry_t = 1
    var list: [NSDictionary] = []
    
    let result = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching(name), &iterator)
    if result != kIOReturnSuccess {
        print("Error IOServiceGetMatchingServices(): " + (String(cString: mach_error_string(result), encoding: String.Encoding.ascii) ?? "unknown error"))
        return nil
    }
    
    while obj != 0 {
        obj = IOIteratorNext(iterator)
        if let props = getIOProperties(obj) {
            list.append(props)
        }
        IOObjectRelease(obj)
    }
    IOObjectRelease(iterator)
    
    return list.isEmpty ? nil : list
}

func getIOProperties(_ entry: io_registry_entry_t) -> NSDictionary? {
    var properties: Unmanaged<CFMutableDictionary>? = nil
    
    if IORegistryEntryCreateCFProperties(entry, &properties, kCFAllocatorDefault, 0) != kIOReturnSuccess {
        return nil
    }
    
    defer {
        properties?.release()
    }
    
    return properties?.takeUnretainedValue()
}

class GPUMonitor {
    func readGPUUtilization() -> [(name: String, utilization: Double?, temperature: Double?)] {
        guard let accelerators = fetchIOService("IOAccelerator") else {
            print("No GPU accelerators found")
            return []
        }
        
        var gpuInfos: [(name: String, utilization: Double?, temperature: Double?)] = []
        
        for (index, accelerator) in accelerators.enumerated() {
            guard let ioClass = accelerator.object(forKey: "IOClass") as? String else {
                continue
            }
            
            guard let stats = accelerator["PerformanceStatistics"] as? [String: Any] else {
                continue
            }
            
            // Debug: Print all available stats
            print("=== Debug: All Performance Statistics ===")
            for (key, value) in stats.sorted(by: { $0.key < $1.key }) {
                print("  \(key): \(value)")
            }
            print("==========================================\n")
            
            // Get GPU name and type
            let ioClassLower = ioClass.lowercased()
            var gpuName = "Unknown GPU"
            
            if ioClassLower == "nvaccelerator" || ioClassLower.contains("nvidia") {
                gpuName = "NVIDIA GPU"
            } else if ioClassLower.contains("amd") {
                gpuName = "AMD GPU"
            } else if ioClassLower.contains("intel") {
                gpuName = "Intel GPU"
            } else if ioClassLower.contains("agx") {
                gpuName = stats["model"] as? String ?? "Apple Silicon GPU"
            }
            
            // Add index if multiple GPUs of same type
            if accelerators.count > 1 {
                gpuName += " #\(index)"
            }
            
            // Get utilization percentage
            let utilization: Int? = stats["Device Utilization %"] as? Int ?? stats["GPU Activity(%)"] as? Int
            let utilizationPercent = utilization != nil ? Double(utilization!) / 100.0 : nil
            
            // Get temperature
            let temperature: Int? = stats["Temperature(C)"] as? Int
            let tempCelsius = temperature != nil ? Double(temperature!) : nil
            
            gpuInfos.append((name: gpuName, utilization: utilizationPercent, temperature: tempCelsius))
        }
        
        return gpuInfos
    }
    
    func displayGPUInfo() {
        let gpuInfos = readGPUUtilization()
        
        if gpuInfos.isEmpty {
            print("No GPU information available")
            return
        }
        
        print("GPU Utilization Monitor")
        print("======================")
        
        for info in gpuInfos {
            print("\n\(info.name):")
            
            if let utilization = info.utilization {
                let percentage = Int(utilization * 100)
                let bar = String(repeating: "█", count: percentage / 5)
                let empty = String(repeating: "░", count: 20 - (percentage / 5))
                print("  Utilization: \(percentage)% [\(bar)\(empty)]")
            } else {
                print("  Utilization: N/A")
            }
            
            if let temperature = info.temperature {
                print("  Temperature: \(Int(temperature))°C")
            } else {
                print("  Temperature: N/A")
            }
        }
    }
    
    func startMonitoring(interval: TimeInterval = 1.0) {
        print("Starting GPU monitoring (press Ctrl+C, Q, or Esc to stop)...")
        print("Update interval: \(interval) seconds\n")
        
        // Set up signal handlers
        signal(SIGINT) { _ in
            print("\nReceived interrupt signal. Stopping...")
            exit(0)
        }
        
        signal(SIGTERM) { _ in
            print("\nReceived termination signal. Stopping...")
            exit(0)
        }
        
        let timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
            // Clear screen
            print("\u{001B}[2J\u{001B}[H", terminator: "")
            self.displayGPUInfo()
            print("\nLast updated: \(Date())")
        }
        
        // Initial display
        displayGPUInfo()
        print("\nLast updated: \(Date())")
        
        // Keep the program running with proper signal handling
        RunLoop.current.run()
    }
}

func printUsage() {
    print("GPU Monitor - Apple Silicon GPU Utilization Tool")
    print("Usage:")
    print("  swift gpu_monitor.swift [options]")
    print("  ./gpu_monitor [options]")
    print("")
    print("Options:")
    print("  -h, --help          Show this help message")
    print("  -o, --once          Show GPU info once and exit")
    print("  -i, --interval N    Update interval in seconds (default: 1.0)")
    print("")
    print("Examples:")
    print("  swift gpu_monitor.swift -o")
    print("  swift gpu_monitor.swift -i 2.0")
}

func main() {
    let args = CommandLine.arguments
    var showOnce = false
    var interval: TimeInterval = 1.0
    
    // Parse command line arguments
    var i = 1
    while i < args.count {
        switch args[i] {
        case "-h", "--help":
            printUsage()
            return
        case "-o", "--once":
            showOnce = true
        case "-i", "--interval":
            if i + 1 < args.count, let value = Double(args[i + 1]) {
                interval = value
                i += 1
            } else {
                print("Error: --interval requires a numeric value")
                return
            }
        default:
            print("Error: Unknown option '\(args[i])'")
            printUsage()
            return
        }
        i += 1
    }
    
    let monitor = GPUMonitor()
    
    if showOnce {
        monitor.displayGPUInfo()
    } else {
        monitor.startMonitoring(interval: interval)
    }
}

// Run the program
main()
