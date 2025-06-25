#!/usr/bin/env swift

import Foundation
import Darwin

// Core type enumeration
enum CoreType: Int {
    case unknown = -1
    case efficiency = 1
    case performance = 2
}

// Structure to represent a CPU core
struct Core {
    let id: Int32
    let type: CoreType
    let clusterId: Int32
}

// Structure to hold CPU load information
struct CPULoad {
    var totalUsage: Double = 0
    var systemLoad: Double = 0
    var userLoad: Double = 0
    var idleLoad: Double = 0
    var usagePerCore: [Double] = []
    var usageECores: Double = 0
    var usagePCores: Double = 0
}

// CPU Monitor class
class CPUMonitor {
    private var cpuInfo: processor_info_array_t!
    private var prevCpuInfo: processor_info_array_t?
    private var numCpuInfo: mach_msg_type_number_t = 0
    private var numPrevCpuInfo: mach_msg_type_number_t = 0
    private var numCPUs: uint = 0
    private var previousInfo = host_cpu_load_info()
    var cores: [Core] = []
    private var eCores: Int32 = 0
    private var pCores: Int32 = 0
    
    init() {
        setupCPUInfo()
        detectCoreTopology()
    }
    
    private func setupCPUInfo() {
        var mib = [CTL_HW, HW_NCPU]
        var sizeOfNumCPUs: size_t = MemoryLayout<uint>.size
        let status = sysctl(&mib, 2, &numCPUs, &sizeOfNumCPUs, nil, 0)
        if status != 0 {
            numCPUs = 1
        }
    }
    
    private func detectCoreTopology() {
        searchInIORegistry()
        cores.sort { $0.id < $1.id }
    }
    
    private func searchInIORegistry() {
        searchForCPUEntries()
        if cores.isEmpty {
            searchInDeviceTree()
        }
        if cores.isEmpty {
            searchForARMCPUs()
        }
    }
    
    private func searchForCPUEntries() {
        var iterator = io_iterator_t()
        let result = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"), &iterator)
        if result != kIOReturnSuccess {
            return
        }
        
        while case let service = IOIteratorNext(iterator), service != 0 {
            searchServiceRecursively(service, depth: 0)
            IOObjectRelease(service)
        }
        IOObjectRelease(iterator)
    }
    
    private func searchServiceRecursively(_ service: io_registry_entry_t, depth: Int) {
        var name: io_name_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        IORegistryEntryGetName(service, &name)
        let nameStr = withUnsafePointer(to: &name) {
            $0.withMemoryRebound(to: CChar.self, capacity: 128) {
                String(validatingUTF8: $0) ?? ""
            }
        }
        
        // Check if this looks like a CPU entry
        if nameStr.lowercased().contains("cpu") || nameStr.lowercased().contains("core") {
            examineEntry(service, name: nameStr)
        }
        
        // Recurse into children if not too deep
        if depth < 3 {
            var iterator = io_iterator_t()
            if IORegistryEntryGetChildIterator(service, kIOServicePlane, &iterator) == kIOReturnSuccess {
                var child = IOIteratorNext(iterator)
                while child != 0 {
                    searchServiceRecursively(child, depth: depth + 1)
                    IOObjectRelease(child)
                    child = IOIteratorNext(iterator)
                }
                IOObjectRelease(iterator)
            }
        }
    }
    
    private func searchInDeviceTree() {
        let rootEntry = IORegistryGetRootEntry(kIOMainPortDefault)
        
        var iterator = io_iterator_t()
        if IORegistryEntryGetChildIterator(rootEntry, kIODeviceTreePlane, &iterator) == kIOReturnSuccess {
            var child = IOIteratorNext(iterator)
            while child != 0 {
                searchDeviceTreeRecursively(child, depth: 0)
                IOObjectRelease(child)
                child = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }
    }
    
    private func searchDeviceTreeRecursively(_ entry: io_registry_entry_t, depth: Int) {
        var name: io_name_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        IORegistryEntryGetName(entry, &name)
        let nameStr = withUnsafePointer(to: &name) {
            $0.withMemoryRebound(to: CChar.self, capacity: 128) {
                String(validatingUTF8: $0) ?? ""
            }
        }
        
        if nameStr.lowercased().contains("cpu") || nameStr.lowercased().contains("core") {
            examineEntry(entry, name: nameStr)
        }
        
        // Recurse into children if not too deep
        if depth < 2 {
            var iterator = io_iterator_t()
            if IORegistryEntryGetChildIterator(entry, kIODeviceTreePlane, &iterator) == kIOReturnSuccess {
                var child = IOIteratorNext(iterator)
                while child != 0 {
                    searchDeviceTreeRecursively(child, depth: depth + 1)
                    IOObjectRelease(child)
                    child = IOIteratorNext(iterator)
                }
                IOObjectRelease(iterator)
            }
        }
    }
    
    private func searchForARMCPUs() {
        var iterator = io_iterator_t()
        if IOServiceGetMatchingServices(kIOMainPortDefault, nil, &iterator) == kIOReturnSuccess {
            var service = IOIteratorNext(iterator)
            while service != 0 {
                var name: io_name_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                IORegistryEntryGetName(service, &name)
                let nameStr = withUnsafePointer(to: &name) {
                    $0.withMemoryRebound(to: CChar.self, capacity: 128) {
                        String(validatingUTF8: $0) ?? ""
                    }
                }
                
                if nameStr.lowercased().contains("arm") || nameStr.lowercased().contains("cpu") {
                    examineEntry(service, name: nameStr)
                }
                
                IOObjectRelease(service)
                service = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }
    }
    
    private func examineEntry(_ entry: io_registry_entry_t, name: String) {
        var props: Unmanaged<CFMutableDictionary>?
        if IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0) == kIOReturnSuccess,
           let properties = props?.takeRetainedValue() as? [String: Any] {
            
            // Process CPU entries
            if name.hasPrefix("cpu") && name.count > 3 {
                processCPUEntry(properties, name: name)
            } else if name == "cpus" {
                processCPUsEntry(properties)
            }
        }
    }
    
    private func processCPUEntry(_ properties: [String: Any], name: String) {
        var coreType: CoreType = .unknown
        
        // Check for cluster-type
        if let rawType = properties["cluster-type"] as? Data,
           let type = String(data: rawType, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !type.isEmpty {
            
            let firstChar = type.first!
            switch firstChar {
            case "E":
                coreType = .efficiency
            case "P":
                coreType = .performance
            default:
                coreType = .unknown
            }
        }
        
        let cpuId = (properties["cpu-id"] as? Data)?.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        } ?? -1
        
        let clusterId = (properties["cluster-id"] as? Data)?.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        } ?? -1
        
        cores.append(Core(id: cpuId, type: coreType, clusterId: clusterId))
    }
    
    private func processCPUsEntry(_ properties: [String: Any]) {
        eCores = (properties["e-core-count"] as? Data)?.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        } ?? 0
        pCores = (properties["p-core-count"] as? Data)?.withUnsafeBytes { pointer in
            return pointer.load(as: Int32.self)
        } ?? 0
    }
    
    func readCPULoad() -> CPULoad? {
        var response = CPULoad()
        var numCPUsU: natural_t = 0
        
        // Get per-core usage
        let result = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &numCPUsU, &cpuInfo, &numCpuInfo)
        if result == KERN_SUCCESS {
            var usagePerCore: [Double] = []
            
            for i in 0..<Int32(numCPUs) {
                var inUse: Int32
                var total: Int32
                
                if let prevCpuInfo = prevCpuInfo {
                    inUse = cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_USER)]
                        - prevCpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_USER)]
                        + cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_SYSTEM)]
                        - prevCpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_SYSTEM)]
                        + cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_NICE)]
                        - prevCpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_NICE)]
                    total = inUse + (cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_IDLE)]
                        - prevCpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_IDLE)])
                } else {
                    inUse = cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_USER)]
                        + cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_SYSTEM)]
                        + cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_NICE)]
                    total = inUse + cpuInfo[Int(CPU_STATE_MAX * i + CPU_STATE_IDLE)]
                }
                
                if total != 0 {
                    usagePerCore.append(Double(inUse) / Double(total))
                } else {
                    usagePerCore.append(0)
                }
            }
            
            response.usagePerCore = usagePerCore
            
            // Calculate E-core and P-core averages if we have topology info
            if !cores.isEmpty {
                let eCoresList = cores.filter { $0.type == .efficiency }.compactMap { core -> Double? in
                    if usagePerCore.indices.contains(Int(core.id)) {
                        return usagePerCore[Int(core.id)]
                    }
                    return nil
                }
                
                let pCoresList = cores.filter { $0.type == .performance }.compactMap { core -> Double? in
                    if usagePerCore.indices.contains(Int(core.id)) {
                        return usagePerCore[Int(core.id)]
                    }
                    return nil
                }
                
                if !eCoresList.isEmpty {
                    response.usageECores = eCoresList.reduce(0, +) / Double(eCoresList.count)
                }
                if !pCoresList.isEmpty {
                    response.usagePCores = pCoresList.reduce(0, +) / Double(pCoresList.count)
                }
            }
            
            // Clean up
            if let prevCpuInfo = prevCpuInfo {
                let prevCpuInfoSize = MemoryLayout<integer_t>.stride * Int(numPrevCpuInfo)
                vm_deallocate(mach_task_self_, vm_address_t(bitPattern: prevCpuInfo), vm_size_t(prevCpuInfoSize))
            }
            
            prevCpuInfo = cpuInfo
            numPrevCpuInfo = numCpuInfo
            cpuInfo = nil
            numCpuInfo = 0
        }
        
        // Get overall CPU usage
        if let cpuLoadInfo = getHostCPULoadInfo() {
            let userDiff = Double(cpuLoadInfo.cpu_ticks.0 - previousInfo.cpu_ticks.0)
            let sysDiff = Double(cpuLoadInfo.cpu_ticks.1 - previousInfo.cpu_ticks.1)
            let idleDiff = Double(cpuLoadInfo.cpu_ticks.2 - previousInfo.cpu_ticks.2)
            let niceDiff = Double(cpuLoadInfo.cpu_ticks.3 - previousInfo.cpu_ticks.3)
            let totalTicks = sysDiff + userDiff + niceDiff + idleDiff
            
            if totalTicks > 0 {
                response.systemLoad = sysDiff / totalTicks
                response.userLoad = userDiff / totalTicks
                response.idleLoad = idleDiff / totalTicks
                response.totalUsage = response.systemLoad + response.userLoad
            }
            
            previousInfo = cpuLoadInfo
        }
        
        return response
    }
    
    private func getHostCPULoadInfo() -> host_cpu_load_info? {
        let count = MemoryLayout<host_cpu_load_info>.stride / MemoryLayout<integer_t>.stride
        var size = mach_msg_type_number_t(count)
        var cpuLoadInfo = host_cpu_load_info()
        
        let result = withUnsafeMutablePointer(to: &cpuLoadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: count) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &size)
            }
        }
        
        if result != KERN_SUCCESS {
            return nil
        }
        
        return cpuLoadInfo
    }
    
    func printTopology() {
        print("\n=== CPU Topology ===")
        print("Total Cores: \(numCPUs)")
        
        let detectedECores = cores.filter { $0.type == .efficiency }.count
        let detectedPCores = cores.filter { $0.type == .performance }.count
        
        if detectedECores > 0 || detectedPCores > 0 {
            print("Efficiency Cores: \(detectedECores)")
            print("Performance Cores: \(detectedPCores)")
        }
        
        // Group cores by cluster (filter out invalid cores)
        let validCores = cores.filter { $0.id >= 0 && $0.clusterId >= 0 }
        let clusters = Dictionary(grouping: validCores) { $0.clusterId }
        let sortedClusterIds = clusters.keys.sorted()
        
        if !sortedClusterIds.isEmpty {
            print("\nCluster Layout:")
            for clusterId in sortedClusterIds {
                let clusterCores = clusters[clusterId]!.sorted { $0.id < $1.id }
                let clusterType = clusterCores.first?.type
                let typeStr = clusterType == .efficiency ? "E-Cluster" : (clusterType == .performance ? "P-Cluster" : "Unknown")
                
                let coreIds = clusterCores.map { "\($0.id)" }.joined(separator: ", ")
                print("  Cluster \(clusterId) (\(typeStr)): Cores \(coreIds)")
            }
        }
        print("")
    }
}

// Main program
let monitor = CPUMonitor()

// Print topology once
monitor.printTopology()

// Read CPU usage once
print("\n=== CPU Usage Measurement ===")

// Initial read to establish baseline
_ = monitor.readCPULoad()
Thread.sleep(forTimeInterval: 1.0)

// Second read to get actual usage
if let load = monitor.readCPULoad() {
    // Print overall usage
    let totalStr = String(format: "%.1f", load.totalUsage * 100)
    print("Total CPU Usage: \(totalStr)%")
    
    let userStr = String(format: "%.1f", load.userLoad * 100)
    let sysStr = String(format: "%.1f", load.systemLoad * 100)
    let idleStr = String(format: "%.1f", load.idleLoad * 100)
    print("User: \(userStr)% | System: \(sysStr)% | Idle: \(idleStr)%")
    
    // Print E-core and P-core usage if available
    if load.usageECores > 0 || load.usagePCores > 0 {
        let eCoreStr = String(format: "%.1f", load.usageECores * 100)
        let pCoreStr = String(format: "%.1f", load.usagePCores * 100)
        print("E-Cores: \(eCoreStr)% | P-Cores: \(pCoreStr)%")
    }
    
    // Print per-core usage
    print("\nPer-Core Usage:")
    for (index, usage) in load.usagePerCore.enumerated() {
        if let core = monitor.cores.first(where: { $0.id == Int32(index) }) {
            let typeStr = core.type == .efficiency ? "E" : (core.type == .performance ? "P" : "?")
            let usageStr = String(format: "%.1f", usage * 100)
            print("Core \(index) (\(typeStr)\(core.clusterId)): \(usageStr)%")
        } else {
            let usageStr = String(format: "%.1f", usage * 100)
            print("Core \(index) (?): \(usageStr)%")
        }
    }
    
} else {
    print("ERROR: Failed to read CPU usage")
}