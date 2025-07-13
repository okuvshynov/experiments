#!/usr/bin/env swift

import Foundation
import Darwin
import IOKit

// MARK: - CPU Monitoring

enum CoreType: Int {
    case unknown = -1
    case efficiency = 1
    case performance = 2
}

struct Core {
    let id: Int32
    let type: CoreType
    let clusterId: Int32
}

struct CPULoad {
    var totalUsage: Double = 0
    var usagePerCore: [Double] = []
}

class CPUMonitor {
    private var cpuInfo: processor_info_array_t!
    private var prevCpuInfo: processor_info_array_t?
    private var numCpuInfo: mach_msg_type_number_t = 0
    private var numPrevCpuInfo: mach_msg_type_number_t = 0
    private var numCPUs: uint = 0
    private var previousInfo = host_cpu_load_info()
    var cores: [Core] = []
    
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
        
        if nameStr.lowercased().contains("cpu") || nameStr.lowercased().contains("core") {
            examineEntry(service, name: nameStr)
        }
        
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
            
            if name.hasPrefix("cpu") && name.count > 3 {
                processCPUEntry(properties, name: name)
            }
        }
    }
    
    private func processCPUEntry(_ properties: [String: Any], name: String) {
        var coreType: CoreType = .unknown
        
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
    
    func readCPULoad() -> CPULoad? {
        var response = CPULoad()
        var numCPUsU: natural_t = 0
        
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
            
            if let prevCpuInfo = prevCpuInfo {
                let prevCpuInfoSize = MemoryLayout<integer_t>.stride * Int(numPrevCpuInfo)
                vm_deallocate(mach_task_self_, vm_address_t(bitPattern: prevCpuInfo), vm_size_t(prevCpuInfoSize))
            }
            
            prevCpuInfo = cpuInfo
            numPrevCpuInfo = numCpuInfo
            cpuInfo = nil
            numCpuInfo = 0
        }
        
        return response
    }
}

// MARK: - GPU Monitoring

func fetchIOService(_ name: String) -> [NSDictionary]? {
    var iterator: io_iterator_t = io_iterator_t()
    var obj: io_registry_entry_t = 1
    var list: [NSDictionary] = []
    
    let result = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching(name), &iterator)
    if result != kIOReturnSuccess {
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
    func readGPUUtilization() -> Double? {
        guard let accelerators = fetchIOService("IOAccelerator") else {
            return nil
        }
        
        for accelerator in accelerators {
            guard let stats = accelerator["PerformanceStatistics"] as? [String: Any] else {
                continue
            }
            
            let utilization: Int? = stats["Device Utilization %"] as? Int ?? stats["GPU Activity(%)"] as? Int
            if let utilization = utilization {
                return Double(utilization) / 100.0
            }
        }
        
        return nil
    }
}

// MARK: - Memory Monitoring

class MemoryMonitor {
    func readMemoryInfo() -> (total: UInt64, used: UInt64, wired: UInt64)? {
        // Get total physical memory
        var totalMemory: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &totalMemory, &size, nil, 0)
        
        // Get VM statistics
        var vmInfo = vm_statistics64()
        var vmInfoSize = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmInfoSize)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmInfoSize)
            }
        }
        
        if result != KERN_SUCCESS {
            return nil
        }
        
        // Calculate memory values in bytes
        let pageSize = UInt64(vm_page_size)
        let wiredMemory = UInt64(vmInfo.wire_count) * pageSize
        let activeMemory = UInt64(vmInfo.active_count) * pageSize
        let compressedMemory = UInt64(vmInfo.compressor_page_count) * pageSize
        
        // Calculate used memory (active + wired + compressed)
        // This approximates what Activity Monitor shows as "Memory Used"
        let usedMemory = activeMemory + wiredMemory + compressedMemory
        
        return (total: totalMemory, used: usedMemory, wired: wiredMemory)
    }
}

// MARK: - Main Program

let cpuMonitor = CPUMonitor()
let gpuMonitor = GPUMonitor()
let memoryMonitor = MemoryMonitor()

// Initial read to establish baseline for CPU
_ = cpuMonitor.readCPULoad()
Thread.sleep(forTimeInterval: 0.5)

// Get CPU utilization
let cpuLoad = cpuMonitor.readCPULoad()

// Get GPU utilization
let gpuUtilization = gpuMonitor.readGPUUtilization() ?? 0.0

// Get memory info
let memoryInfo = memoryMonitor.readMemoryInfo()

// Print GPU utilization
print("gpu_util=\(String(format: "%.3f", gpuUtilization))")

// Print memory utilization
if let memory = memoryInfo {
    // Convert to GB for readability
    let totalGB = Double(memory.total) / (1024 * 1024 * 1024)
    let usedGB = Double(memory.used) / (1024 * 1024 * 1024)
    let wiredGB = Double(memory.wired) / (1024 * 1024 * 1024)
    
    // Calculate utilization ratios
    let usedRatio = usedGB / totalGB
    let wiredRatio = wiredGB / totalGB
    
    print("memory.total_gb=\(String(format: "%.3f", totalGB))")
    print("memory.used_gb=\(String(format: "%.3f", usedGB))")
    print("memory.wired_gb=\(String(format: "%.3f", wiredGB))")
    print("memory.used_ratio=\(String(format: "%.3f", usedRatio))")
    print("memory.wired_ratio=\(String(format: "%.3f", wiredRatio))")
    
    // Also provide raw bytes for compatibility
    print("memory.used_bytes=\(memory.used)")
    print("memory.wired_bytes=\(memory.wired)")
}

// Print CPU utilization per core with cluster info
if let cpuLoad = cpuLoad {
    for (index, usage) in cpuLoad.usagePerCore.enumerated() {
        if let core = cpuMonitor.cores.first(where: { $0.id == Int32(index) }) {
            let typeChar = core.type == .efficiency ? "E" : (core.type == .performance ? "P" : "U")
            print("cpu.\(typeChar)\(core.clusterId).\(core.id)_util=\(String(format: "%.3f", usage))")
        } else {
            // Fallback if core topology not detected
            print("cpu.U0.\(index)_util=\(String(format: "%.3f", usage))")
        }
    }
}