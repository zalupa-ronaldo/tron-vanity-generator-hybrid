#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct CpuInfo {
    std::string brand;
    unsigned int logicalCores = 0;
    bool aesni = false;
    bool avx2 = false;
};

struct GpuDevice {
    std::string platform;
    std::string name;
    std::string vendor;      // Intel / AMD / NVIDIA ...
    unsigned int computeUnits = 0;
    unsigned int clockMHz = 0;
    uint64_t globalMemBytes = 0;
    bool integrated = false;  // 由 HOST_UNIFIED_MEMORY 推断
    void* platformId = nullptr;  // cl_platform_id
    void* deviceId = nullptr;    // cl_device_id
    int cudaOrdinal = -1;
    int cudaCapability = 0;
};

struct HardwareReport {
    CpuInfo cpu;
    std::vector<GpuDevice> gpus;
    std::vector<GpuDevice> cudaGpus;
    std::string cudaNote;
    bool openclAvailable = false;
    std::string openclNote;   // 说明为什么没有 GPU（缺 OpenCL.dll 等）
};

CpuInfo detectCpu();
HardwareReport detectHardware();
