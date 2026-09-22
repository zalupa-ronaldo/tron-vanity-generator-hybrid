#include "hwdetect.h"
#include "ocl.h"
#include "cuda_driver.h"

#include <cstring>
#include <iostream>
#include <thread>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#  include <cpuid.h>
#elif defined(_MSC_VER)
#  include <intrin.h>
#endif

// ---------------- CPU ----------------

CpuInfo detectCpu() {
    CpuInfo info;
    info.logicalCores = std::thread::hardware_concurrency();
    if (info.logicalCores == 0) info.logicalCores = 4;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    auto cpuid = [](unsigned int leaf, unsigned int sub,
                    unsigned int& a, unsigned int& b, unsigned int& c, unsigned int& d) {
#  if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
        a = r[0]; b = r[1]; c = r[2]; d = r[3];
#  else
        unsigned int ra = 0, rb = 0, rc = 0, rd = 0;
        __get_cpuid_count(leaf, sub, &ra, &rb, &rc, &rd);
        a = ra; b = rb; c = rc; d = rd;
#  endif
    };

    unsigned int a, b, c, d;
    char brand[49] = {0};
    cpuid(0x80000000u, 0, a, b, c, d);
    if (a >= 0x80000004u) {
        unsigned int* p = reinterpret_cast<unsigned int*>(brand);
        cpuid(0x80000002u, 0, p[0], p[1], p[2], p[3]);
        cpuid(0x80000003u, 0, p[4], p[5], p[6], p[7]);
        cpuid(0x80000004u, 0, p[8], p[9], p[10], p[11]);
        info.brand = brand;
        while (!info.brand.empty() && info.brand.front() == ' ') info.brand.erase(info.brand.begin());
    }
    cpuid(1, 0, a, b, c, d);
    info.aesni = (c & (1u << 25)) != 0;
    cpuid(7, 0, a, b, c, d);
    info.avx2 = (b & (1u << 5)) != 0;
#endif
    if (info.brand.empty()) info.brand = "Unknown CPU";
    return info;
}

// ---------------- OpenCL（委托给 src/ocl 的动态加载器） ----------------

static void detectGpus(HardwareReport& rep) {
    std::string err;
    if (!ocl::load(&err)) { rep.openclNote = err; return; }
    rep.openclAvailable = true;
    for (const auto& d : ocl::enumerateGpus()) {
        GpuDevice g;
        g.platform = d.platformName;
        g.name = d.name;
        g.vendor = d.vendor;
        g.computeUnits = d.computeUnits;
        g.clockMHz = d.clockMHz;
        g.globalMemBytes = d.globalMem;
        g.integrated = d.unifiedMemory;
        g.platformId = d.platform;
        g.deviceId = d.device;
        rep.gpus.push_back(std::move(g));
    }
    if (rep.gpus.empty()) rep.openclNote = "OpenCL loaded, but no GPU devices were enumerated";
}

HardwareReport detectHardware() {
    HardwareReport rep;
    rep.cpu = detectCpu();
    detectGpus(rep);
    rep.cudaGpus = cuda::enumerateGpus(&rep.cudaNote);
    return rep;
}
