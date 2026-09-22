#include "ocl.h"

#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace ocl {

namespace {

using cl_int = int;
using cl_uint = unsigned int;

// --- 函数指针 ---
cl_int (*pGetPlatformIDs)(cl_uint, id*, cl_uint*);
cl_int (*pGetPlatformInfo)(id, cl_uint, size_t, void*, size_t*);
cl_int (*pGetDeviceIDs)(id, unsigned long long, cl_uint, id*, cl_uint*);
cl_int (*pGetDeviceInfo)(id, cl_uint, size_t, void*, size_t*);
id (*pCreateContext)(const intptr_t*, cl_uint, const id*, void*, void*, cl_int*);
id (*pCreateCommandQueue)(id, id, unsigned long long, cl_int*);
id (*pCreateProgramWithSource)(id, cl_uint, const char**, const size_t*, cl_int*);
cl_int (*pBuildProgram)(id, cl_uint, const id*, const char*, void*, void*);
cl_int (*pGetProgramBuildInfo)(id, id, cl_uint, size_t, void*, size_t*);
id (*pCreateKernel)(id, const char*, cl_int*);
id (*pCreateBuffer)(id, unsigned long long, size_t, void*, cl_int*);
cl_int (*pSetKernelArg)(id, cl_uint, size_t, const void*);
cl_int (*pEnqueueNDRangeKernel)(id, id, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
cl_int (*pEnqueueReadBuffer)(id, id, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
cl_int (*pEnqueueWriteBuffer)(id, id, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
cl_int (*pFinish)(id);
cl_int (*pReleaseMemObject)(id);
cl_int (*pReleaseKernel)(id);
cl_int (*pReleaseProgram)(id);
cl_int (*pReleaseCommandQueue)(id);
cl_int (*pReleaseContext)(id);
cl_int (*pReleaseEvent)(id);
cl_int (*pGetEventProfilingInfo)(id, cl_uint, size_t, void*, size_t*);
cl_int (*pGetKernelWorkGroupInfo)(id, id, cl_uint, size_t, void*, size_t*);

bool g_loaded = false;

constexpr cl_uint PLATFORM_NAME = 0x0902;
constexpr cl_uint DEVICE_NAME = 0x102B;
constexpr cl_uint DEVICE_VENDOR = 0x102C;
constexpr cl_uint DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_uint DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
constexpr cl_uint DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_uint DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
constexpr cl_uint PROGRAM_BUILD_LOG = 0x1183;
constexpr unsigned long long DEVICE_TYPE_GPU = 1ull << 2;
constexpr cl_uint TRUE_ = 1;

std::string infoStr(id obj, cl_uint param, bool device) {
    char buf[1024] = {0};
    size_t n = 0;
    if (device) { if (pGetDeviceInfo(obj, param, sizeof(buf), buf, &n) != 0) return ""; }
    else        { if (pGetPlatformInfo(obj, param, sizeof(buf), buf, &n) != 0) return ""; }
    return std::string(buf);
}

}  // namespace

bool loaded() { return g_loaded; }

bool load(std::string* err) {
    if (g_loaded) return true;
#if defined(_WIN32)
    HMODULE lib = LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) { if (err) *err = "未找到 OpenCL.dll（安装最新 Intel/AMD 显卡驱动）"; return false; }
    auto G = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(lib, n)); };
#else
#if defined(__APPLE__)
    void* lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW | RTLD_LOCAL);
#else
    void* lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!lib) { if (err) *err = "OpenCL runtime not found"; return false; }
    auto G = [&](const char* n) { return dlsym(lib, n); };
#endif
    #define LD(var, name) var = reinterpret_cast<decltype(var)>(G(name)); if (!var) { if (err) *err = std::string("OpenCL missing ") + name; return false; }
    LD(pGetPlatformIDs, "clGetPlatformIDs");
    LD(pGetPlatformInfo, "clGetPlatformInfo");
    LD(pGetDeviceIDs, "clGetDeviceIDs");
    LD(pGetDeviceInfo, "clGetDeviceInfo");
    LD(pCreateContext, "clCreateContext");
    LD(pCreateCommandQueue, "clCreateCommandQueue");
    LD(pCreateProgramWithSource, "clCreateProgramWithSource");
    LD(pBuildProgram, "clBuildProgram");
    LD(pGetProgramBuildInfo, "clGetProgramBuildInfo");
    LD(pCreateKernel, "clCreateKernel");
    LD(pCreateBuffer, "clCreateBuffer");
    LD(pSetKernelArg, "clSetKernelArg");
    LD(pEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    LD(pEnqueueReadBuffer, "clEnqueueReadBuffer");
    LD(pEnqueueWriteBuffer, "clEnqueueWriteBuffer");
    LD(pFinish, "clFinish");
    LD(pReleaseMemObject, "clReleaseMemObject");
    LD(pReleaseKernel, "clReleaseKernel");
    LD(pReleaseProgram, "clReleaseProgram");
    LD(pReleaseCommandQueue, "clReleaseCommandQueue");
    LD(pReleaseContext, "clReleaseContext");
    LD(pReleaseEvent, "clReleaseEvent");
    LD(pGetEventProfilingInfo, "clGetEventProfilingInfo");
    LD(pGetKernelWorkGroupInfo, "clGetKernelWorkGroupInfo");
    #undef LD
    g_loaded = true;
    return true;
}

static std::vector<DeviceInfo> enumerateDevices(unsigned long long type) {
    std::vector<DeviceInfo> out;
    if (!g_loaded) return out;
    cl_uint np = 0;
    if (pGetPlatformIDs(0, nullptr, &np) != 0 || np == 0) return out;
    std::vector<id> plats(np);
    pGetPlatformIDs(np, plats.data(), nullptr);
    for (id p : plats) {
        cl_uint nd = 0;
        if (pGetDeviceIDs(p, type, 0, nullptr, &nd) != 0 || nd == 0) continue;
        std::vector<id> devs(nd);
        pGetDeviceIDs(p, type, nd, devs.data(), nullptr);
        for (id d : devs) {
            DeviceInfo di;
            di.platform = p;
            di.device = d;
            di.platformName = infoStr(p, PLATFORM_NAME, false);
            di.name = infoStr(d, DEVICE_NAME, true);
            di.vendor = infoStr(d, DEVICE_VENDOR, true);
            cl_uint cu = 0, mhz = 0, uni = 0;
            unsigned long long mem = 0;
            pGetDeviceInfo(d, DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            pGetDeviceInfo(d, DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
            pGetDeviceInfo(d, DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
            pGetDeviceInfo(d, DEVICE_HOST_UNIFIED_MEMORY, sizeof(uni), &uni, nullptr);
            di.computeUnits = cu; di.clockMHz = mhz; di.globalMem = mem; di.unifiedMemory = uni != 0;
            out.push_back(std::move(di));
        }
    }
    return out;
}

std::vector<DeviceInfo> enumerateGpus() { return enumerateDevices(DEVICE_TYPE_GPU); }
std::vector<DeviceInfo> enumerateTestDevices() { return enumerateDevices(0xffffffffULL); }

std::string deviceDescription(id device) {
    if (!g_loaded) return "OpenCL not loaded";
    return infoStr(device, DEVICE_NAME, true) + " / " + infoStr(device, DEVICE_VENDOR, true) +
        "; driver " + infoStr(device, 0x102D, true) + "; " + infoStr(device, 0x102F, true) +
        "; " + infoStr(device, 0x103D, true);
}

bool Program::build(id platform, id device, const std::string& source,
                    const std::string& opts, std::string* err, bool profiling, bool trace) {
    if (ctx_ || !load(err)) return false;
    profiling_ = profiling;
    trace_ = trace;
    const auto started = std::chrono::steady_clock::now();
    auto mark = [&](const char* message) {
        if (trace_) std::cerr << "\n  OpenCL API +"
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count()
            << " ms: " << message << std::endl;
    };
    if (trace_) std::cerr << "\n  " << deviceDescription(device) << "\n  Build options: " << opts << std::endl;
    cl_int e = 0;
    device_ = device;
    // Bind the exact enumerated ICD platform rather than leaving selection
    // implementation-defined on systems with several vendor runtimes.
    const intptr_t properties[] = {0x1084, reinterpret_cast<intptr_t>(platform), 0};
    mark("clCreateContext BEGIN");
    ctx_ = pCreateContext(platform ? properties : nullptr, 1, &device, nullptr, nullptr, &e);
    mark("clCreateContext returned");
    if (!ctx_ || e != 0) { if (err) *err = "clCreateContext 失败 " + std::to_string(e); return false; }
    mark("clCreateCommandQueue BEGIN");
    queue_ = pCreateCommandQueue(ctx_, device, profiling ? 2ULL : 0ULL, &e);
    mark("clCreateCommandQueue returned");
    if (!queue_ || e != 0) { if (err) *err = "clCreateCommandQueue 失败 " + std::to_string(e); return false; }
    return buildAdditional(source, opts, err);
}

bool Program::buildAdditional(const std::string& source, const std::string& opts, std::string* err) {
    if (!ctx_ || !queue_) { if (err) *err = "OpenCL context is not initialized"; return false; }
    const auto started = std::chrono::steady_clock::now();
    auto mark = [&](const char* message) {
        if (trace_) std::cerr << "\n  OpenCL build +"
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count()
            << " ms: " << message << std::endl;
    };
    if (trace_) std::cerr << "  Program options: " << opts << std::endl;
    cl_int e = 0;
    const char* src = source.c_str();
    size_t len = source.size();
    mark("clCreateProgramWithSource BEGIN");
    program_ = pCreateProgramWithSource(ctx_, 1, &src, &len, &e);
    mark("clCreateProgramWithSource returned");
    if (!program_ || e != 0) { if (err) *err = "clCreateProgramWithSource 失败 " + std::to_string(e); return false; }
    programs_.push_back(program_);
    mark("clBuildProgram BEGIN");
    e = pBuildProgram(program_, 1, &device_, opts.c_str(), nullptr, nullptr);
    mark("clBuildProgram returned");
    if (e != 0) {
        std::vector<char> log(65536);
        size_t n = 0;
        pGetProgramBuildInfo(program_, device_, PROGRAM_BUILD_LOG, log.size(), log.data(), &n);
        n = std::min(n, log.size());
        if (err) *err = "内核编译失败:\n" + std::string(log.data(), n ? n - 1 : 0);
        return false;
    }
    return true;
}

Program::~Program() {
    if (queue_) pFinish(queue_);
    for (id event : timingEvents_) pReleaseEvent(event);
    for (id k : kernels_) pReleaseKernel(k);
    for (id b : buffers_) pReleaseMemObject(b);
    for (id program : programs_) pReleaseProgram(program);
    if (queue_) pReleaseCommandQueue(queue_);
    if (ctx_) pReleaseContext(ctx_);
}

id Program::kernel(const char* name, std::string* err) {
    cl_int e = 0;
    if (trace_) std::cerr << "  OpenCL API: clCreateKernel(" << name << ") BEGIN" << std::endl;
    id k = pCreateKernel(program_, name, &e);
    if (trace_) std::cerr << "  OpenCL API: clCreateKernel returned " << e << std::endl;
    if (!k || e != 0) { if (err) *err = std::string("clCreateKernel(") + name + ") 失败 " + std::to_string(e); return nullptr; }
    kernels_.push_back(k);
    return k;
}

id Program::buffer(unsigned long long flags, size_t bytes, void* host, std::string* err) {
    cl_int e = 0;
    id b = pCreateBuffer(ctx_, flags, bytes, host, &e);
    if (!b || e != 0) { if (err) *err = "clCreateBuffer 失败 " + std::to_string(e); return nullptr; }
    buffers_.push_back(b);
    return b;
}

bool Program::setArg(id k, unsigned idx, size_t sz, const void* val) {
    return pSetKernelArg(k, idx, sz, val) == 0;
}

bool Program::run1D(id k, size_t global, size_t local, std::string* err, bool appendTiming) {
    if (!appendTiming) {
        for (id event : timingEvents_) pReleaseEvent(event);
        timingEvents_.clear();
    }
    id event = nullptr;
    const size_t* lp = local ? &local : nullptr;
    cl_int e = pEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, lp, 0, nullptr,
                                   profiling_ ? &event : nullptr);
    if (event) timingEvents_.push_back(event);
    if (e != 0) { if (err) *err = "clEnqueueNDRangeKernel 失败 " + std::to_string(e); return false; }
    return true;
}

bool Program::read(id buf, size_t bytes, void* dst) {
    return readAt(buf, 0, bytes, dst);
}
bool Program::readAt(id buf, size_t offset, size_t bytes, void* dst) {
    return pEnqueueReadBuffer(queue_, buf, TRUE_, offset, bytes, dst, 0, nullptr, nullptr) == 0;
}
bool Program::write(id buf, size_t bytes, const void* src) {
    return writeAt(buf, 0, bytes, src);
}
bool Program::writeAt(id buf, size_t offset, size_t bytes, const void* src) {
    return pEnqueueWriteBuffer(queue_, buf, TRUE_, offset, bytes, src, 0, nullptr, nullptr) == 0;
}
bool Program::finish() { return pFinish(queue_) == 0; }
void Program::release(id mem) {
    auto it = std::find(buffers_.begin(), buffers_.end(), mem);
    if (it != buffers_.end()) { pReleaseMemObject(mem); buffers_.erase(it); }
}

bool Program::lastKernelMilliseconds(double& ms, std::vector<double>* eventMs) const {
    if (timingEvents_.empty()) return false;
    ms = 0;
    if (eventMs) { eventMs->clear(); eventMs->reserve(timingEvents_.size()); }
    for (id event : timingEvents_) {
        unsigned long long start = 0, end = 0;
        if (pGetEventProfilingInfo(event, 0x1282, sizeof(start), &start, nullptr) ||
            pGetEventProfilingInfo(event, 0x1283, sizeof(end), &end, nullptr) || end <= start) return false;
        const double duration = static_cast<double>(end - start) / 1e6;
        ms += duration;
        if (eventMs) eventMs->push_back(duration);
    }
    return true;
}

bool Program::kernelLimits(id kernel, size_t& maxGroup, size_t& preferredMultiple,
                           unsigned long long& privateBytes) const {
    return pGetKernelWorkGroupInfo(kernel, device_, 0x11B0, sizeof(maxGroup), &maxGroup, nullptr) == 0 &&
           pGetKernelWorkGroupInfo(kernel, device_, 0x11B3, sizeof(preferredMultiple), &preferredMultiple, nullptr) == 0 &&
           pGetKernelWorkGroupInfo(kernel, device_, 0x11B4, sizeof(privateBytes), &privateBytes, nullptr) == 0;
}

}  // namespace ocl
