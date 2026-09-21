#include "ocl.h"

#include <cstring>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
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
    HMODULE lib = LoadLibraryA("OpenCL.dll");
    if (!lib) { if (err) *err = "未找到 OpenCL.dll（安装最新 Intel/AMD 显卡驱动）"; return false; }
    auto G = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(lib, n)); };
    #define LD(var, name) *reinterpret_cast<void**>(&var) = G(name); if (!var) { if (err) *err = std::string("OpenCL.dll 缺少 ") + name; return false; }
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
    #undef LD
    g_loaded = true;
    return true;
#else
    if (err) *err = "仅支持 Windows";
    return false;
#endif
}

std::vector<DeviceInfo> enumerateGpus() {
    std::vector<DeviceInfo> out;
    if (!g_loaded) return out;
    cl_uint np = 0;
    if (pGetPlatformIDs(0, nullptr, &np) != 0 || np == 0) return out;
    std::vector<id> plats(np);
    pGetPlatformIDs(np, plats.data(), nullptr);
    for (id p : plats) {
        cl_uint nd = 0;
        if (pGetDeviceIDs(p, DEVICE_TYPE_GPU, 0, nullptr, &nd) != 0 || nd == 0) continue;
        std::vector<id> devs(nd);
        pGetDeviceIDs(p, DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
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

bool Program::build(id platform, id device, const std::string& source,
                    const std::string& opts, std::string* err) {
    (void)platform;
    cl_int e = 0;
    device_ = device;
    ctx_ = pCreateContext(nullptr, 1, &device, nullptr, nullptr, &e);
    if (!ctx_ || e != 0) { if (err) *err = "clCreateContext 失败 " + std::to_string(e); return false; }
    queue_ = pCreateCommandQueue(ctx_, device, 0, &e);
    if (!queue_ || e != 0) { if (err) *err = "clCreateCommandQueue 失败 " + std::to_string(e); return false; }
    const char* src = source.c_str();
    size_t len = source.size();
    program_ = pCreateProgramWithSource(ctx_, 1, &src, &len, &e);
    if (!program_ || e != 0) { if (err) *err = "clCreateProgramWithSource 失败 " + std::to_string(e); return false; }
    e = pBuildProgram(program_, 1, &device, opts.c_str(), nullptr, nullptr);
    if (e != 0) {
        std::vector<char> log(65536);
        size_t n = 0;
        pGetProgramBuildInfo(program_, device, PROGRAM_BUILD_LOG, log.size(), log.data(), &n);
        if (err) *err = "内核编译失败:\n" + std::string(log.data(), n ? n - 1 : 0);
        return false;
    }
    return true;
}

Program::~Program() {}  // 进程退出时由 OS 回收；不做细粒度释放

id Program::kernel(const char* name, std::string* err) {
    cl_int e = 0;
    id k = pCreateKernel(program_, name, &e);
    if (!k || e != 0) { if (err) *err = std::string("clCreateKernel(") + name + ") 失败 " + std::to_string(e); return nullptr; }
    return k;
}

id Program::buffer(unsigned long long flags, size_t bytes, void* host, std::string* err) {
    cl_int e = 0;
    id b = pCreateBuffer(ctx_, flags, bytes, host, &e);
    if (!b || e != 0) { if (err) *err = "clCreateBuffer 失败 " + std::to_string(e); return nullptr; }
    return b;
}

bool Program::setArg(id k, unsigned idx, size_t sz, const void* val) {
    return pSetKernelArg(k, idx, sz, val) == 0;
}

bool Program::run1D(id k, size_t global, size_t local, std::string* err) {
    const size_t* lp = local ? &local : nullptr;
    cl_int e = pEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, lp, 0, nullptr, nullptr);
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
void Program::release(id mem) { if (mem) pReleaseMemObject(mem); }

}  // namespace ocl
