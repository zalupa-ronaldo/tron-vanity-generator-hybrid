#include "cuda_driver.h"
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define CUDA_CALL
#endif
#ifdef TRON_ENABLE_CUDA
#include "cuda_ptx.h"
#endif

namespace cuda {
namespace {
using DevicePtr = unsigned long long;
static_assert(sizeof(void*) == sizeof(DevicePtr), "CUDA requires a 64-bit build");

struct Driver {
    int (CUDA_CALL *init)(unsigned) = nullptr;
    int (CUDA_CALL *count)(int*) = nullptr;
    int (CUDA_CALL *device)(int*, int) = nullptr;
    int (CUDA_CALL *name)(char*, int, int) = nullptr;
    int (CUDA_CALL *attribute)(int*, int, int) = nullptr;
    int (CUDA_CALL *memory)(size_t*, int) = nullptr;
    int (CUDA_CALL *retain)(id*, int) = nullptr;
    int (CUDA_CALL *release)(int) = nullptr;
    int (CUDA_CALL *current)(id) = nullptr;
    int (CUDA_CALL *moduleLoad)(id*, const void*) = nullptr;
    int (CUDA_CALL *moduleUnload)(id) = nullptr;
    int (CUDA_CALL *function)(id*, id, const char*) = nullptr;
    int (CUDA_CALL *alloc)(DevicePtr*, size_t) = nullptr;
    int (CUDA_CALL *free)(DevicePtr) = nullptr;
    int (CUDA_CALL *htod)(DevicePtr, const void*, size_t) = nullptr;
    int (CUDA_CALL *dtoh)(void*, DevicePtr, size_t) = nullptr;
    int (CUDA_CALL *launch)(id, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, id, void**, void**) = nullptr;
    int (CUDA_CALL *streamCreate)(id*, unsigned) = nullptr;
    int (CUDA_CALL *streamDestroy)(id) = nullptr;
    int (CUDA_CALL *streamSync)(id) = nullptr;
    int (CUDA_CALL *errorString)(int, const char**) = nullptr;
    std::once_flag once;
    std::string failure;

    bool load(std::string* error) {
        std::call_once(once, [&] {
#if defined(_WIN32)
            auto lib = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            auto symbol = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(lib, n)); };
#else
            auto lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
            auto symbol = [&](const char* n) { return dlsym(lib, n); };
#endif
            if (!lib) { failure = "NVIDIA CUDA driver not found"; return; }
#define LOAD(member, symbolName) member = reinterpret_cast<decltype(member)>(symbol(symbolName)); \
    if (!member) { failure = "NVIDIA driver is missing " symbolName; return; }
            LOAD(init, "cuInit");
            LOAD(count, "cuDeviceGetCount");
            LOAD(device, "cuDeviceGet");
            LOAD(name, "cuDeviceGetName");
            LOAD(attribute, "cuDeviceGetAttribute");
            LOAD(memory, "cuDeviceTotalMem_v2");
            LOAD(retain, "cuDevicePrimaryCtxRetain");
            LOAD(release, "cuDevicePrimaryCtxRelease_v2");
            LOAD(current, "cuCtxSetCurrent");
            LOAD(moduleLoad, "cuModuleLoadData");
            LOAD(moduleUnload, "cuModuleUnload");
            LOAD(function, "cuModuleGetFunction");
            LOAD(alloc, "cuMemAlloc_v2");
            LOAD(free, "cuMemFree_v2");
            LOAD(htod, "cuMemcpyHtoD_v2");
            LOAD(dtoh, "cuMemcpyDtoH_v2");
            LOAD(launch, "cuLaunchKernel");
            LOAD(streamCreate, "cuStreamCreate");
            LOAD(streamDestroy, "cuStreamDestroy_v2");
            LOAD(streamSync, "cuStreamSynchronize");
            LOAD(errorString, "cuGetErrorString");
#undef LOAD
            const int result = init(0);
            if (result) failure = "cuInit: " + describe(result);
            // The library remains loaded for the process lifetime. Contexts,
            // streams, modules and allocations have per-Program ownership.
        });
        if (!failure.empty() && error) *error = failure;
        return failure.empty();
    }
    std::string describe(int result) const {
        const char* message = nullptr;
        if (errorString) errorString(result, &message);
        return std::string(message ? message : "CUDA error") + " (" + std::to_string(result) + ")";
    }
};
Driver& driver() { static Driver value; return value; }
DevicePtr address(id buffer) { return static_cast<DevicePtr>(reinterpret_cast<uintptr_t>(buffer)); }
} // namespace

bool compiled() {
#ifdef TRON_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

std::vector<GpuDevice> enumerateGpus(std::string* error) {
    std::vector<GpuDevice> devices;
    if (!compiled()) { if (error) *error = "CUDA kernels not included; rebuild with TRON_ENABLE_CUDA=ON"; return devices; }
    auto& api = driver();
    if (!api.load(error)) return devices;
    int count = 0;
    int result = api.count(&count);
    if (result) { if (error) *error = "cuDeviceGetCount: " + api.describe(result); return devices; }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        int device = 0, major = 0, minor = 0, sms = 0, clock = 0;
        char name[256]{};
        size_t bytes = 0;
        if (api.device(&device, ordinal) || api.name(name, sizeof(name), device) ||
            api.attribute(&major, 75, device) || api.attribute(&minor, 76, device) ||
            api.attribute(&sms, 16, device) || api.attribute(&clock, 13, device) ||
            api.memory(&bytes, device)) continue;
        if (major * 10 + minor < 52) continue;
        GpuDevice gpu;
        gpu.platform = "NVIDIA CUDA Driver API";
        gpu.name = name;
        gpu.vendor = "NVIDIA";
        gpu.computeUnits = static_cast<unsigned>(sms);
        gpu.clockMHz = static_cast<unsigned>(clock / 1000);
        gpu.globalMemBytes = bytes;
        gpu.cudaOrdinal = ordinal;
        gpu.cudaCapability = major * 10 + minor;
        devices.push_back(std::move(gpu));
    }
    if (devices.empty() && error) *error = "no NVIDIA CUDA device with compute capability 5.2+";
    return devices;
}

struct Program::Impl {
    int device = -1;
    id context = nullptr, module = nullptr, stream = nullptr;
    std::string* error = nullptr;
    std::unordered_map<id, size_t> buffers;
    // Driver argument addresses point to stable, suitably aligned host values.
    struct Argument { alignas(16) std::array<unsigned char, 16> value{}; size_t size = 0; };
    std::unordered_map<id, std::vector<Argument>> arguments;
    bool check(int code, const char* operation) {
        if (code && error) *error = std::string(operation) + ": " + driver().describe(code);
        return code == 0;
    }
    bool activate() { return context && check(driver().current(context), "cuCtxSetCurrent"); }
    bool range(id buffer, size_t offset, size_t bytes) {
        auto it = buffers.find(buffer);
        if (it != buffers.end() && offset <= it->second && bytes <= it->second - offset) return true;
        if (error) *error = "CUDA buffer access out of bounds";
        return false;
    }
    ~Impl() {
        if (!context) return;
        auto& api = driver();
        if (api.current(context) == 0) {
            if (stream) api.streamSync(stream);
            for (const auto& entry : buffers) api.free(address(entry.first));
            if (module) api.moduleUnload(module);
            if (stream) api.streamDestroy(stream);
        }
        api.release(device);
    }
};

Program::Program() : impl_(std::make_unique<Impl>()) {}
Program::~Program() = default;

bool Program::build(int ordinal, int rngMode, std::string* error) {
    auto& p = *impl_;
    p.error = error;
#ifndef TRON_ENABLE_CUDA
    if (error) *error = "CUDA kernels not included in this build";
    return false;
#else
    if (rngMode < 1 || rngMode > 3 || p.context) return false;
    auto& api = driver();
    if (!api.load(error) || !p.check(api.device(&p.device, ordinal), "cuDeviceGet") ||
        !p.check(api.retain(&p.context, p.device), "cuDevicePrimaryCtxRetain") || !p.activate() ||
        !p.check(api.streamCreate(&p.stream, 0), "cuStreamCreate")) return false;
    std::string ptx;
    for (auto part = kCudaPtx[rngMode - 1]; *part; ++part) ptx += *part;
    if (!p.check(api.moduleLoad(&p.module, ptx.c_str()), "cuModuleLoadData")) {
        if (error) *error += "; update the NVIDIA driver to one supporting CUDA 12.6 PTX";
        return false;
    }
    return true;
#endif
}

id Program::kernel(const char* name, std::string* error) {
    impl_->error = error;
    id function = nullptr;
    if (!impl_->activate() || !impl_->check(driver().function(&function, impl_->module, name), "cuModuleGetFunction")) return nullptr;
    impl_->arguments.emplace(function, std::vector<Impl::Argument>{});
    return function;
}
id Program::buffer(unsigned long long, size_t bytes, void* host, std::string* error) {
    impl_->error = error;
    DevicePtr ptr = 0;
    if (!impl_->activate() || !impl_->check(driver().alloc(&ptr, bytes), "cuMemAlloc")) return nullptr;
    id buffer = reinterpret_cast<id>(static_cast<uintptr_t>(ptr));
    impl_->buffers.emplace(buffer, bytes);
    if (host && !write(buffer, bytes, host)) return nullptr;
    return buffer;
}
bool Program::setArg(id kernel, unsigned index, size_t bytes, const void* value) {
    auto it = impl_->arguments.find(kernel);
    if (it == impl_->arguments.end() || !value || !bytes || bytes > 16 || index > 31) return false;
    auto& args = it->second;
    if (args.size() <= index) args.resize(index + 1);
    std::memcpy(args[index].value.data(), value, bytes);
    args[index].size = bytes;
    return true;
}
bool Program::run1D(id kernel, size_t global, size_t local, std::string* error) {
    impl_->error = error;
    auto it = impl_->arguments.find(kernel);
    if (it == impl_->arguments.end() || !local || !global || global % local || local > 1024 ||
        global / local > std::numeric_limits<unsigned>::max()) {
        if (error) *error = "invalid CUDA launch dimensions or kernel";
        return false;
    }
    std::vector<void*> args;
    for (auto& arg : it->second) {
        if (!arg.size) { if (error) *error = "missing CUDA kernel argument"; return false; }
        args.push_back(arg.value.data());
    }
    return impl_->activate() && impl_->check(driver().launch(kernel, static_cast<unsigned>(global / local), 1, 1,
        static_cast<unsigned>(local), 1, 1, 0, impl_->stream, args.data(), nullptr), "cuLaunchKernel");
}
bool Program::finish() {
    return impl_->activate() && impl_->check(driver().streamSync(impl_->stream), "cuStreamSynchronize");
}
bool Program::read(id buffer, size_t bytes, void* dst) { return readAt(buffer, 0, bytes, dst); }
bool Program::write(id buffer, size_t bytes, const void* src) { return writeAt(buffer, 0, bytes, src); }
bool Program::readAt(id buffer, size_t offset, size_t bytes, void* dst) {
    return impl_->range(buffer, offset, bytes) && finish() &&
        impl_->check(driver().dtoh(dst, address(buffer) + offset, bytes), "cuMemcpyDtoH");
}
bool Program::writeAt(id buffer, size_t offset, size_t bytes, const void* src) {
    return impl_->range(buffer, offset, bytes) && finish() &&
        impl_->check(driver().htod(address(buffer) + offset, src, bytes), "cuMemcpyHtoD");
}
} // namespace cuda
