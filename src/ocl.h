#pragma once
// 运行期动态加载的极简 OpenCL 封装（无需 OpenCL SDK / 头文件 / 导入库）。
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ocl {

using id = void*;
constexpr unsigned long long MEM_READ_ONLY = 1ull << 2;
constexpr unsigned long long MEM_WRITE_ONLY = 1ull << 1;
constexpr unsigned long long MEM_READ_WRITE = 1ull << 0;
constexpr unsigned long long MEM_COPY_HOST_PTR = 1ull << 5;
constexpr unsigned long long MEM_ALLOC_HOST_PTR = 1ull << 4;

bool load(std::string* err);          // 加载 OpenCL.dll，解析符号
bool loaded();

struct DeviceInfo {
    id platform = nullptr;
    id device = nullptr;
    std::string platformName, name, vendor;
    unsigned computeUnits = 0, clockMHz = 0;
    unsigned long long globalMem = 0;
    bool unifiedMemory = false;
};

std::vector<DeviceInfo> enumerateGpus();

// 一个设备 + context + queue + program 的薄封装。
class Program {
public:
    bool build(id platform, id device, const std::string& source,
               const std::string& opts, std::string* err);
    ~Program();

    id kernel(const char* name, std::string* err);
    id buffer(unsigned long long flags, size_t bytes, void* host, std::string* err);
    bool setArg(id k, unsigned idx, size_t sz, const void* val);
    bool run1D(id k, size_t global, size_t local, std::string* err);   // local=0 → 让实现选
    bool read(id buf, size_t bytes, void* dst);
    bool readAt(id buf, size_t offset, size_t bytes, void* dst);
    bool write(id buf, size_t bytes, const void* src);
    bool writeAt(id buf, size_t offset, size_t bytes, const void* src);
    bool finish();
    void release(id mem);

private:
    id ctx_ = nullptr, queue_ = nullptr, program_ = nullptr, device_ = nullptr;
};

}  // namespace ocl
