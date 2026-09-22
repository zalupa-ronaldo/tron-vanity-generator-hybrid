#pragma once

#include "hwdetect.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cuda {
using id = void*;
bool compiled();
std::vector<GpuDevice> enumerateGpus(std::string* error);

// Native Driver API transport for the shared resident engine. Each operation
// binds its context because initialization and generation use different threads.
class Program {
public:
    Program();
    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    bool build(int ordinal, int rngMode, std::string* error);
    id kernel(const char* name, std::string* error);
    id buffer(unsigned long long flags, size_t bytes, void* host, std::string* error);
    bool setArg(id kernel, unsigned index, size_t bytes, const void* value);
    bool run1D(id kernel, size_t global, size_t local, std::string* error);
    bool read(id buffer, size_t bytes, void* dst);
    bool readAt(id buffer, size_t offset, size_t bytes, void* dst);
    bool write(id buffer, size_t bytes, const void* src);
    bool writeAt(id buffer, size_t offset, size_t bytes, const void* src);
    bool finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cuda
