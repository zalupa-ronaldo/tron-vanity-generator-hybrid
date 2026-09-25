#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "matcher.h"
#include "dictionary.h"

struct RunConfig {
    int minLen = 5;
    unsigned int threads = 0;      // 0 = 自动
    uint64_t maxAttempts = 0;      // 0 = 无限
    uint64_t seconds = 60;         // 0 = until Ctrl+C
    bool verbose = false;
    // Production searches report every matching address by default. The
    // legacy one-result-per-word behavior is opt-in for bounded runs.
    bool uniquePerWord = false;
    std::shared_ptr<const Dictionary> dictionary;
};

struct FoundKey {
    std::string address;
    std::string privHex;
    MatchResult match;
    std::vector<std::string> words;
};

// 全局运行状态（所有 backend 共享）
struct RunState {
    std::atomic<uint64_t> checked{0};
    std::atomic<uint64_t> cpuChecked{0};
    std::atomic<uint64_t> gpuChecked{0};
    std::atomic<uint64_t> found{0};
    std::atomic<uint64_t> uniqueWords{0};
    std::atomic<bool> stop{false};
};

using ReportFn = std::function<void(const FoundKey&)>;

// 供报告展示用的多行硬件信息。
struct BackendInfo {
    std::string kind;               // "CPU" / "GPU"
    std::string title;              // 型号，如 "12th Gen Intel(R) Core(TM) i5-12400"
    std::vector<std::string> lines; // 细节行，如 {"12 threads", "AVX2   OK"}
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;         // 简短标签，如 "CPU" / "OpenCL GPU"
    virtual bool available() const = 0;
    virtual std::string note() const { return ""; }  // 不可用/受限原因
    virtual BackendInfo info() const = 0;

    // 运行约 seconds 秒的压测，返回 keys/sec；不写文件、不触发回调
    virtual double benchmark(double seconds) = 0;

    // 主运行循环；持续调用 report 直到 state.stop 或达到 maxAttempts
    virtual void run(const RunConfig& cfg, RunState& state, const ReportFn& report) = 0;
};

struct GpuDevice;  // hwdetect.h
std::unique_ptr<Backend> makeCpuBackend();
std::unique_ptr<Backend> makeGpuBackend(const GpuDevice& dev, std::shared_ptr<const Dictionary> dictionary);
inline std::unique_ptr<Backend> makeGpuBackend(const GpuDevice& dev) {
    return makeGpuBackend(dev, nullptr);
}
