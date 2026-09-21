#include "backend.h"
#include "crypto.h"
#include "hwdetect.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

int selftest();
int hashtest();
int matchtest();
int gpuSelfTest(const GpuDevice& dev);
int gpuBench(const GpuDevice& dev, double secs);
void gpuSetKeysPerItem(uint32_t n);
void gpuSetEcWindow(uint32_t w);
void gpuSetMontN(uint32_t n);

namespace {

RunState* gRunState = nullptr;
void onSigint(int) { if (gRunState) gRunState->stop.store(true); }

struct Options {
    uint64_t seconds = 60;
    uint64_t maxAttempts = 0;
    unsigned threads = 0;
    std::string words = "words.txt";
    std::string out = "results";
    std::string output;
    std::string backend = "auto";
    bool caseSensitive = false;
    bool verbose = false;
    bool list = false;
    uint32_t keysPerItem = 1;
    uint32_t ecWindow = 0;
    uint32_t montN = 0;
};

void usage() {
    std::cout <<
        "TRON vanity generator - CPU + AMD/NVIDIA OpenCL\n"
        "  --seconds N       run duration; 0 = until Ctrl+C\n"
        "  --threads N       CPU threads; default = logical cores\n"
        "  --words FILE      dictionary; default words.txt\n"
        "  --out DIR         output directory; default results\n"
        "  --output FILE     direct JSONL output override\n"
        "  --backend auto|cpu|opencl\n"
        "  --case-sensitive  exact case matching\n"
        "  --list            list CPU/OpenCL devices and exit\n"
        "  --verbose         more frequent progress updates\n"
        "  --selftest | --hashtest | --matchtest | --gputest | --bench\n";
}

bool parse(int argc, char** argv, Options& o) {
    auto next = [&](int& i, const char* name) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
        return argv[++i];
    };
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--help" || a == "-h") { usage(); return false; }
            if (a == "--seconds") o.seconds = std::stoull(next(i, "--seconds"));
            else if (a == "--threads") o.threads = std::stoul(next(i, "--threads"));
            else if (a == "--words") o.words = next(i, "--words");
            else if (a == "--out") o.out = next(i, "--out");
            else if (a == "--output") o.output = next(i, "--output");
            else if (a == "--max") o.maxAttempts = std::stoull(next(i, "--max"));
            else if (a == "--backend") o.backend = next(i, "--backend");
            else if (a == "--case-sensitive") o.caseSensitive = true;
            else if (a == "--verbose") o.verbose = true;
            else if (a == "--list") o.list = true;
            else if (a == "--keys-per-item") o.keysPerItem = std::stoul(next(i, "--keys-per-item"));
            else if (a == "--ec-window") o.ecWindow = std::stoul(next(i, "--ec-window"));
            else if (a == "--mont-n") o.montN = std::stoul(next(i, "--mont-n"));
            else if (a == "--backend") o.backend = next(i, "--backend");
            else if (a == "--selftest" || a == "--hashtest" || a == "--matchtest" || a == "--gputest" || a == "--bench") {
                // handled by main's early command dispatch
            } else { std::cerr << "unknown option: " << a << "\n"; usage(); return false; }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"; return false;
    }
    if (o.backend == "gpu") o.backend = "opencl";
    if (o.backend != "auto" && o.backend != "cpu" && o.backend != "opencl") {
        std::cerr << "backend must be auto, cpu, or opencl\n"; return false;
    }
    return true;
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
        else if (c >= 0x20) out.push_back(static_cast<char>(c));
    }
    return out;
}

class JsonSink {
public:
    explicit JsonSink(std::string path) : path_(std::move(path)) {}

    void operator()(const FoundKey& key) {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::string> fresh;
        for (const auto& word : key.words) if (seen_.insert(word).second) fresh.push_back(word);
        if (fresh.empty()) return;
        std::ofstream out(path_, std::ios::app | std::ios::binary);
        if (!out) { std::cerr << "cannot write " << path_ << "\n"; return; }
        out << "{\"address\":\"" << jsonEscape(key.address) << "\",\"words\":[";
        for (size_t i = 0; i < fresh.size(); ++i) {
            if (i) out << ',';
            out << "\"" << jsonEscape(fresh[i]) << "\"";
        }
        out << "],\"private_key\":\"" << jsonEscape(key.privHex) << "\"}\n";
        out.flush();
        std::cout << "\nFOUND " << key.address << " words=" << fresh.size() << "\n";
    }

private:
    std::string path_;
    std::mutex mu_;
    std::unordered_set<std::string> seen_;
};

std::string rate(uint64_t n, double seconds) {
    double r = seconds > 0 ? static_cast<double>(n) / seconds : 0;
    std::ostringstream s;
    if (r >= 1e6) s << std::fixed << std::setprecision(2) << r / 1e6 << " M/s";
    else if (r >= 1e3) s << std::fixed << std::setprecision(1) << r / 1e3 << " K/s";
    else s << static_cast<uint64_t>(r) << " /s";
    return s.str();
}

void printDevices(const HardwareReport& hw) {
    auto ci = detectCpu();
    std::cout << "CPU: " << ci.brand << " (" << std::thread::hardware_concurrency() << " threads)\n";
    if (hw.gpus.empty()) std::cout << "OpenCL GPU: none (" << hw.openclNote << ")\n";
    for (const auto& g : hw.gpus) {
        std::cout << "OpenCL GPU: " << g.name << " / " << g.vendor << " / "
                  << g.computeUnits << " CU @ " << g.clockMHz << " MHz\n";
    }
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") return selftest();
        if (a == "--hashtest") return hashtest();
        if (a == "--matchtest") return matchtest();
    }
    Options opt;
    if (!parse(argc, argv, opt)) return 0;
    HardwareReport hw = detectHardware();
    if (opt.list) { printDevices(hw); return 0; }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gputest") {
            if (hw.gpus.empty()) { std::cerr << "no OpenCL GPU: " << hw.openclNote << "\n"; return 1; }
            return gpuSelfTest(hw.gpus.front());
        }
        if (std::string(argv[i]) == "--bench") {
            if (hw.gpus.empty()) { std::cerr << "no OpenCL GPU: " << hw.openclNote << "\n"; return 1; }
            return gpuBench(hw.gpus.front(), 5.0);
        }
    }

    std::string error;
    auto dictionary = Dictionary::load(opt.words, opt.caseSensitive, &error);
    if (!dictionary) { std::cerr << error << "\n"; return 1; }
    gpuSetKeysPerItem(opt.keysPerItem);
    gpuSetEcWindow(opt.ecWindow);
    gpuSetMontN(opt.montN);

    std::vector<std::unique_ptr<Backend>> backends;
    if (opt.backend == "auto" || opt.backend == "cpu") backends.push_back(makeCpuBackend());
    if (opt.backend == "auto" || opt.backend == "opencl") {
        for (const auto& g : hw.gpus) backends.push_back(makeGpuBackend(g, dictionary));
        if (hw.gpus.empty() && opt.backend == "opencl") {
            std::cerr << "OpenCL unavailable; falling back to CPU\n";
            backends.push_back(makeCpuBackend());
        }
    }
    if (backends.empty()) { std::cerr << "no backend available\n"; return 1; }

    for (auto& b : backends) {
        auto info = b->info();
        std::cout << info.kind << ": " << info.title << "\n";
        for (const auto& line : info.lines) std::cout << "  " << line << "\n";
        if (!b->available()) std::cerr << "  unavailable: " << b->note() << "\n";
    }

    std::filesystem::path output;
    if (!opt.output.empty()) output = opt.output;
    else {
        std::filesystem::create_directories(opt.out);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        output = std::filesystem::path(opt.out) / ("wallets-" + std::to_string(ns) + ".jsonl");
    }
    JsonSink sink(output.string());
    ReportFn report = [&sink](const FoundKey& key) { sink(key); };
    RunConfig cfg;
    cfg.threads = opt.threads;
    cfg.maxAttempts = opt.maxAttempts;
    cfg.seconds = opt.seconds;
    cfg.verbose = opt.verbose;
    cfg.dictionary = dictionary;
    RunState state;
    gRunState = &state;
    std::signal(SIGINT, onSigint);

    std::cout << "\nDictionary: " << dictionary->words.size() << " words\n"
              << "CPU threads: " << (opt.threads ? opt.threads : std::thread::hardware_concurrency()) << "\n"
              << "Output: " << output.string() << "\n";

    auto started = std::chrono::steady_clock::now();
    std::atomic<bool> finished{false};
    std::thread timer([&] {
        while (!finished.load()) {
            if (opt.seconds && std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - started).count() >= opt.seconds) {
                state.stop.store(true); break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    std::thread progress([&] {
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (finished.load() || state.stop.load()) break;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            uint64_t cpu = state.cpuChecked.load(), gpu = state.gpuChecked.load();
            double pct = opt.seconds ? std::min(100.0, elapsed * 100.0 / opt.seconds) : 0.0;
            std::cout << "\r[" << std::fixed << std::setprecision(1) << pct << "%] "
                      << "CPU " << rate(cpu, elapsed) << " (" << (opt.threads ? opt.threads : std::thread::hardware_concurrency())
                      << "T) | GPU " << rate(gpu, elapsed) << " | total " << rate(cpu + gpu, elapsed)
                      << " | matches " << state.found.load() << std::flush;
            if (state.stop.load()) break;
        }
    });

    std::vector<std::thread> workers;
    for (auto& backend : backends) {
        if (!backend->available()) continue;
        workers.emplace_back([&, b = backend.get()] { b->run(cfg, state, report); });
    }
    for (auto& worker : workers) worker.join();
    state.stop.store(true);
    finished.store(true);
    timer.join();
    progress.join();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "\nDone: " << state.checked.load() << " keys, " << rate(state.checked.load(), elapsed)
              << ", CPU=" << rate(state.cpuChecked.load(), elapsed)
              << ", GPU=" << rate(state.gpuChecked.load(), elapsed) << "\n";
    return 0;
}
