#include "backend.h"
#include "crypto.h"
#include "hwdetect.h"
#include "metal_backend.h"
#include "metal_hardware_profile.h"
#include "resident_backend.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
void gpuSetBatch(uint32_t n);
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
    bool help = false;
    bool benchResidentOnly = false;
    bool openclProfile = false;
    std::string openclDiagnostic;
    OpenclResidentOptions openclOptions;
    bool metalProfileStages = false;
    int metalProfileStage = -1;
    bool metalProfileScalarKeccak = true;
    uint32_t metalProfileMaxThreads = 0;
    bool metalHardwareProfile = false;
    std::string metalHardwareCase = "all";
    std::string metalHardwareJson;
    uint32_t keysPerItem = 1;
    uint32_t gpuBatch = 0;
    uint32_t ecWindow = 0;
    uint32_t montN = 0;
    bool gpuResident = false;
    std::string gpuRng = "chacha12";
    uint32_t gpuBufferMiB = 128;
    uint32_t gpuChunkMs = 32;
    uint32_t gpuPollMs = 50;
    uint32_t gpuGroupSize = 256;
    uint32_t metalKeysPerLane = 32;
    double benchSeconds = 1.0;
};

void usage() {
    std::cout <<
        "TRON vanity generator - CPU + OpenCL + CUDA + Apple Metal\n"
        "  --seconds N       run duration; 0 = until Ctrl+C\n"
        "  --threads N       CPU threads; default = logical cores\n"
        "  --words FILE      dictionary; default words.txt\n"
        "  --out DIR         output directory; default results\n"
        "  --output FILE     direct JSONL output override\n"
        "  --backend auto|cpu|opencl|cuda|metal\n"
        "  --gpu-batch N     GPU batch size (power of two, 1024..1048576)\n"
        "  --gpu-resident    GPU CSPRNG + device result ring mode\n"
        "  --gpu-rng NAME    chacha12, aes-ctr, or philox (resident mode)\n"
        "  --gpu-buffer-mb N device result ring size, default 128\n"
        "  --gpu-chunk-ms N  bounded GPU chunk target, default 32\n"
        "  --gpu-poll-ms N   compatibility option; bounded GPU dispatches poll on completion\n"
        "  --gpu-group-size N resident work-group size: 64, 128, or 256 (default 256)\n"
        "  --metal-keys-per-lane N Metal point-walk batch: power of two, 1..1024 (default 32)\n"
        "  --bench-seconds N seconds per benchmark method; --bench runs all available methods\n"
        "  --bench-resident  benchmark resident GPU backends only (skip legacy tuning)\n"
        "  --opencl-profile  time selected resident OpenCL mode; no wallets written\n"
        "  --opencl-inverse single|pair  resident field inversion (default single)\n"
        "  --opencl-compiler compact|default  resident compiler mode (default compact)\n"
        "  --opencl-pipeline monolithic|staged  resident program layout (default monolithic)\n"
        "  --opencl-diagnose smoke|rng|scan|full  isolated checks; no wallet output\n"
        "                    build-curve|build-affine|build-keccak|build-checksum|build-base58|build-match\n"
        "                    build-address (legacy combined stage; compile only)\n"
        "  --opencl-host-seed  use OS CSPRNG, exclude GPU RNG from resident OpenCL\n"
        "  --metal-profile-stages  profile the active Metal resident pipeline by stage; no wallets written\n"
        "  --metal-profile-stage N  profile only stage 0..5 (0 is full resident)\n"
        "  --metal-profile-indexed-keccak profile the slower indexed reference Keccak\n"
        "  --metal-profile-max-threads N set compiler occupancy hint for profile only\n"
        "  --metal-hw-profile run deterministic Metal hardware microbenchmarks\n"
        "  --metal-hw-case NAME filter hardware cases by name/category\n"
        "  --metal-hw-json FILE write hardware profile results as JSON\n"
        "  --case-sensitive  exact case matching\n"
        "  --list            list CPU/OpenCL/CUDA devices and exit\n"
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
            if (a == "--help" || a == "-h") { o.help = true; usage(); return false; }
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
            else if (a == "--gpu-batch") o.gpuBatch = std::stoul(next(i, "--gpu-batch"));
            else if (a == "--gpu-resident") o.gpuResident = true;
            else if (a == "--gpu-rng") o.gpuRng = next(i, "--gpu-rng");
            else if (a == "--gpu-buffer-mb") o.gpuBufferMiB = std::stoul(next(i, "--gpu-buffer-mb"));
            else if (a == "--gpu-chunk-ms") o.gpuChunkMs = std::stoul(next(i, "--gpu-chunk-ms"));
            else if (a == "--gpu-poll-ms") o.gpuPollMs = std::stoul(next(i, "--gpu-poll-ms"));
            else if (a == "--gpu-group-size") o.gpuGroupSize = std::stoul(next(i, "--gpu-group-size"));
            else if (a == "--metal-keys-per-lane") o.metalKeysPerLane = std::stoul(next(i, "--metal-keys-per-lane"));
            else if (a == "--bench-seconds") o.benchSeconds = std::stod(next(i, "--bench-seconds"));
            else if (a == "--bench-resident") o.benchResidentOnly = true;
            else if (a == "--opencl-profile") o.openclProfile = true;
            else if (a == "--opencl-diagnose") o.openclDiagnostic = next(i, "--opencl-diagnose");
            else if (a == "--opencl-host-seed") { o.openclOptions.hostSeed = true; o.gpuResident = true; }
            else if (a == "--opencl-pipeline") {
                const auto value = next(i, "--opencl-pipeline");
                if (value != "monolithic" && value != "staged")
                    throw std::runtime_error("--opencl-pipeline must be monolithic or staged");
                o.openclOptions.staged = value == "staged";
                o.gpuResident = true;
            }
            else if (a == "--opencl-inverse") {
                const auto value = next(i, "--opencl-inverse");
                if (value != "single" && value != "pair") throw std::runtime_error("--opencl-inverse must be single or pair");
                o.openclOptions.pairInverse = value == "pair";
            }
            else if (a == "--opencl-compiler") {
                const auto value = next(i, "--opencl-compiler");
                if (value != "compact" && value != "default")
                    throw std::runtime_error("--opencl-compiler must be compact or default");
                o.openclOptions.compact = value == "compact";
            }
            else if (a == "--metal-profile-stages") o.metalProfileStages = true;
            else if (a == "--metal-profile-stage") o.metalProfileStage = std::stoi(next(i, "--metal-profile-stage"));
            else if (a == "--metal-profile-indexed-keccak") o.metalProfileScalarKeccak = false;
            else if (a == "--metal-profile-max-threads") o.metalProfileMaxThreads = std::stoul(next(i, "--metal-profile-max-threads"));
            else if (a == "--metal-hw-profile") o.metalHardwareProfile = true;
            else if (a == "--metal-hw-case") { o.metalHardwareCase = next(i, "--metal-hw-case"); o.metalHardwareProfile = true; }
            else if (a == "--metal-hw-json") { o.metalHardwareJson = next(i, "--metal-hw-json"); o.metalHardwareProfile = true; }
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
    if (o.backend != "auto" && o.backend != "cpu" && o.backend != "opencl" && o.backend != "cuda" && o.backend != "metal") {
        std::cerr << "backend must be auto, cpu, opencl, cuda, or metal\n"; return false;
    }
    if (!std::isfinite(o.benchSeconds) || o.benchSeconds <= 0) {
        std::cerr << "--bench-seconds must be finite and positive\n"; return false;
    }
    if (o.openclOptions.hostSeed && o.backend != "opencl") {
        std::cerr << "--opencl-host-seed requires --backend opencl\n"; return false;
    }
    if (o.openclOptions.staged && o.backend != "opencl") {
        std::cerr << "--opencl-pipeline staged requires --backend opencl\n"; return false;
    }
    if (!o.openclDiagnostic.empty() && o.openclDiagnostic != "smoke" && o.openclDiagnostic != "rng" &&
        o.openclDiagnostic != "scan" && o.openclDiagnostic != "full" &&
        o.openclDiagnostic != "build-curve" && o.openclDiagnostic != "build-affine" &&
        o.openclDiagnostic != "build-keccak" && o.openclDiagnostic != "build-checksum" &&
        o.openclDiagnostic != "build-base58" && o.openclDiagnostic != "build-match" &&
        o.openclDiagnostic != "build-address") {
        std::cerr << "unknown --opencl-diagnose stage; see --help\n"; return false;
    }
    if (o.gpuGroupSize != 64 && o.gpuGroupSize != 128 && o.gpuGroupSize != 256) {
        std::cerr << "--gpu-group-size must be 64, 128, or 256\n"; return false;
    }
    if (o.metalKeysPerLane == 0 || o.metalKeysPerLane > 1024 ||
        (o.metalKeysPerLane & (o.metalKeysPerLane - 1)) != 0) {
        std::cerr << "--metal-keys-per-lane must be a power of two from 1 to 1024\n"; return false;
    }
    if (static_cast<uint64_t>(o.gpuGroupSize) * o.metalKeysPerLane > 65536) {
        std::cerr << "Metal group size times keys per lane must not exceed 65536\n";
        return false;
    }
    if (o.metalProfileStage < -1 || o.metalProfileStage > 5) {
        std::cerr << "--metal-profile-stage must be 0..5\n"; return false;
    }
    if (o.metalProfileMaxThreads &&
        (o.metalProfileMaxThreads < 32 || o.metalProfileMaxThreads > 1024 ||
         o.metalProfileMaxThreads % 32 != 0)) {
        std::cerr << "--metal-profile-max-threads must be 0 or a multiple of 32 up to 1024\n";
        return false;
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

std::string benchRate(double keysPerSecond) {
    std::ostringstream s;
    if (keysPerSecond >= 1e6) {
        s << std::fixed << std::setprecision(2) << keysPerSecond / 1e6 << " M/s";
    } else if (keysPerSecond >= 1e3) {
        s << std::fixed << std::setprecision(1) << keysPerSecond / 1e3 << " K/s";
    } else {
        s << std::fixed << std::setprecision(0) << keysPerSecond << " /s";
    }
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
    if (hw.cudaGpus.empty()) std::cout << "CUDA GPU: none (" << hw.cudaNote << ")\n";
    for (const auto& g : hw.cudaGpus) {
        std::cout << "CUDA GPU: " << g.name << " / compute " << g.cudaCapability / 10 << "."
                  << g.cudaCapability % 10 << " / " << g.computeUnits << " SM / "
                  << g.globalMemBytes / (1024 * 1024) << " MiB\n";
    }
#if defined(__APPLE__)
    std::string note;
    std::cout << "Metal GPU: " << (metalAvailable(&note) ? note : "none (" + note + ")") << "\n";
#endif
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
    if (!parse(argc, argv, opt)) return opt.help ? 0 : 1;
    HardwareReport hw = detectHardware();
    if (opt.list) { printDevices(hw); return 0; }
    if (!opt.openclDiagnostic.empty()) {
        if (opt.backend != "auto" && opt.backend != "opencl") {
            std::cerr << "--opencl-diagnose requires --backend opencl\n"; return 1;
        }
        if (hw.gpus.empty()) { std::cerr << "OpenCL unavailable: " << hw.openclNote << "\n"; return 1; }
        for (const auto& device : hw.gpus)
            if (diagnoseOpencl(device, opt.openclDiagnostic, opt.gpuRng, opt.openclOptions)) return 1;
        return 0;
    }
    if (opt.metalHardwareProfile) {
        return runMetalHardwareProfile({opt.benchSeconds, opt.metalHardwareCase, opt.metalHardwareJson});
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gputest") {
            if (opt.backend == "cuda") {
                if (hw.cudaGpus.empty()) { std::cerr << "CUDA unavailable: " << hw.cudaNote << "\n"; return 1; }
                for (const auto& device : hw.cudaGpus) if (cudaSelfTest(device)) return 1;
                return 0;
            }
            if (hw.gpus.empty()) { std::cerr << "no OpenCL GPU: " << hw.openclNote << "\n"; return 1; }
            if (opt.gpuResident) {
                for (const auto& device : hw.gpus)
                    if (openclResidentSelfTest(device, opt.gpuRng, opt.openclOptions)) return 1;
                return 0;
            }
            return gpuSelfTest(hw.gpus.front());
        }
    }

    std::string error;
    auto dictionary = Dictionary::load(opt.words, opt.caseSensitive, &error);
    if (!dictionary) { std::cerr << error << "\n"; return 1; }
    if (opt.openclProfile) {
        if (opt.backend != "auto" && opt.backend != "opencl") {
            std::cerr << "--opencl-profile requires --backend opencl\n"; return 1;
        }
        if (hw.gpus.empty()) { std::cerr << "OpenCL unavailable: " << hw.openclNote << "\n"; return 1; }
        std::cout << "OpenCL resident profile; no wallets or private keys written\n"
                  << "Dictionary: " << dictionary->words.size() << " words; RNG "
                  << (opt.openclOptions.hostSeed ? "OS CSPRNG" : opt.gpuRng)
                  << "; compiler " << (opt.openclOptions.compact ? "compact" : "default")
                  << "; inverse " << (opt.openclOptions.pairInverse ? "pair" : "single")
                  << "; pipeline " << (opt.openclOptions.staged ? "staged" : "monolithic")
                  << "; group " << opt.gpuGroupSize << "\n"
                  << "Includes full address/dictionary math and metadata drain; excludes CPU match verification/output.\n";
        for (const auto& device : hw.gpus) {
            auto p = profileOpenclResident(device, dictionary, opt.gpuRng, opt.gpuBufferMiB,
                opt.gpuChunkMs, opt.gpuGroupSize, opt.openclOptions, opt.benchSeconds);
            if (!p.error.empty()) { std::cerr << "OpenCL profile failed: " << p.error << "\n"; return 1; }
            std::cout << std::fixed << std::setprecision(3) << device.name
                      << ": " << p.keys << " keys / " << p.dispatches << " chunks\n"
                      << "wall " << p.wallSeconds << " s, wall speed " << p.keys / p.wallSeconds / 1e6 << " M/s\n"
                      << "base preparation " << p.baseSeconds << " s, scan+wait " << p.scanSeconds
                      << " s, remaining host/drain " << std::max(0.0, p.wallSeconds - p.baseSeconds - p.scanSeconds) << " s\n";
            if (p.gpuTimingValid && p.gpuSeconds > 0)
                std::cout << "Driver-reported GPU scan " << p.gpuSeconds << " s, kernel-only speed " << p.keys / p.gpuSeconds / 1e6
                          << " M/s, max chunk GPU time " << p.maxGpuMs << " ms\n";
            else std::cout << "GPU event timestamps unavailable (wall timing remains valid)\n";
            std::cout << "Use wall speed for comparisons; event time excludes queueing, transfers and host work.\n";
        }
        return 0;
    }
    if (opt.metalProfileStages || opt.metalProfileStage >= 0) {
#if defined(__APPLE__)
        if (opt.backend != "auto" && opt.backend != "metal") {
            std::cerr << "--metal-profile-stages requires --backend metal\n";
            return 1;
        }
        const char* labels[] = {"full resident", "EC + affine", "+ Keccak", "+ SHA256d",
                                "+ Base58", "+ dictionary"};
        std::cout << "Metal resident stage profile (throwaway benchmark run; no wallet output)\n"
                  << "Stage variants are separate compiled kernels; deltas are diagnostic, not exact function timings.\n"
                  << "stage             wall M/s   GPU M/s   GPU ns/key  GPU ms/cmd   GPU busy\n";
        const uint32_t firstStage = opt.metalProfileStage >= 0 ? static_cast<uint32_t>(opt.metalProfileStage) : 0;
        const uint32_t lastStage = opt.metalProfileStage >= 0 ? firstStage : 5;
        for (uint32_t stage = firstStage; stage <= lastStage; ++stage) {
            MetalProfileResult p = profileMetalResidentStage(dictionary, opt.gpuRng,
                opt.gpuBufferMiB, opt.gpuChunkMs, opt.gpuGroupSize,
                opt.metalKeysPerLane, stage, opt.benchSeconds, opt.metalProfileScalarKeccak,
                opt.metalProfileMaxThreads);
            if (!p.error.empty()) {
                std::cerr << "Metal profile stage " << stage << " failed: " << p.error << "\n";
                return 1;
            }
            const double wallRate = p.wallSeconds > 0 ? p.keys / p.wallSeconds / 1e6 : 0.0;
            const double gpuRate = p.gpuSeconds > 0 ? p.keys / p.gpuSeconds / 1e6 : 0.0;
            const double nsPerKey = p.keys ? p.gpuSeconds / p.keys * 1e9 : 0.0;
            const double msPerCommand = p.dispatches ? p.gpuSeconds * 1e3 / p.dispatches : 0.0;
            const double busy = p.wallSeconds > 0 ? p.gpuSeconds / p.wallSeconds * 100.0 : 0.0;
            std::cout << std::left << std::setw(18) << labels[stage]
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(9) << wallRate << std::setw(10) << gpuRate
                      << std::setw(13) << nsPerKey << std::setw(12) << msPerCommand
                      << std::setw(10) << busy << "%\n";
        }
        return 0;
#else
        std::cerr << "--metal-profile-stages requires macOS\n";
        return 1;
#endif
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench" || std::string(argv[i]) == "--bench-resident") {
            const double seconds = opt.benchSeconds;
            const bool all = opt.backend == "auto";
            const bool wantCpu = !opt.benchResidentOnly && (all || opt.backend == "cpu");
            const bool wantOpencl = all || opt.backend == "opencl";
            const bool wantCuda = all || opt.backend == "cuda";
            const bool wantMetal = all || opt.backend == "metal";
            const std::vector<std::string> rngs = {"chacha12", "aes-ctr", "philox"};
            std::cout << "\n=== " << (opt.benchResidentOnly ? "Resident GPU benchmark" : "Full benchmark matrix")
                      << " (" << seconds << " s per method) ===\n"
                      << (opt.benchResidentOnly ? "Resident OpenCL/CUDA/Metal RNGs; legacy tuning skipped\n"
                                                 : "CPU + legacy OpenCL tuning + resident OpenCL/CUDA/Metal RNGs\n")
                      << "No wallet output is written by benchmark mode.\n\n"
                      << std::left << std::setw(38) << "method"
                      << std::right << std::setw(16) << "throughput" << "\n"
                      << std::string(56, '-') << "\n";

            if (wantCpu) {
                auto cpu = makeCpuBackend();
                double r = cpu->benchmark(seconds);
                std::cout << std::left << std::setw(38) << "CPU (all host threads)"
                          << std::right << std::setw(16) << benchRate(r) << "\n";
            }

            if (wantOpencl && !hw.gpus.empty()) {
                if (!opt.benchResidentOnly) for (const auto& g : hw.gpus) {
                    std::cout << "\n[legacy OpenCL tuning] " << g.name << "\n";
                    double r = gpuBench(g, seconds);
                    std::cout << std::left << std::setw(38) << ("legacy OpenCL best / " + g.name)
                              << std::right << std::setw(16) << benchRate(r) << "\n";
                }
                for (const auto& g : hw.gpus) {
                    const auto openclRngs = opt.openclOptions.hostSeed ? std::vector<std::string>{opt.gpuRng} : rngs;
                    for (const auto& rng : openclRngs) {
                        const std::string rngLabel = opt.openclOptions.hostSeed ? "OS CSPRNG" : rng;
                        auto resident = makeResidentGpuBackend(g, dictionary, rng,
                                                               opt.gpuBufferMiB, opt.gpuChunkMs, opt.gpuPollMs,
                                                               opt.gpuGroupSize, opt.openclOptions);
                        if (!resident || !resident->available()) {
                            std::cout << "resident OpenCL " << rngLabel << " / " << g.name
                                      << ": unavailable (" << (resident ? resident->note() : "not built") << ")\n";
                            continue;
                        }
                        double r = resident->benchmark(seconds);
                        if (!resident->note().empty()) {
                            std::cerr << "OpenCL benchmark failed: " << resident->note() << "\n";
                            return 1;
                        }
                        std::cout << std::left << std::setw(38)
                                  << ("resident OpenCL " + rngLabel + " / " + g.name)
                                  << std::right << std::setw(16) << benchRate(r) << "\n";
                    }
                }
            } else if (wantOpencl) {
                std::cout << "OpenCL: unavailable (" << hw.openclNote << ")\n";
            }

            if (wantCuda) {
                if (hw.cudaGpus.empty()) {
                    std::cerr << "CUDA unavailable: " << hw.cudaNote << "\n";
                    if (!all) return 1;
                }
                for (const auto& device : hw.cudaGpus) for (const auto& rng : rngs) {
                    auto cuda = makeCudaBackend(device, dictionary, rng, opt.gpuBufferMiB,
                                                opt.gpuChunkMs, opt.gpuGroupSize);
                    if (!cuda->available()) {
                        std::cerr << "CUDA " << rng << ": " << cuda->note() << "\n";
                        if (!all) return 1;
                        continue;
                    }
                    const double speed = cuda->benchmark(seconds);
                    if (!cuda->note().empty()) {
                        std::cerr << "CUDA benchmark failed: " << cuda->note() << "\n";
                        if (!all) return 1;
                        continue;
                    }
                    std::cout << std::left << std::setw(38) << ("CUDA " + rng + " / " + device.name)
                              << std::right << std::setw(16) << benchRate(speed) << "\n";
                }
            }
#if defined(__APPLE__)
            if (wantMetal) {
                for (const auto& rng : rngs) {
                    auto metal = makeMetalResidentBackend(dictionary, rng,
                                                           opt.gpuBufferMiB, opt.gpuChunkMs,
                                                           opt.gpuGroupSize, opt.metalKeysPerLane);
                    if (!metal || !metal->available()) {
                        std::cout << "Metal " << rng << ": unavailable ("
                                  << (metal ? metal->note() : "not built") << ")\n";
                        continue;
                    }
                    double r = metal->benchmark(seconds);
                    std::cout << std::left << std::setw(38) << ("Metal resident " + rng)
                              << std::right << std::setw(16) << benchRate(r) << "\n";
                }
            }
#else
            if (wantMetal) std::cout << "Metal: unavailable (not an Apple build)\n";
#endif
            std::cout << "\nBenchmark complete. Use --bench-seconds N to adjust each row.\n";
            return 0;
        }
    }
    gpuSetKeysPerItem(opt.keysPerItem);
    if (opt.gpuBatch) {
        gpuSetBatch(opt.gpuBatch);
    } else {
        uint32_t batch = 1u << 16;
        for (const auto& g : hw.gpus) if (!g.integrated) batch = 1u << 20;
        gpuSetBatch(batch);
    }
    gpuSetEcWindow(opt.ecWindow);
    gpuSetMontN(opt.montN);

    std::vector<std::unique_ptr<Backend>> backends;
    if (opt.backend == "auto" || opt.backend == "cpu") backends.push_back(makeCpuBackend());
    if (opt.backend == "cuda") {
        if (hw.cudaGpus.empty()) { std::cerr << "CUDA unavailable: " << hw.cudaNote << "\n"; return 1; }
        for (const auto& device : hw.cudaGpus)
            backends.push_back(makeCudaBackend(device, dictionary, opt.gpuRng,
                opt.gpuBufferMiB, opt.gpuChunkMs, opt.gpuGroupSize));
    }
    bool autoMetal = false;
#if defined(__APPLE__)
    autoMetal = opt.backend == "auto";
#endif
    if ((opt.gpuResident || opt.backend == "metal") && (opt.backend == "metal" || autoMetal)) {
        auto metal = makeMetalResidentBackend(dictionary, opt.gpuRng,
                                              opt.gpuBufferMiB, opt.gpuChunkMs,
                                              opt.gpuGroupSize, opt.metalKeysPerLane);
        if (metal) backends.push_back(std::move(metal));
    } else if (opt.gpuResident && (opt.backend == "auto" || opt.backend == "opencl")) {
        for (const auto& g : hw.gpus) {
            backends.push_back(makeResidentGpuBackend(g, dictionary, opt.gpuRng,
                                                      opt.gpuBufferMiB, opt.gpuChunkMs, opt.gpuPollMs,
                                                      opt.gpuGroupSize, opt.openclOptions));
        }
        if (hw.gpus.empty() && opt.backend == "opencl") {
            std::cerr << "OpenCL unavailable; falling back to CPU\n";
            backends.push_back(makeCpuBackend());
        }
    } else if (!opt.gpuResident && ((opt.backend == "auto" && !autoMetal) || opt.backend == "opencl")) {
        for (const auto& g : hw.gpus) backends.push_back(makeGpuBackend(g, dictionary));
        if (hw.gpus.empty() && opt.backend == "opencl") {
            std::cerr << "OpenCL unavailable; falling back to CPU\n";
            backends.push_back(makeCpuBackend());
        }
    }
    if (opt.backend == "metal" && backends.empty()) {
        std::cerr << "Metal backend is not available in this build/platform\n";
    }
    if (backends.empty()) { std::cerr << "no backend available\n"; return 1; }

    size_t availableBackends = 0;
    for (auto& b : backends) {
        auto info = b->info();
        std::cout << info.kind << ": " << info.title << "\n";
        for (const auto& line : info.lines) std::cout << "  " << line << "\n";
        if (!b->available()) std::cerr << "  unavailable: " << b->note() << "\n";
        else ++availableBackends;
    }
    if (!availableBackends) { std::cerr << "no usable backend\n"; return 1; }

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
    std::thread progressThread([&] {
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
    progressThread.join();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "\nDone: " << state.checked.load() << " keys, " << rate(state.checked.load(), elapsed)
              << ", CPU=" << rate(state.cpuChecked.load(), elapsed)
              << ", GPU=" << rate(state.gpuChecked.load(), elapsed) << "\n";
    for (const auto& b : backends) if (b->available() && !b->note().empty()) return 1;
    return 0;
}
