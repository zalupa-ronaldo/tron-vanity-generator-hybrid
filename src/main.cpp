#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <secp256k1.h>

#include "backend.h"
#include "crypto.h"
#include "hwdetect.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace {

// Windows 控制台：让 UTF-8 中文正常显示，并关闭 stdout 缓冲（否则输出会卡在缓冲区里，
// 看起来像“只有光标在闪、没反应”）。
void consoleInit() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << std::unitbuf;   // 每次 << 之后立即刷新
    std::cerr << std::unitbuf;
}

RunState* gState = nullptr;

void onSigint(int) {
    if (gState) gState->stop.store(true);
}

struct Options {
    int minLen = 6;
    unsigned int threads = 0;
    uint64_t maxAttempts = 0;
    uint64_t seconds = 60;
    std::string words = "words.txt";
    std::string output = "results";
    std::string backend = "auto";   // auto | cpu | opencl
    double benchSeconds = 2.0;
    uint32_t keysPerItem = 1;       // GPU: 每 work-item 处理的私钥数
    uint32_t ecWindow = 0;          // GPU: 固定基点窗口位宽（0=用内置默认）
    uint32_t montN = 0;             // GPU: Montgomery 批量求逆 work-group 大小（0=默认）
    bool verbose = false;
    bool listOnly = false;
    bool caseSensitive = false;
};

void printUsage() {
    std::cout <<
        "TRON vanity generator (CPU + AMD/NVIDIA OpenCL)\n"
        "Ищет слова из dictionary в любой позиции после начальной T.\n\n"
        "用法: tron_vanity_generator [选项]\n"
        "  --seconds N      длительность, 0=до Ctrl+C\n"
        "  --threads N      CPU 线程数，默认=逻辑核心数\n"
        "  --words FILE     словарь, default words.txt\n"
        "  --out DIR        output directory, default results\n"
        "  --max N          最大尝试次数，0=无限(默认)\n"
        "  --output FILE    direct JSONL output override\n"
        "  --backend X      auto | cpu | opencl，default auto (CPU+GPU)\n"
        "  --case-sensitive точный регистр\n"
        "  --bench-seconds S 自动选择时每个后端的压测秒数，默认 2\n"
        "  --keys-per-item N GPU 每个 work-item 连续处理的私钥数，默认 1\n"
        "  --ec-window N     GPU 固定基点窗口位宽 (1=逐bit, 2..8=comb)，默认 7\n"
        "  --mont-n N        GPU Montgomery 批量求逆 work-group 大小 (1=关)，默认 1\n"
        "  --verbose        输出进度\n"
        "  --list           只打印硬件检测结果后退出\n"
        "  --bench [秒]     GPU 压测：扫 keys-per-item 甜点位 + 各尾号规则吞吐\n"
        "  --profile [秒]   GPU 内核逐阶段耗时占比\n"
        "  --gputest        GPU 内核逐项对照 CPU\n"
        "  --selftest / --hashtest / --matchtest   地址算法 / 哈希 / 命中规则 自检\n"
        "  --help           显示本帮助\n";
}

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "缺少参数: " << what << "\n"; return ""; }
            return argv[++i];
        };
        if (a == "--seconds") o.seconds = std::stoull(next("--seconds"));
        else if (a == "--threads") o.threads = static_cast<unsigned int>(std::stoul(next("--threads")));
        else if (a == "--words") o.words = next("--words");
        else if (a == "--out") o.output = next("--out");
        else if (a == "--max") o.maxAttempts = std::stoull(next("--max"));
        else if (a == "--output") o.output = next("--output");
        else if (a == "--backend") o.backend = next("--backend");
        else if (a == "--case-sensitive") o.caseSensitive = true;
        else if (a == "--bench-seconds") o.benchSeconds = std::stod(next("--bench-seconds"));
        else if (a == "--keys-per-item") o.keysPerItem = static_cast<uint32_t>(std::stoul(next("--keys-per-item")));
        else if (a == "--ec-window") o.ecWindow = static_cast<uint32_t>(std::stoul(next("--ec-window")));
        else if (a == "--mont-n") o.montN = static_cast<uint32_t>(std::stoul(next("--mont-n")));
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--list") o.listOnly = true;
        else if (a == "--help" || a == "-h") { printUsage(); return false; }
        else { std::cerr << "未知参数: " << a << "\n"; printUsage(); return false; }
    }
    if (o.backend == "gpu") o.backend = "opencl";
    if (o.backend != "auto" && o.backend != "cpu" && o.backend != "opencl") {
        std::cerr << "--backend must be auto, cpu, or opencl\n"; return false;
    }
    return true;
}

class FileSink {
public:
    explicit FileSink(std::string path) : path_(std::move(path)) {}

    void operator()(const FoundKey& fk) {
        std::lock_guard<std::mutex> lk(mu_);
        std::ofstream out(path_, std::ios::app | std::ios::binary);
        if (out.is_open()) {
            out << "{\"address\":\"" << escape(fk.address) << "\",\"words\":[";
            for (size_t i = 0; i < fk.words.size(); ++i) {
                if (i) out << ',';
                out << "\"" << escape(fk.words[i]) << "\"";
            }
            out << "],\"private_key\":\"" << escape(fk.privHex) << "\"}\n";
        } else {
            std::cerr << "无法写入 " << path_ << "\n";
        }
        std::cout << "[match #" << (++shown_) << "] " << fk.address
                  << " words=" << fk.words.size() << "\n";
    }

private:
    static std::string escape(const std::string& s) {
        std::string out;
        for (unsigned char c : s) {
            if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
            else if (c >= 0x20) out.push_back(static_cast<char>(c));
        }
        return out;
    }
    std::string path_;
    std::mutex mu_;
    uint64_t shown_ = 0;
};

std::string humanRate(double r) {
    char buf[64];
    if (r >= 1e6) std::snprintf(buf, sizeof(buf), "%.2f M keys/s", r / 1e6);
    else if (r >= 1e3) std::snprintf(buf, sizeof(buf), "%.1f K keys/s", r / 1e3);
    else std::snprintf(buf, sizeof(buf), "%.0f keys/s", r);
    return buf;
}

std::string groupThousands(uint64_t v) {
    std::string s = std::to_string(v), out;
    int c = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (c && c % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++c;
    }
    return std::string(out.rbegin(), out.rend());
}

std::string keysPerSec(double r) { return groupThousands(static_cast<uint64_t>(r + 0.5)) + " keys/s"; }

}  // namespace

int selftest() {
    // privkey = 0x00..01 的已知 TRON 地址
    unsigned char sk[32] = {0};
    sk[31] = 1;
    secp256k1_context* c = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    secp256k1_pubkey pub;
    secp256k1_ec_pubkey_create(c, &pub, sk);
    unsigned char out[65];
    size_t len = 65;
    secp256k1_ec_pubkey_serialize(c, out, &len, &pub, SECP256K1_EC_UNCOMPRESSED);
    secp256k1_context_destroy(c);
    std::string addr = tronAddressFromPubXY(out + 1);
    std::cout << "privkey=1 -> " << addr << "\n";
    const std::string expect = "TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC";
    std::cout << (addr == expect ? "SELFTEST OK\n" : "SELFTEST MISMATCH (expected " + expect + ")\n");
    return addr == expect ? 0 : 1;
}

int hashtest();

int matchtest() {
    struct Case { const char* addr; int minLen; bool wantMatch; const char* wantKind; };
    const Case cases[] = {
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxAAAAA", 5, true,  "相同"},
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxx12345", 5, true,  "连续"},
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxabcde", 5, true,  "连续"},
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxvwxyz", 5, true,  "连续"},
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxdefgh", 5, true,  "连续"},
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxABCDE", 5, true,  "连续"},
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxxWXYZ", 4, true,  "连续"},
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxx54321", 5, false, ""},      // 降序（只认升序）
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxedcba", 5, false, ""},      // 降序
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxZYXWV", 5, false, ""},      // 降序
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxWXYZ1", 5, false, ""},      // 大写跨到数字
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxx89123", 5, false, ""},      // 跨过缺失的 0
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxxyzab", 5, false, ""},      // z→a 回绕
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxx9abcd", 5, false, ""},      // 数字跨字母
        {"TxxxxxxxxxxxxxxxxxxxxxxxxxxxFGHJ", 5, false, ""},      // 跳过缺失的 I（大写）
        {"Txxxxxxxxxxxxxxxxxxxxxxxxxxaaaa5", 5, false, ""},      // 只有 4 位相同
    };
    int fails = 0;
    for (const auto& c : cases) {
        std::string a = c.addr;
        MatchResult r = evaluateAddress(a, c.minLen);
        bool ok = (r.matched == c.wantMatch) &&
                  (!c.wantMatch || std::string(r.kind) == c.wantKind);
        std::cout << (ok ? "  OK   " : "  FAIL ") << "..." << a.substr(a.size() - 6)
                  << "  -> matched=" << r.matched << " kind=" << (r.matched ? r.kind : "-")
                  << " len=" << r.runLen << "\n";
        if (!ok) ++fails;
    }
    std::cout << (fails ? std::to_string(fails) + " 个失败\n" : "MATCHTEST OK\n");
    return fails ? 1 : 0;
}

int gpuSelfTest(const GpuDevice& dev);
int gpuBench(const GpuDevice& dev, double secs);
int gpuProfile(const GpuDevice& dev, double secs);
void gpuSetKeysPerItem(uint32_t n);
void gpuSetEcWindow(uint32_t w);
void gpuSetMontN(uint32_t n);

int legacyMain(int argc, char** argv) {
    consoleInit();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") return selftest();
        if (a == "--hashtest") return hashtest();
        if (a == "--matchtest") return matchtest();
        if (a == "--gputest") {
            HardwareReport hw = detectHardware();
            if (hw.gpus.empty()) { std::cerr << "无 GPU: " << hw.openclNote << "\n"; return 1; }
            return gpuSelfTest(hw.gpus[0]);
        }
        if (a == "--bench" || a == "--profile") {
            HardwareReport hw = detectHardware();
            if (hw.gpus.empty()) { std::cerr << "无 GPU: " << hw.openclNote << "\n"; return 1; }
            double secs = (i + 1 < argc) ? std::atof(argv[i + 1]) : 2.0;
            if (secs <= 0) secs = 2.0;
            return a == "--profile" ? gpuProfile(hw.gpus[0], secs) : gpuBench(hw.gpus[0], secs);
        }
    }

    Options opt;
    if (!parseArgs(argc, argv, opt)) return 0;

    gpuSetKeysPerItem(opt.keysPerItem);
    gpuSetEcWindow(opt.ecWindow);
    gpuSetMontN(opt.montN);
    HardwareReport hw = detectHardware();

    // 候选后端
    std::vector<std::unique_ptr<Backend>> backends;
    if (opt.backend == "cpu" || opt.backend == "auto")
        backends.push_back(makeCpuBackend());
    if (opt.backend == "gpu" || opt.backend == "auto") {
        for (const auto& g : hw.gpus) backends.push_back(makeGpuBackend(g));
        if (hw.gpus.empty() && opt.backend == "gpu") {
            std::cerr << "未检测到可用 GPU (" << hw.openclNote << ")，回退到 CPU。\n";
            backends.push_back(makeCpuBackend());
        }
    }

    auto line = [] { std::cout << std::string(44, '-') << "\n"; };
    std::cout << "\n";
    line();
    std::cout << "        TRON VANITY GENERATOR\n";
    line();
    std::cout << "\n";

    struct Cand { Backend* b; double rate; bool ok; };
    std::vector<Cand> cands;
    bool doBench = !opt.listOnly && backends.size() > 1;

    for (auto& up : backends) {
        Backend* b = up.get();
        BackendInfo bi = b->info();
        std::cout << bi.kind << "  : " << bi.title << "\n";
        for (auto& l : bi.lines) std::cout << "       " << l << "\n";

        double rate = 0.0;
        if (bi.kind == "GPU" && (opt.backend == "gpu" || opt.backend == "auto"))
            std::cout << "       (首次准备 GPU：编译 OpenCL 内核，约需数秒...)\n";
        bool ok = b->available();
        if (!ok) {
            std::cout << "       [" << b->note() << "]\n";
        } else if (doBench) {
            std::cout << "       (压测算力 ~" << (opt.benchSeconds * (bi.kind == "GPU" ? 1.6 : 1.0))
                      << "s...)\n";
            rate = b->benchmark(opt.benchSeconds);
            std::cout << "       " << keysPerSec(rate) << "\n";
        }
        cands.push_back({b, rate, ok});
        std::cout << "\n";
    }

    if (hw.gpus.empty() && opt.backend != "cpu") {
        std::cout << "GPU  : 无可用 OpenCL GPU\n       [" << hw.openclNote << "]\n\n";
    }

    if (opt.listOnly) return 0;

    // 选择最快的可用后端
    Backend* chosen = nullptr;
    double chosenRate = 0.0;
    for (auto& c : cands) {
        if (c.ok && (!chosen || c.rate > chosenRate)) { chosen = c.b; chosenRate = c.rate; }
    }
    if (!chosen) { std::cerr << "没有可用的计算后端。\n"; return 1; }

    // 组织 Reason 文案
    std::string reason;
    if (doBench) {
        for (auto& c : cands) {
            if (c.b == chosen || !c.ok) continue;
            reason = chosen->name() + " faster (" + c.b->name() + " " + keysPerSec(c.rate) + ")";
        }
    }
    for (auto& c : cands) {
        if (!c.ok && chosen->info().kind == "CPU")
            reason = c.b->name() + " 不可用: " + c.b->note();
    }

    line();
    std::cout << "Backend : " << chosen->name() << "\n";
    if (!reason.empty()) std::cout << "Reason  : " << reason << "\n";
    if (doBench) std::cout << "Speed   : " << keysPerSec(chosenRate) << "\n";
    line();
    std::cout << "\n";

    RunConfig cfg;
    cfg.minLen = opt.minLen;
    cfg.threads = opt.threads;
    cfg.maxAttempts = opt.maxAttempts;
    cfg.verbose = opt.verbose;

    RunState state;
    gState = &state;
    std::signal(SIGINT, onSigint);

    double rate = chosenRate > 0 ? chosenRate : (chosen->info().kind == "GPU" ? 700000.0 : 250000.0);
    // 命中 minLen 位相同/连续尾号的粗略概率（相同 + 递增 + 递减）
    double p = 3.0 * std::pow(1.0 / 58.0, opt.minLen - 1);
    double etaSec = p > 0 ? 1.0 / (rate * p) : 0;

    std::cout << "最小位数 : " << opt.minLen << "\n"
              << "输出文件 : " << opt.output << "\n"
              << "尝试上限 : " << (opt.maxAttempts ? groupThousands(opt.maxAttempts) : "无限") << "\n";
    if (etaSec > 0) {
        std::string eta = etaSec < 1     ? "不到 1 秒"
                        : etaSec < 90    ? std::to_string((long)(etaSec + 0.5)) + " 秒"
                        : etaSec < 5400  ? std::to_string((long)(etaSec / 60 + 0.5)) + " 分钟"
                        : etaSec < 172800 ? std::to_string((long)(etaSec / 3600 + 0.5)) + " 小时"
                        : std::to_string((long)(etaSec / 86400 + 0.5)) + " 天";
        std::cout << "预计每命中一个约需 " << eta << "（粗略估计）\n";
    }
    std::cout << "运行中，每 10 秒报告一次进度。按 Ctrl+C 停止。\n\n";

    FileSink sink(opt.output);
    ReportFn report = [&sink](const FoundKey& fk) { sink(fk); };

    std::atomic<bool> done{false};
    auto start = std::chrono::steady_clock::now();

    const int reportEvery = opt.verbose ? 5 : 10;
    std::thread progress([&] {
        int ticks = 0;
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (done.load()) break;
            if (++ticks % (reportEvery * 4) != 0) continue;
            double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            uint64_t c = state.checked.load();
            std::cout << "[" << (long)el << "s] 已检查 " << groupThousands(c) << " ("
                      << keysPerSec(el > 0 ? c / el : 0) << ")，命中 " << state.found.load() << "\n";
        }
    });

    chosen->run(cfg, state, report);
    done.store(true);
    progress.join();

    double el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    uint64_t total = state.checked.load();
    std::cout << "\n";
    line();
    std::cout << "Backend    : " << chosen->name() << "\n"
              << "Total keys : " << groupThousands(total) << "\n"
              << "Elapsed    : " << (long)el << " s\n"
              << "Keys/s     : " << keysPerSec(el > 0 ? total / el : 0) << "\n"
              << "Matches    : " << state.found.load() << "\n";
    line();
    return 0;
}
