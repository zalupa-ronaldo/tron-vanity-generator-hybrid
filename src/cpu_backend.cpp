#include "backend.h"
#include "crypto.h"
#include "hwdetect.h"
#include "rng.h"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <secp256k1.h>

namespace {

// 每 batch 换一个随机起点，中间用大端自增，省去每次的 RNG 系统调用。
const uint64_t kBatch = 4096;

void incBE(unsigned char* k) {
    for (int i = 31; i >= 0; --i) {
        if (++k[i] != 0) break;
    }
}

struct Ctx {
    secp256k1_context* ctx = nullptr;
    unsigned char sk[32];
    uint64_t inBatch = kBatch;

    bool init() {
        ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        return ctx != nullptr;
    }
    ~Ctx() { if (ctx) secp256k1_context_destroy(ctx); }

    // 生成下一个公钥 (X||Y, 64 字节)；返回 false 表示该私钥无效，需跳过。
    bool next(unsigned char xy[64]) {
        if (inBatch >= kBatch) {
            if (!randBytes(sk, 32)) return false;
            inBatch = 0;
        } else {
            incBE(sk);
        }
        ++inBatch;

        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, sk)) return false;
        unsigned char out[65];
        size_t len = 65;
        secp256k1_ec_pubkey_serialize(ctx, out, &len, &pub, SECP256K1_EC_UNCOMPRESSED);
        std::memcpy(xy, out + 1, 64);
        return true;
    }

    std::string privHex() const { return bytesToHexUpper(sk, 32); }
};

uint64_t threadLoop(const RunConfig& cfg, RunState* state, const ReportFn* report,
                    bool countGlobal) {
    Ctx c;
    if (!c.init()) return 0;

    uint64_t local = 0;
    unsigned char xy[64];

    while (!state->stop.load(std::memory_order_relaxed)) {
        if (cfg.maxAttempts && countGlobal &&
            state->checked.load(std::memory_order_relaxed) >= cfg.maxAttempts) {
            state->stop.store(true);
            break;
        }
        if (c.next(xy)) {
            std::string addr = tronAddressFromPubXY(xy);
            if (report) {
                std::vector<std::string> words = cfg.dictionary ? cfg.dictionary->matchWords(addr) : std::vector<std::string>{};
                MatchResult m = cfg.dictionary ? MatchResult{} : evaluateAddress(addr, cfg.minLen);
                if (!words.empty() || m.matched) {
                    FoundKey fk{addr, c.privHex(), std::move(m), std::move(words)};
                    state->found.fetch_add(1, std::memory_order_relaxed);
                    (*report)(fk);
                }
            }
            (void)addr;
        }
        ++local;
        if (countGlobal) {
            state->checked.fetch_add(1, std::memory_order_relaxed);
            state->cpuChecked.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return local;
}

class CpuBackend : public Backend {
public:
    std::string name() const override { return "CPU"; }
    bool available() const override { return true; }

    BackendInfo info() const override {
        CpuInfo ci = detectCpu();
        unsigned int t = std::thread::hardware_concurrency();
        if (t == 0) t = 4;
        BackendInfo bi;
        bi.kind = "CPU";
        bi.title = ci.brand;
        bi.lines.push_back(std::to_string(t) + " threads");
        bi.lines.push_back(std::string("AVX2      ") + (ci.avx2 ? "OK" : "-"));
        bi.lines.push_back(std::string("AES-NI    ") + (ci.aesni ? "OK" : "-"));
        return bi;
    }

    double benchmark(double seconds) override {
        RunState st;
        unsigned int t = std::thread::hardware_concurrency();
        if (t == 0) t = 4;
        std::vector<std::thread> pool;
        std::vector<uint64_t> counts(t, 0);
        RunConfig cfg;
        auto start = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < t; ++i)
            pool.emplace_back([&, i] { counts[i] = threadLoop(cfg, &st, nullptr, false); });
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        st.stop.store(true);
        for (auto& th : pool) th.join();
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        uint64_t total = 0;
        for (auto c : counts) total += c;
        return elapsed > 0 ? total / elapsed : 0.0;
    }

    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        unsigned int t = cfg.threads ? cfg.threads : std::thread::hardware_concurrency();
        if (t == 0) t = 4;
        std::vector<std::thread> pool;
        for (unsigned int i = 0; i < t; ++i)
            pool.emplace_back([&] { threadLoop(cfg, &state, &report, true); });
        for (auto& th : pool) th.join();
    }
};

}  // namespace

std::unique_ptr<Backend> makeCpuBackend() {
    return std::make_unique<CpuBackend>();
}
