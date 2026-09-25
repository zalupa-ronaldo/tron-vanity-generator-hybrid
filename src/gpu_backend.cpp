#include "backend.h"
#include "crypto.h"
#include "hwdetect.h"
#include "ocl.h"
#include "rng.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <secp256k1.h>

#include "kernel_src.h"

namespace {

// 每次内核启动扫描的连续私钥数。原来是 2^20：在 UHD 730 上单次 GPU kernel 会连续跑
// ~1.3 秒不间断——集成显卡同时还要管显示合成(DWM)，长时间几乎满载跑这种单次超过 1 秒的
// 大批次，被观察到会导致显示子系统卡死到需要硬重启（个别机器/驱动组合下 WDDM 抢占不够
// 及时）。集成 GPU 使用 2^16；独立 GPU по умолчанию получает 2^20,
// чтобы сократить host<->device паузы между kernel-запусками.
// 对吞吐影响可忽略（host<->device 开销本来就在 profile 里测出接近 0）。
constexpr uint32_t kBatch = 1u << 16;
constexpr uint32_t kMaxBatch = 1u << 20;
constexpr uint32_t kEcBits = 20;        // 固定基点标量乘覆盖的位数；> log2(kBatch) 也没问题，
                                        // 只是表里高位窗口永远用不到，不影响正确性
constexpr uint32_t kOutCap = 65536;     // 单批命中回传上限（每项两个 uint）

// 每个 work-item 连续处理多少私钥。UHD 730 上实测 N=1 最快（寄存器压力 + 每个私钥仍需一次
// 模逆，未做 Montgomery 批量求逆）。保留可调，供不同 GPU 找甜点位。
uint32_t g_keysPerItem = 1;
uint32_t g_batch = kBatch;
// 固定基点窗口位宽：1=逐bit(baseline)。UHD 730 实测 ECW=7 甜点位（3 次点加，24KB 表，
// 约 +58%），再大表翻倍收益 <1%。换 GPU 可用 --bench 重新找。
uint32_t g_ecWindow = 7;
// Montgomery 批量求逆的 work-group 大小：1=每 WI 各自求逆（v0.2）。>=2=work-group 内 N 个点
// 共用一次模逆。用 --bench 找甜点位。
uint32_t g_montN = 1;

bool pubXY(secp256k1_context* c, const unsigned char sk[32], unsigned char out64[64]) {
    secp256k1_pubkey p;
    if (!secp256k1_ec_pubkey_create(c, &p, sk)) return false;
    unsigned char o[65];
    size_t l = 65;
    secp256k1_ec_pubkey_serialize(c, o, &l, &p, SECP256K1_EC_UNCOMPRESSED);
    std::memcpy(out64, o + 1, 64);
    return true;
}

void scalarBE(uint32_t s, unsigned char out[32]) {
    std::memset(out, 0, 32);
    out[28] = static_cast<unsigned char>(s >> 24);
    out[29] = static_cast<unsigned char>(s >> 16);
    out[30] = static_cast<unsigned char>(s >> 8);
    out[31] = static_cast<unsigned char>(s);
}

// 生成固定基点预计算表
std::vector<unsigned char> genTable(secp256k1_context* c, uint32_t ecw) {
    std::vector<unsigned char> t;
    if (ecw <= 1) {
        t.assign(32 * 64, 0);
        for (int j = 0; j < 32; ++j) {
            unsigned char sk[32] = {0};
            sk[31 - j / 8] = static_cast<unsigned char>(1u << (j % 8));
            pubXY(c, sk, &t[j * 64]);
        }
        return t;
    }
    uint32_t digits = 1u << ecw;
    uint32_t windows = (kEcBits + ecw - 1) / ecw;
    t.assign(static_cast<size_t>(windows) * digits * 64, 0);
    for (uint32_t w = 0; w < windows; ++w)
        for (uint32_t d = 1; d < digits; ++d) {
            uint64_t scalar = static_cast<uint64_t>(d) << (w * ecw);
            unsigned char sk[32] = {0};
            for (int b = 0; b < 8; ++b) sk[31 - b] = static_cast<unsigned char>(scalar >> (8 * b));
            pubXY(c, sk, &t[(static_cast<size_t>(w) * digits + d) * 64]);
        }
    return t;
}

std::shared_ptr<const Dictionary> emptyDictionary() {
    auto d = std::make_shared<Dictionary>();
    d->words = {"__benchmark__"};
    d->dfa.assign(Dictionary::Alphabet, 0);
    d->outStart = {0};
    d->outLen = {0};
    // OpenCL forbids zero-byte buffers. The benchmark never emits matches,
    // but still needs a valid placeholder for the kernel argument.
    d->outIds = {0};
    return d;
}

class GpuBackend : public Backend {
public:
    explicit GpuBackend(GpuDevice hw) : GpuBackend(std::move(hw), emptyDictionary()) {}
    explicit GpuBackend(GpuDevice hw, std::shared_ptr<const Dictionary> dictionary)
        : hw_(std::move(hw)), dictionary_(std::move(dictionary)) {
        ctx_ = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    }

    ~GpuBackend() override { secp256k1_context_destroy(ctx_); }

    std::string name() const override { return "OpenCL GPU"; }

    BackendInfo info() const override {
        BackendInfo bi;
        bi.kind = "GPU";
        bi.title = hw_.name;
        bi.lines.push_back("OpenCL  " + hw_.platform);
        bi.lines.push_back(std::to_string(hw_.computeUnits) + " CU");
        bi.lines.push_back(std::to_string(hw_.clockMHz) + " MHz");
        bi.lines.push_back(hw_.integrated ? "Integrated GPU (unified memory)" : "Discrete GPU");
        return bi;
    }

    bool available() const override { return const_cast<GpuBackend*>(this)->ensureReady(); }
    std::string note() const override { return err_; }

    double benchmark(double seconds) override {
        if (!ensureReady()) return 0.0;
        benchN(seconds * 0.6, 20);          // 预热：让集成显卡降到稳态
        return benchN(seconds, 20);
    }

    // 完整流水线压测：secp256k1 + keccak + sha + base58 + 匹配 + 回传
    double benchN(double seconds, uint32_t minLen) {
        if (!ensureReady()) return 0.0;
        auto start = std::chrono::steady_clock::now();
        uint64_t done = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
            if (!runBatch(nullptr, nullptr)) break;
            done += g_batch;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return el > 0 ? done / el : 0.0;
    }

    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        if (!ensureReady()) { std::cerr << "GPU backend unavailable: " << err_ << "\n"; return; }
        while (!state.stop.load(std::memory_order_relaxed)) {
            if (cfg.maxAttempts && state.checked.load() >= cfg.maxAttempts) { state.stop = true; break; }
            std::vector<uint32_t> hits;
            unsigned char k0[32];
            if (!runBatch(&hits, k0)) {
                std::cerr << "GPU batch execution failed: " << err_ << "\n";
                state.stop.store(true);
                return;
            }
            bool outputsChanged = false;
            std::unordered_set<uint32_t> processedScalars;
            processedScalars.reserve(hits.size() / 2);
            for (size_t hi = 0; hi + 1 < hits.size(); hi += 2) {
                uint32_t s = hits[hi];
                uint32_t wordId = hits[hi + 1];
                if (wordId >= dictionary_->words.size()) continue;
                if (!processedScalars.insert(s).second) continue;
                unsigned char k[32];
                std::memcpy(k, k0, 32);
                unsigned char tw[32];
                scalarBE(s, tw);
                if (!secp256k1_ec_seckey_tweak_add(ctx_, k, tw)) continue;
                unsigned char pub[64];
                if (!pubXY(ctx_, k, pub)) continue;
                std::string addr = tronAddressFromPubXY(pub);
                auto ids = dictionary_->matchIds(addr);
                if (std::find(ids.begin(), ids.end(), wordId) == ids.end()) continue;
                std::vector<std::string> words;
                for (uint32_t id : ids) {
                    if (id >= dictionary_->words.size()) continue;
                    const uint32_t mask = 1U << (id & 31);
                    if (!cfg.uniquePerWord || (activeWords_[id >> 5] & mask)) {
                        if (cfg.uniquePerWord) {
                            activeWords_[id >> 5] &= ~mask;
                            outputsChanged = true;
                        }
                        words.push_back(dictionary_->words[id]);
                    }
                }
                if (words.empty()) continue;
                MatchResult m{};
                FoundKey fk{addr, bytesToHexUpper(k, 32), std::move(m), std::move(words)};
                state.found.fetch_add(1, std::memory_order_relaxed);
                report(fk);
            }
            if (outputsChanged && !refreshActiveOutputs()) {
                std::cerr << "GPU dictionary update failed: " << err_ << "\n";
                state.stop.store(true);
                return;
            }
            state.checked.fetch_add(g_batch, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(g_batch, std::memory_order_relaxed);
        }
    }

    // ---- 供 --gputest 使用 ----
    static int selfTest(const GpuDevice& hw);

private:
    GpuDevice hw_;
    std::shared_ptr<const Dictionary> dictionary_;
    secp256k1_context* ctx_ = nullptr;
    bool tried_ = false, ready_ = false;
    std::string err_;

    ocl::Program prog_;
    ocl::id kProbe_ = nullptr;
    ocl::id bufTable_ = nullptr, bufP0_ = nullptr, bufCount_ = nullptr, bufOutHits_ = nullptr;
    ocl::id bufDfa_ = nullptr, bufOutStart_ = nullptr, bufOutLen_ = nullptr, bufOutIds_ = nullptr;
    std::vector<unsigned char> table_;
    std::vector<uint32_t> activeWords_;
    std::vector<uint32_t> activeOutLen_;
    std::vector<uint32_t> activeOutIds_;
    uint32_t builtKpi_ = 1;
    uint32_t builtEcw_ = 1;
    uint32_t builtMont_ = 1;

    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
        if (!ocl::load(&err_)) return false;
        if (!hw_.deviceId) { err_ = "no OpenCL device handle"; return false; }
        if (!dictionary_ || dictionary_->words.empty()) { err_ = "dictionary is empty"; return false; }
        builtMont_ = g_montN >= 2 ? g_montN : 1;
        builtKpi_ = builtMont_ >= 2 ? 1 : (g_keysPerItem ? g_keysPerItem : 1);
        while (g_batch % builtKpi_) --builtKpi_;
        builtEcw_ = g_ecWindow ? g_ecWindow : 1;
        std::string opts = "-D KPI=" + std::to_string(builtKpi_) +
                           " -D ECW=" + std::to_string(builtEcw_) +
                           " -D ECBITS=" + std::to_string(kEcBits) +
                           " -D MONT_N=" + std::to_string(builtMont_);
        if (!prog_.build(hw_.platformId, hw_.deviceId, kGpuKernelSource, opts, &err_)) return false;
        kProbe_ = prog_.kernel("tron_vanity_probe", &err_);
        if (!kProbe_) return false;

        table_ = genTable(ctx_, builtEcw_);
        activeWords_.assign((dictionary_->words.size() + 31) / 32, 0xffffffffU);
        activeOutLen_ = dictionary_->outLen;
        activeOutIds_ = dictionary_->outIds;
        bufTable_ = prog_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, table_.size(), table_.data(), &err_);
        bufP0_ = prog_.buffer(ocl::MEM_READ_ONLY, 64, nullptr, &err_);
        bufDfa_ = prog_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                               dictionary_->dfa.size() * sizeof(uint32_t),
                               const_cast<uint32_t*>(dictionary_->dfa.data()), &err_);
        bufOutStart_ = prog_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                    dictionary_->outStart.size() * sizeof(uint32_t),
                                    const_cast<uint32_t*>(dictionary_->outStart.data()), &err_);
        bufOutLen_ = prog_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  activeOutLen_.size() * sizeof(uint32_t),
                                  activeOutLen_.data(), &err_);
        bufOutIds_ = prog_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  activeOutIds_.size() * sizeof(uint32_t),
                                  activeOutIds_.data(), &err_);
        bufCount_ = prog_.buffer(ocl::MEM_READ_WRITE, 4, nullptr, &err_);
        bufOutHits_ = prog_.buffer(ocl::MEM_WRITE_ONLY, kOutCap * 2 * 4, nullptr, &err_);
        if (!bufTable_ || !bufP0_ || !bufCount_ || !bufOutHits_ || !bufDfa_ || !bufOutStart_ || !bufOutLen_ || !bufOutIds_) return false;

        ready_ = true;
        return true;
    }

    bool refreshActiveOutputs() {
        for (size_t state = 0; state < dictionary_->outLen.size(); ++state) {
            const uint32_t start = dictionary_->outStart[state];
            const uint32_t originalLength = dictionary_->outLen[state];
            uint32_t activeLength = 0;
            for (uint32_t j = 0; j < originalLength; ++j) {
                const uint32_t id = dictionary_->outIds[start + j];
                if (id >= dictionary_->words.size()) continue;
                const uint32_t mask = 1U << (id & 31);
                if (activeWords_[id >> 5] & mask)
                    activeOutIds_[start + activeLength++] = id;
            }
            activeOutLen_[state] = activeLength;
        }
        if (!prog_.write(bufOutLen_, activeOutLen_.size() * sizeof(uint32_t), activeOutLen_.data()) ||
            !prog_.write(bufOutIds_, activeOutIds_.size() * sizeof(uint32_t), activeOutIds_.data())) {
            err_ = "updating active OpenCL dictionary outputs failed";
            return false;
        }
        return true;
    }

    // 跑一批：随机基私钥 k0，扫描 k0+[0,kBatch)。命中的 s 写入 outHits（可空）。
    bool runBatch(std::vector<uint32_t>* outHits, unsigned char outK0[32]) {
        unsigned char k0[32], p0[64];
        do {
            if (!randBytes(k0, 32)) return false;
        } while (!pubXY(ctx_, k0, p0));
        if (outK0) std::memcpy(outK0, k0, 32);

        if (!prog_.write(bufP0_, 64, p0)) return false;
        uint32_t zero = 0;
        if (!prog_.write(bufCount_, 4, &zero)) return false;

        uint32_t global = builtMont_ >= 2 ? g_batch : g_batch / builtKpi_;
        size_t local = builtMont_ >= 2 ? builtMont_ : 0;
        if (!prog_.setArg(kProbe_, 0, sizeof(ocl::id), &bufP0_)) return false;
        prog_.setArg(kProbe_, 1, sizeof(ocl::id), &bufTable_);
        prog_.setArg(kProbe_, 2, sizeof(ocl::id), &bufDfa_);
        prog_.setArg(kProbe_, 3, sizeof(ocl::id), &bufOutStart_);
        prog_.setArg(kProbe_, 4, sizeof(ocl::id), &bufOutLen_);
        prog_.setArg(kProbe_, 5, sizeof(ocl::id), &bufOutIds_);
        prog_.setArg(kProbe_, 6, sizeof(ocl::id), &bufCount_);
        prog_.setArg(kProbe_, 7, sizeof(ocl::id), &bufOutHits_);
        uint32_t cap = kOutCap;
        prog_.setArg(kProbe_, 8, sizeof(uint32_t), &cap);

        if (!prog_.run1D(kProbe_, global, local, &err_)) return false;
        if (!prog_.finish()) return false;

        if (outHits) {
            uint32_t cnt = 0;
            prog_.read(bufCount_, 4, &cnt);
            if (cnt > kOutCap) cnt = kOutCap;
            outHits->resize(cnt * 2);
            if (cnt) prog_.read(bufOutHits_, cnt * 2 * 4, outHits->data());
        }
        return true;
    }
};

int GpuBackend::selfTest(const GpuDevice& hw) {
    std::string err;
    if (!ocl::load(&err)) { std::cout << "OpenCL load failed: " << err << "\n"; return 1; }
    ocl::Program prog;
    if (!prog.build(hw.platformId, hw.deviceId, kGpuKernelSource, "", &err)) {
        std::cout << err << "\n";
        return 1;
    }
    ocl::id kTest = prog.kernel("test_pub", &err);
    if (!kTest) { std::cout << err << "\n"; return 1; }

    secp256k1_context* c = secp256k1_context_create(SECP256K1_CONTEXT_NONE);

    std::vector<unsigned char> table(32 * 64, 0);
    for (int j = 0; j < 32; ++j) {
        unsigned char sk[32] = {0};
        sk[31 - j / 8] = static_cast<unsigned char>(1u << (j % 8));
        pubXY(c, sk, &table[j * 64]);
    }
    unsigned char k0[32], p0[64];
    do { randBytes(k0, 32); } while (!pubXY(c, k0, p0));

    std::vector<uint32_t> scalars = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 255, 256,
                                     1000, 65535, 65536, 123456, 7777777, 0x7FFFFFFF, 0xFFFFFFFF};
    uint32_t n = static_cast<uint32_t>(scalars.size());
    std::vector<unsigned char> pubout(n * 64, 0);

    ocl::id bT = prog.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, table.size(), table.data(), &err);
    ocl::id bP = prog.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, 64, p0, &err);
    ocl::id bS = prog.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, n * 4, scalars.data(), &err);
    ocl::id bO = prog.buffer(ocl::MEM_WRITE_ONLY, n * 64, nullptr, &err);
    prog.setArg(kTest, 0, sizeof(ocl::id), &bP);
    prog.setArg(kTest, 1, sizeof(ocl::id), &bT);
    prog.setArg(kTest, 2, sizeof(ocl::id), &bS);
    prog.setArg(kTest, 3, sizeof(ocl::id), &bO);
    prog.setArg(kTest, 4, sizeof(uint32_t), &n);
    if (!prog.run1D(kTest, n, 0, &err)) { std::cout << err << "\n"; return 1; }
    prog.finish();
    prog.read(bO, n * 64, pubout.data());

    int fails = 0;
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char k[32];
        std::memcpy(k, k0, 32);
        unsigned char tw[32];
        scalarBE(scalars[i], tw);
        unsigned char cpuPub[64];
        bool ok = secp256k1_ec_seckey_tweak_add(c, k, tw) && pubXY(c, k, cpuPub);
        bool match = ok && std::memcmp(cpuPub, &pubout[i * 64], 64) == 0;
        std::string cpuAddr = ok ? tronAddressFromPubXY(cpuPub) : "?";
        std::string gpuAddr = tronAddressFromPubXY(&pubout[i * 64]);
        std::cout << (match ? "  OK  " : "  FAIL") << "  s=" << scalars[i]
                  << "  cpu=" << cpuAddr << "  gpu=" << gpuAddr << "\n";
        if (!match) ++fails;
    }
    std::cout << (fails ? std::to_string(fails) + " mismatches\n"
                        : "test_pub: GPU secp256k1/keccak/base58 matches CPU\n");

    // ---- Montgomery 批量求逆：对照 libsecp256k1（其自带模逆 = 逐点单独求逆基准）----
    const uint32_t ecw = 7;
    std::vector<unsigned char> ct = genTable(c, ecw);
    std::cout << "\ntest_mont (ECW=" << ecw << ", batch inversion vs point-by-point):\n";
    for (uint32_t N : {2u, 4u, 8u, 16u, 32u}) {
        ocl::Program pm;
        std::string opts = "-D KPI=1 -D ECW=" + std::to_string(ecw) +
                           " -D ECBITS=" + std::to_string(kEcBits) +
                           " -D MONT_N=" + std::to_string(N);
        if (!pm.build(hw.platformId, hw.deviceId, kGpuKernelSource, opts, &err)) {
            std::cout << "  N=" << N << " compile failed:\n" << err << "\n"; ++fails; continue;
        }
        ocl::id km = pm.kernel("test_mont", &err);
        if (!km) { std::cout << "  N=" << N << ": " << err << "\n"; ++fails; continue; }

        uint32_t cnt = 8 * N;                       // 多个 work-group
        std::vector<uint32_t> sc(cnt);
        for (uint32_t i = 0; i < cnt; ++i) sc[i] = (i * 2654435761u) & ((1u << kEcBits) - 1);
        std::vector<unsigned char> po(cnt * 64, 0);

        ocl::id mT = pm.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, ct.size(), ct.data(), &err);
        ocl::id mP = pm.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, 64, p0, &err);
        ocl::id mS = pm.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, cnt * 4, sc.data(), &err);
        ocl::id mO = pm.buffer(ocl::MEM_WRITE_ONLY, cnt * 64, nullptr, &err);
        pm.setArg(km, 0, sizeof(ocl::id), &mP);
        pm.setArg(km, 1, sizeof(ocl::id), &mT);
        pm.setArg(km, 2, sizeof(ocl::id), &mS);
        pm.setArg(km, 3, sizeof(ocl::id), &mO);
        pm.setArg(km, 4, sizeof(uint32_t), &cnt);
        if (!pm.run1D(km, cnt, N, &err)) { std::cout << "  N=" << N << " run: " << err << "\n"; ++fails; continue; }
        pm.finish();
        pm.read(mO, cnt * 64, po.data());

        int bad = 0;
        for (uint32_t i = 0; i < cnt; ++i) {
            unsigned char k[32];
            std::memcpy(k, k0, 32);
            unsigned char tw[32];
            scalarBE(sc[i], tw);
            unsigned char cpuPub[64];
            if (!(secp256k1_ec_seckey_tweak_add(c, k, tw) && pubXY(c, k, cpuPub)) ||
                std::memcmp(cpuPub, &po[i * 64], 64) != 0)
                ++bad;
        }
        std::cout << "  N=" << N << " : " << (bad ? std::to_string(bad) + " / " + std::to_string(cnt) + " mismatches"
                                                  : std::to_string(cnt) + " vectors match") << "\n";
        if (bad) ++fails;
    }

    std::cout << (fails ? "\nGPU validation failed\n" : "\nAll GPU checks passed\n");
    return fails ? 1 : 0;
}

}  // namespace

std::unique_ptr<Backend> makeGpuBackend(const GpuDevice& dev, std::shared_ptr<const Dictionary> dictionary) {
    return std::make_unique<GpuBackend>(dev, std::move(dictionary));
}

int gpuSelfTest(const GpuDevice& dev) { return GpuBackend::selfTest(dev); }

void gpuSetKeysPerItem(uint32_t n) { if (n) g_keysPerItem = n; }
void gpuSetBatch(uint32_t n) {
    if (n < 1024) n = 1024;
    if (n > kMaxBatch) n = kMaxBatch;
    uint32_t p = 1;
    while ((p << 1) <= n) p <<= 1;
    g_batch = p;
}
void gpuSetEcWindow(uint32_t w) { if (w >= 1 && w <= 8) g_ecWindow = w; }
void gpuSetMontN(uint32_t n) { if (n >= 1 && n <= 64) g_montN = n; }

// --profile：逐阶段消融，估算内核各阶段占比
int gpuProfile(const GpuDevice& dev, double secs) {
    std::string err;
    if (!ocl::load(&err)) { std::cout << err << "\n"; return 1; }
    secp256k1_context* c = secp256k1_context_create(SECP256K1_CONTEXT_NONE);

    uint32_t ecw = g_ecWindow ? g_ecWindow : 1;
    std::vector<unsigned char> table = genTable(c, ecw);
    unsigned char k0[32], p0[64];
    do { randBytes(k0, 32); } while (!pubXY(c, k0, p0));

    const uint32_t kb = 1u << 20;
    const char* names[7] = { "", "EC 标量乘 (固定基点窗口法)", "模逆 + 转仿射", "Keccak-256",
                             "SHA-256d (校验和)", "Base58 尾部", "尾号匹配" };
    double nsPerKey[7] = {0};

    double hostNs = 0;

    std::cout << "\n== TRON GPU Profile (" << dev.name << ", KPI=1, ECW=" << ecw << ") ==\n"
              << "逐阶段消融：每阶段单独编译内核，预热到稳态后计时。\n\n";

    for (int stage = 1; stage <= 6; ++stage) {
        ocl::Program p;
        std::string opts = "-D KPI=1 -D ECW=" + std::to_string(ecw) +
                           " -D ECBITS=" + std::to_string(kEcBits) +
                           " -D PROF_STAGE=" + std::to_string(stage);
        if (!p.build(dev.platformId, dev.deviceId, kGpuKernelSource, opts, &err)) {
            std::cout << "stage " << stage << " 编译失败:\n" << err << "\n";
            return 1;
        }
        ocl::id k = p.kernel("prof", &err);
        ocl::id bT = p.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, table.size(), table.data(), &err);
        ocl::id bPp = p.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, 64, p0, &err);
        ocl::id bS = p.buffer(ocl::MEM_WRITE_ONLY, kb * 4, nullptr, &err);
        if (!k || !bT || !bPp || !bS) { std::cout << "缓冲/内核创建失败: " << err << "\n"; return 1; }
        p.setArg(k, 0, sizeof(ocl::id), &bPp);
        p.setArg(k, 1, sizeof(ocl::id), &bT);
        p.setArg(k, 2, sizeof(ocl::id), &bS);

        auto measure = [&](double dur) -> double {
            auto st = std::chrono::steady_clock::now();
            uint64_t done = 0;
            double el;
            do {
                if (!p.run1D(k, kb, 0, &err) || !p.finish()) return 0.0;
                done += kb;
                el = std::chrono::duration<double>(std::chrono::steady_clock::now() - st).count();
            } while (el < dur);
            return el > 0 ? done / el : 0.0;
        };
        measure(secs * 0.6);
        double kps = measure(secs);
        nsPerKey[stage] = kps > 0 ? 1e9 / kps : 0;
        std::printf("  [%d/6] %-22s  %.0f keys/s\n", stage, names[stage], kps);

        if (stage == 6) {
            // host 侧开销：randBytes + pubXY + 上传 P0 + 读回计数（不启动内核）
            const int iters = 300;
            auto st = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                unsigned char kk[32], pp[64];
                do { randBytes(kk, 32); } while (!pubXY(c, kk, pp));
                p.write(bPp, 64, pp);
                uint32_t tmp;
                p.read(bS, 4, &tmp);
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - st).count();
            hostNs = el / iters / kb * 1e9;
        }
    }

    double total = nsPerKey[6] + hostNs;
    std::cout << "\n阶段                       本阶段 ns/key    占比\n";
    std::cout << "  " << std::string(48, '-') << "\n";
    double prev = 0, maxPct = 0;
    int maxI = 1;
    for (int s = 1; s <= 6; ++s) {
        double d = nsPerKey[s] - prev;
        if (d < 0) d = 0;
        double pct = total > 0 ? d / total * 100 : 0;
        std::printf("  %-24s  %9.0f      %5.1f%%\n", names[s], d, pct);
        if (pct > maxPct) { maxPct = pct; maxI = s; }
        prev = nsPerKey[s];
    }
    {
        double pct = total > 0 ? hostNs / total * 100 : 0;
        std::printf("  %-24s  %9.0f      %5.1f%%\n", "Host <-> Device + 私钥", hostNs, pct);
        if (pct > maxPct) { maxPct = pct; maxI = 0; }
    }
    std::cout << "  " << std::string(48, '-') << "\n";
    std::printf("  合计 ~%.0f ns/key  (~%.0f keys/s)\n", total, total > 0 ? 1e9 / total : 0);
    std::printf("\n结论: 最大头是 %s (~%.0f%%)，优先攻这里。\n",
                maxI == 0 ? "Host<->Device" : names[maxI], maxPct);
    std::cout << "（消融法为近似：加阶段会改变寄存器占用，百分比看量级即可）\n";
    return 0;
}

// --bench：EC 窗口法 × keys-per-item 矩阵 + 尾号规则吞吐
// 每档单独编译内核、重新生成预计算表，预热到稳态再计时。
int gpuBench(const GpuDevice& dev, double secs) {
    std::cout << "\nLegacy OpenCL tuning\n"
              << "  Warm-up: " << secs << " s | timed: " << secs
              << " s | steady-state throughput\n";

    uint32_t savedKpi = g_keysPerItem, savedEcw = g_ecWindow, savedMont = g_montN, savedBatch = g_batch;

    std::cout << "\n[1/5] Fixed-base EC window scan (KPI=1, min_len=20)\n";
    std::cout << "  ECW=1 baseline; ECW>=2 uses comb precomputation\n";
    g_keysPerItem = 1;
    uint32_t bestEcw = 1;
    double bestR = 0;
    double baseR = 0;
    for (uint32_t w : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        g_ecWindow = w;
        GpuBackend b(dev);
        if (!b.available()) { std::cout << "  ECW=" << w << " : build failed\n"; continue; }
        b.benchN(secs, 20);
        double r = b.benchN(secs, 20);
        if (w == 1) baseR = r;
        uint32_t windows = (kEcBits + w - 1) / w;
        uint32_t kb = (w == 1 ? 2u : windows * (1u << w) * 64 / 1024);
        std::printf("  ECW=%u  windows=%-2u table=%3u KB : %10.0f keys/s   %+5.0f%%%s\n",
                    w, windows, kb, r, baseR > 0 ? (r - baseR) / baseR * 100 : 0,
                    r > bestR ? "  <-" : "");
        if (r > bestR) { bestR = r; bestEcw = w; }
    }
    std::printf("  Best ECW: %u  (%.0f keys/s)\n", bestEcw, bestR);

    std::cout << "\n[2/5] Montgomery batch inversion scan (ECW=" << bestEcw << ")\n";
    g_ecWindow = bestEcw;
    g_keysPerItem = 1;
    double montBase = 0, bestMontR = 0;
    uint32_t bestMont = 1;
    for (uint32_t nn : {1u, 2u, 4u, 8u, 16u, 32u}) {
        g_montN = nn;
        GpuBackend b(dev);
        if (!b.available()) { std::cout << "  N=" << nn << " : build failed\n"; continue; }
        b.benchN(secs, 20);
        double r = b.benchN(secs, 20);
        if (nn == 1) montBase = r;
        std::printf("  N=%-3u : %10.0f keys/s   %+5.0f%%%s\n", nn, r,
                    montBase > 0 ? (r - montBase) / montBase * 100 : 0,
                    r > bestMontR ? "  <-" : "");
        if (r > bestMontR) { bestMontR = r; bestMont = nn; }
    }
    std::printf("  Best MONT_N: %u  (%.0f keys/s)\n", bestMont, bestMontR);

    std::cout << "\n[3/5] Keys-per-item scan (ECW=" << bestEcw << ", MONT_N=1)\n";
    g_montN = 1;
    double bestAll = std::max({bestR, bestMontR});
    for (uint32_t n : {1u, 2u, 4u, 8u}) {
        g_keysPerItem = n;
        GpuBackend b(dev);
        if (!b.available()) continue;
        b.benchN(secs, 20);
        double r = b.benchN(secs, 20);
        std::printf("  N=%-3u : %10.0f keys/s\n", n, r);
        bestAll = std::max(bestAll, r);
    }
    g_keysPerItem = 1;

    std::cout << "\n[4/5] Match-rule overhead (ECW=" << bestEcw << ", KPI=1)\n";
    g_keysPerItem = 1;
    GpuBackend b(dev);
    b.available();
    b.benchN(secs, 5);
    for (uint32_t ml : {5u, 6u, 7u, 8u}) {
        double r = b.benchN(secs, ml);
        std::printf("  suffix match >= %u chars : %10.0f keys/s\n", ml, r);
        bestAll = std::max(bestAll, r);
    }

    std::cout << "\n[5/5] GPU batch-size scan (ECW=" << bestEcw << ", KPI=1, MONT_N=1)\n";
    g_ecWindow = bestEcw;
    g_keysPerItem = 1;
    g_montN = 1;
    for (uint32_t batch : {1u << 14, 1u << 16, 1u << 18, 1u << 20}) {
        g_batch = batch;
        GpuBackend batchBackend(dev);
        if (!batchBackend.available()) continue;
        batchBackend.benchN(secs, 20);
        double r = batchBackend.benchN(secs, 20);
        std::printf("  batch=2^%u (%u) : %10.0f keys/s\n",
                    static_cast<unsigned>(std::log2(batch)), batch, r);
        bestAll = std::max(bestAll, r);
    }

    g_keysPerItem = savedKpi;
    g_ecWindow = savedEcw;
    g_montN = savedMont;
    g_batch = savedBatch;
    std::printf("\n  Legacy OpenCL best: %.0f keys/s\n", bestAll);
    return static_cast<int>(bestAll);
}
