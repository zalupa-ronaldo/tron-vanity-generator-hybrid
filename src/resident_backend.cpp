#include "resident_backend.h"

#include "crypto.h"
#include "hwdetect.h"
#include "ocl.h"
#include "rng.h"
#include "resident_protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <secp256k1.h>

#include "kernel_src.h"

namespace {

constexpr uint32_t kRecordBytes = resident_protocol::kRecordBytes;
constexpr uint32_t kMaxMatches = resident_protocol::kMaxMatches;
constexpr uint32_t kMetaWords = resident_protocol::kMetaWords;
constexpr uint32_t kResidentEcbits = 256;
constexpr uint32_t kResidentEcw = 8;
constexpr uint32_t kResidentKpi = 2;
constexpr uint32_t kResidentLocalSize = 128;

std::vector<unsigned char> genResidentTable(secp256k1_context* c) {
    const uint32_t digits = 1u << kResidentEcw;
    const uint32_t windows = (kResidentEcbits + kResidentEcw - 1) / kResidentEcw;
    std::vector<unsigned char> table(static_cast<size_t>(windows) * digits * 64, 0);
    for (uint32_t w = 0; w < windows; ++w) {
        for (uint32_t d = 1; d < digits; ++d) {
            unsigned char sk[32] = {0};
            uint32_t value = d;
            for (uint32_t bit = 0; bit < kResidentEcw; ++bit) {
                uint32_t absolute = w * kResidentEcw + bit;
                if (absolute < kResidentEcbits && ((value >> bit) & 1u))
                    sk[31 - absolute / 8] |= static_cast<unsigned char>(1u << (absolute & 7));
            }
            secp256k1_pubkey p;
            if (!secp256k1_ec_pubkey_create(c, &p, sk)) continue;
            unsigned char encoded[65];
            size_t len = sizeof(encoded);
            secp256k1_ec_pubkey_serialize(c, encoded, &len, &p, SECP256K1_EC_UNCOMPRESSED);
            std::memcpy(&table[(static_cast<size_t>(w) * digits + d) * 64], encoded + 1, 64);
        }
    }
    return table;
}

uint32_t read32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (i * 8);
    return v;
}

std::string hexUpper(const unsigned char* p, size_t n) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(hex[p[i] >> 4]);
        out.push_back(hex[p[i] & 15]);
    }
    return out;
}

class ResidentGpuBackend final : public Backend {
public:
    ResidentGpuBackend(GpuDevice device, std::shared_ptr<const Dictionary> dictionary,
                       std::string rng, uint32_t bufferMiB,
                       uint32_t chunkMs, uint32_t pollMs)
        : device_(std::move(device)), dictionary_(std::move(dictionary)),
          rng_(std::move(rng)), bufferMiB_(std::max(8u, bufferMiB)),
          chunkMs_(std::clamp(chunkMs, 8u, 100u)),
          pollMs_(std::clamp(pollMs, 10u, 1000u)) {
        context_ = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        workItems_ = std::clamp((1u << 18) * chunkMs_ / 32u, 1u << 14, 1u << 20);
    }

    std::string name() const override { return "OpenCL GPU resident"; }

    BackendInfo info() const override {
        BackendInfo out;
        out.kind = "GPU-resident";
        out.title = device_.name;
        out.lines.push_back("OpenCL " + device_.platform);
        out.lines.push_back("CSPRNG " + rng_ + ", chunk " + std::to_string(chunkMs_) + " ms");
        out.lines.push_back(std::to_string(bufferMiB_) + " MiB device result ring");
        return out;
    }

    bool available() const override { return const_cast<ResidentGpuBackend*>(this)->ensureReady(); }
    std::string note() const override { return error_; }

    double benchmark(double seconds) override {
        if (!ensureReady()) return 0.0;
        auto start = std::chrono::steady_clock::now();
        uint64_t generated = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
            if (!runChunk(nullptr, nullptr, nullptr)) break;
            generated += static_cast<uint64_t>(workItems_) * kResidentKpi;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return elapsed > 0 ? generated / elapsed : 0.0;
    }

    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        if (!ensureReady()) {
            std::cerr << "GPU resident unavailable: " << error_ << "\n";
            return;
        }
        auto lastPoll = std::chrono::steady_clock::now();
        while (!state.stop.load(std::memory_order_relaxed)) {
            if (cfg.maxAttempts && state.checked.load() >= cfg.maxAttempts) break;
            uint32_t produced = 0, overflow = 0;
            ReportFn counted = [&](const FoundKey& key) {
                state.found.fetch_add(1, std::memory_order_relaxed);
                report(key);
            };
            if (!runChunk(&produced, &overflow, counted)) {
                std::cerr << "GPU resident chunk failed: " << error_ << "\n";
                state.stop.store(true);
                return;
            }
            state.checked.fetch_add(static_cast<uint64_t>(workItems_) * kResidentKpi, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(static_cast<uint64_t>(workItems_) * kResidentKpi, std::memory_order_relaxed);
            if (overflow) {
                std::cerr << "GPU resident result ring overflow; stopping to avoid lost matches\n";
                state.stop.store(true);
                return;
            }
            auto now = std::chrono::steady_clock::now();
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPoll).count();
            // runChunk already synchronizes and polls the metadata. Do not add
            // an unconditional delay after a productive chunk: that throttles
            // high-match dictionaries and can waste a large fraction of GPU time.
            if (produced == 0 && elapsedMs < pollMs_)
                std::this_thread::sleep_for(std::chrono::milliseconds(pollMs_ - elapsedMs));
            lastPoll = std::chrono::steady_clock::now();
        }
    }

private:
    GpuDevice device_;
    std::shared_ptr<const Dictionary> dictionary_;
    std::string rng_;
    uint32_t bufferMiB_, chunkMs_, pollMs_;
    uint32_t workItems_ = 1u << 18;
    uint32_t ringSlots_ = 0;
    uint64_t streamBase_ = 0;
    secp256k1_context* context_ = nullptr;
    bool tried_ = false, ready_ = false;
    std::string error_;
    ocl::Program program_;
    ocl::id kernel_ = nullptr;
    ocl::id seed_ = nullptr, table_ = nullptr, dfa_ = nullptr;
    ocl::id outStart_ = nullptr, outLen_ = nullptr, outIds_ = nullptr;
    ocl::id meta_ = nullptr, records_ = nullptr;
    std::array<unsigned char, 32> seedBytes_{};
    uint32_t readPos_ = 0;

    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
        if (rng_ != "chacha12" && rng_ != "philox" && rng_ != "aes-ctr") {
            error_ = "resident RNG must be chacha12, aes-ctr, or philox";
            return false;
        }
        if (!ocl::load(&error_)) return false;
        if (!device_.deviceId || !dictionary_ || dictionary_->words.empty()) {
            error_ = "missing OpenCL device or dictionary";
            return false;
        }
        const int rngMode = rng_ == "philox" ? 2 : (rng_ == "aes-ctr" ? 3 : 1);
        std::string opts = "-D RESIDENT=1 -D RESIDENT_RNG=" + std::to_string(rngMode) +
                           " -D ECW=" + std::to_string(kResidentEcw) +
                           " -D ECBITS=" + std::to_string(kResidentEcbits) +
                           " -D KPI=" + std::to_string(kResidentKpi) + " -D MONT_N=1";
        if (!program_.build(device_.platformId, device_.deviceId, kGpuKernelSource, opts, &error_)) return false;
        kernel_ = program_.kernel("tron_vanity_resident", &error_);
        if (!kernel_) return false;
        if (!randBytes(seedBytes_.data(), seedBytes_.size())) {
            error_ = "OS CSPRNG seed generation failed";
            return false;
        }
        auto table = genResidentTable(context_);
        seed_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, seedBytes_.size(), seedBytes_.data(), &error_);
        table_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, table.size(), table.data(), &error_);
        dfa_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                               dictionary_->dfa.size() * sizeof(uint32_t),
                               const_cast<uint32_t*>(dictionary_->dfa.data()), &error_);
        outStart_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                    dictionary_->outStart.size() * sizeof(uint32_t),
                                    const_cast<uint32_t*>(dictionary_->outStart.data()), &error_);
        outLen_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  dictionary_->outLen.size() * sizeof(uint32_t),
                                  const_cast<uint32_t*>(dictionary_->outLen.data()), &error_);
        outIds_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  dictionary_->outIds.size() * sizeof(uint32_t),
                                  const_cast<uint32_t*>(dictionary_->outIds.data()), &error_);
        ringSlots_ = (bufferMiB_ * 1024u * 1024u) / kRecordBytes;
        meta_ = program_.buffer(ocl::MEM_READ_WRITE, kMetaWords * sizeof(uint32_t), nullptr, &error_);
        records_ = program_.buffer(ocl::MEM_READ_WRITE,
                                   static_cast<size_t>(ringSlots_) * kRecordBytes, nullptr, &error_);
        std::array<uint32_t, kMetaWords> zero{};
        if (!program_.write(meta_, sizeof(zero), zero.data()) ||
            !seed_ || !table_ || !dfa_ || !outStart_ || !outLen_ || !outIds_ || !meta_ || !records_) return false;
        ready_ = true;
        return true;
    }

    bool bindKernel() {
        uint64_t stream = streamBase_;
        return program_.setArg(kernel_, 0, sizeof(ocl::id), &seed_) &&
               program_.setArg(kernel_, 1, sizeof(ocl::id), &table_) &&
               program_.setArg(kernel_, 2, sizeof(ocl::id), &dfa_) &&
               program_.setArg(kernel_, 3, sizeof(ocl::id), &outStart_) &&
               program_.setArg(kernel_, 4, sizeof(ocl::id), &outLen_) &&
               program_.setArg(kernel_, 5, sizeof(ocl::id), &outIds_) &&
               program_.setArg(kernel_, 6, sizeof(ocl::id), &meta_) &&
               program_.setArg(kernel_, 7, sizeof(ocl::id), &records_) &&
               program_.setArg(kernel_, 8, sizeof(uint32_t), &ringSlots_) &&
               program_.setArg(kernel_, 9, sizeof(uint64_t), &stream);
    }

    bool runChunk(uint32_t* produced, uint32_t* overflow, const ReportFn& report) {
        if (!bindKernel()) { error_ = "setting resident kernel arguments failed"; return false; }
        if (!program_.run1D(kernel_, workItems_, kResidentLocalSize, &error_) || !program_.finish()) return false;
        std::array<uint32_t, kMetaWords> meta{};
        if (!program_.read(meta_, sizeof(meta), meta.data())) { error_ = "reading resident metadata failed"; return false; }
        const uint32_t writePos = meta[0];
        const uint32_t available = writePos - readPos_;
        const uint32_t count = std::min(available, ringSlots_);
        if (produced) *produced = count;
        if (overflow) *overflow = meta[2];
        if (report && count) {
            std::vector<unsigned char> bytes(static_cast<size_t>(count) * kRecordBytes);
            uint32_t first = std::min(count, ringSlots_ - (readPos_ % ringSlots_));
            if (!program_.readAt(records_, static_cast<size_t>(readPos_ % ringSlots_) * kRecordBytes,
                                 static_cast<size_t>(first) * kRecordBytes, bytes.data())) return false;
            if (first < count && !program_.readAt(records_, 0,
                                                  static_cast<size_t>(count - first) * kRecordBytes,
                                                  bytes.data() + static_cast<size_t>(first) * kRecordBytes)) return false;
            for (uint32_t i = 0; i < count; ++i) {
                const unsigned char* rec = bytes.data() + static_cast<size_t>(i) * kRecordBytes;
                FoundKey key;
                key.privHex = hexUpper(rec, 32);
                key.address.assign(reinterpret_cast<const char*>(rec + 32), 34);
                uint32_t n = std::min(read32(rec + 68), kMaxMatches);
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t id = read32(rec + 72 + j * 4);
                    if (id < dictionary_->words.size()) key.words.push_back(dictionary_->words[id]);
                }
                if (!key.words.empty()) report(key);
            }
        }
        readPos_ = writePos;
        if (!program_.writeAt(meta_, sizeof(uint32_t), sizeof(uint32_t), &readPos_)) return false;
        streamBase_ += workItems_ / kResidentLocalSize;
        return true;
    }
};

} // namespace

std::unique_ptr<Backend> makeResidentGpuBackend(
    const GpuDevice& device, std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng, uint32_t bufferMiB, uint32_t chunkMs, uint32_t pollMs) {
    return std::make_unique<ResidentGpuBackend>(device, std::move(dictionary), rng,
                                                bufferMiB, chunkMs, pollMs);
}
