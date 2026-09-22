#include "resident_backend.h"

#include "crypto.h"
#include "hwdetect.h"
#include "ocl.h"
#include "cuda_driver.h"
#include "rng.h"
#include "resident_protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <secp256k1.h>

#include "kernel_src.h"

namespace {

constexpr uint32_t kRecordBytes = resident_protocol::kRecordBytes;
constexpr uint32_t kMetaWords = resident_protocol::kMetaWords;
constexpr uint32_t kResidentEcbits = 32;
constexpr uint32_t kResidentEcw = 8;
constexpr uint32_t kResidentKpi = 2;
constexpr uint32_t kResidentLocalSize = 256;

std::string openclResidentBuildOptions(const std::string& rng, const OpenclResidentOptions& options) {
    const int mode = rng == "philox" ? 2 : (rng == "aes-ctr" ? 3 : 1);
    return "-cl-std=CL1.2 -D RESIDENT=1 -D ECW=8 -D ECBITS=32 -D KPI=2 -D MONT_N=1 -D RESIDENT_RNG=" +
        std::to_string(mode) + " -D RESIDENT_PAIR_INVERSE=" + (options.pairInverse ? "1" : "0") +
        " -D OPENCL_COMPACT=" + (options.compact ? "1" : "0") +
        (options.hostSeed ? " -D RESIDENT_SCAN_ONLY=1" : "");
}

struct PrivateScalar {
    std::array<unsigned char, 32> bytes{};
    unsigned char* data() { return bytes.data(); }
    size_t size() const { return bytes.size(); }
    ~PrivateScalar() {
        volatile unsigned char* p = bytes.data();
        for (size_t i = 0; i < bytes.size(); ++i) p[i] = 0;
    }
};

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

std::string elapsedText(std::chrono::steady_clock::time_point started) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        << " s";
    return out.str();
}

template<class Program>
class ResidentGpuBackend final : public Backend {
    static constexpr bool isCuda = std::is_same_v<Program, cuda::Program>;
public:
    ResidentGpuBackend(GpuDevice device, std::shared_ptr<const Dictionary> dictionary,
                       std::string rng, uint32_t bufferMiB,
                       uint32_t chunkMs, uint32_t pollMs, uint32_t groupSize,
                       OpenclResidentOptions options = {})
        : device_(std::move(device)), dictionary_(std::move(dictionary)),
          rng_(std::move(rng)), bufferMiB_(std::max(8u, bufferMiB)),
          chunkMs_(std::clamp(chunkMs, 8u, 100u)),
          groupSize_(groupSize == 64 || groupSize == 128 || groupSize == 256 ? groupSize : 256),
          openclOptions_(options) {
        (void)pollMs; // retained for CLI/API compatibility; dispatch completion is the poll
        context_ = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        workItems_ = isCuda ? std::clamp((1u << 18) * chunkMs_ / 32u, 1u << 14, 1u << 20) : 1u << 14;
    }

    ~ResidentGpuBackend() override { secp256k1_context_destroy(context_); }

    std::string name() const override { return isCuda ? "CUDA GPU resident" : "OpenCL GPU resident"; }

    BackendInfo info() const override {
        BackendInfo out;
        out.kind = isCuda ? "CUDA-resident" : "GPU-resident";
        out.title = device_.name;
        out.lines.push_back((isCuda ? "CUDA " : "OpenCL ") + device_.platform);
        out.lines.push_back((openclOptions_.hostSeed ? "OS CSPRNG + CPU base expansion + GPU scan" :
                            "GPU CSPRNG + CPU base expansion + GPU scan, " + rng_) +
                            ", chunk " +
                            std::to_string(chunkMs_) + " ms");
        out.lines.push_back(std::to_string(bufferMiB_) + " MiB device result ring");
        if constexpr (!isCuda) out.lines.push_back(std::string("compiler ") +
            (openclOptions_.compact ? "compact" : "default") + ", inversion " +
            (openclOptions_.pairInverse ? "paired" : "single") + ", adaptive dispatch");
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
            generated += lastChunkKeys_;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return elapsed > 0 ? generated / elapsed : 0.0;
    }

    OpenclProfileResult profile(double seconds) {
        if (!ensureReady()) { profile_.error = error_; return profile_; }
        // Warm up code/caches and let bounded dispatch size adapt before timing.
        const auto warm = std::chrono::steady_clock::now();
        do {
            if (!runChunk(nullptr, nullptr, nullptr)) { profile_.error = error_; return profile_; }
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - warm).count() < 0.5);
        profile_ = {};
        const auto start = std::chrono::steady_clock::now();
        benchmark(seconds);
        profile_.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        profile_.error = error_;
        return profile_;
    }

    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        if (!ensureReady()) {
            std::cerr << "GPU resident unavailable: " << error_ << "\n";
            return;
        }
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
            state.checked.fetch_add(lastChunkKeys_, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(lastChunkKeys_, std::memory_order_relaxed);
            if (overflow) {
                std::cerr << "GPU resident result ring overflow; stopping to avoid lost matches\n";
                state.stop.store(true);
                return;
            }
            // Bounded dispatches already synchronize and drain the ring. A
            // post-dispatch polling sleep would idle fast GPUs on rare matches.
        }
    }

    bool selfTest() {
        if (!ensureReady()) return false;
        workItems_ = 256;
        adaptive_ = false;
        pruneOutputs_ = false;
        std::unordered_set<std::string> addresses;
        for (int batch = 0; batch < 2; ++batch) {
            uint32_t produced = 0, overflow = 0;
            if (!runChunk(&produced, &overflow, [&](const FoundKey& key) {
                    addresses.insert(key.address);
                }) || overflow || produced != workItems_ * kResidentKpi) {
                if (error_.empty()) error_ = "resident self-test record count/overflow mismatch: expected " +
                    std::to_string(workItems_ * kResidentKpi) + ", got " + std::to_string(produced) +
                    ", overflow " + std::to_string(overflow);
                return false;
            }
        }
        if (addresses.size() != workItems_ * kResidentKpi * 2) {
            error_ = "resident self-test found duplicate addresses";
            return false;
        }
        return true;
    }

private:
    GpuDevice device_;
    std::shared_ptr<const Dictionary> dictionary_;
    std::string rng_;
    uint32_t bufferMiB_, chunkMs_;
    uint32_t workItems_ = 1u << 18;
    uint32_t ringSlots_ = 0;
    uint32_t groupSize_ = kResidentLocalSize;
    OpenclResidentOptions openclOptions_;
    OpenclProfileResult profile_;
    uint64_t lastChunkKeys_ = 0;
    bool adaptive_ = !isCuda;
    uint64_t rngCounter_ = 0;
    uint64_t sequenceBase_ = 0;
    secp256k1_context* context_ = nullptr;
    bool tried_ = false, ready_ = false, pruneOutputs_ = true;
    std::string error_;
    Program program_;
    ocl::id seedKernel_ = nullptr, probeKernel_ = nullptr;
    ocl::id seed_ = nullptr, table_ = nullptr, dfa_ = nullptr;
    ocl::id baseSk_ = nullptr, basePub_ = nullptr;
    ocl::id outStart_ = nullptr, outLen_ = nullptr, outIds_ = nullptr;
    ocl::id meta_ = nullptr, records_ = nullptr;
    std::array<unsigned char, 32> seedBytes_{};
    std::vector<uint32_t> activeWords_;
    std::vector<uint32_t> activeOutLen_;
    std::vector<uint32_t> activeOutIds_;
    uint32_t readPos_ = 0;

    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
        if (rng_ != "chacha12" && rng_ != "philox" && rng_ != "aes-ctr") {
            error_ = "resident RNG must be chacha12, aes-ctr, or philox";
            return false;
        }
        if constexpr (!isCuda) { if (!ocl::load(&error_)) return false; }
        if ((isCuda ? device_.cudaOrdinal < 0 : !device_.deviceId) || !dictionary_ || dictionary_->words.empty()) {
            error_ = "missing GPU device or dictionary";
            return false;
        }
        const auto initStarted = std::chrono::steady_clock::now();
        auto beginStage = [](const char* text) {
            std::cerr << "  resident init: " << text << "..." << std::flush;
            return std::chrono::steady_clock::now();
        };
        auto finishStage = [](std::chrono::steady_clock::time_point started) {
            std::cerr << " done (" << elapsedText(started) << ")\n";
        };
        auto failStage = [this](std::chrono::steady_clock::time_point started) {
            std::cerr << " failed (" << elapsedText(started) << "): " << error_ << "\n";
        };
        const int rngMode = rng_ == "philox" ? 2 : (rng_ == "aes-ctr" ? 3 : 1);
        const std::string opts = openclResidentBuildOptions(rng_, openclOptions_);
        auto stageStarted = beginStage(isCuda ? "loading native CUDA PTX (first run may populate the driver cache)" :
            openclOptions_.hostSeed ? "creating OpenCL context and building scan-only program" :
            "creating OpenCL context and building RNG + scan program");
        bool built;
        if constexpr (isCuda) built = program_.build(device_.cudaOrdinal, rngMode, &error_);
        else built = program_.build(device_.platformId, device_.deviceId, kGpuKernelSource, opts, &error_,
                                    openclOptions_.profiling, true);
        if (!built) {
            failStage(stageStarted);
            return false;
        }
        finishStage(stageStarted);
        stageStarted = beginStage("creating kernels (driver may finalize machine code)");
        if (!openclOptions_.hostSeed) seedKernel_ = program_.kernel("tron_vanity_resident_seed", &error_);
        probeKernel_ = program_.kernel("tron_vanity_resident_probe", &error_);
        if ((!openclOptions_.hostSeed && !seedKernel_) || !probeKernel_) { failStage(stageStarted); return false; }
        finishStage(stageStarted);
        if constexpr (!isCuda) {
            size_t maxGroup = 0, preferred = 0;
            unsigned long long privateBytes = 0;
            if (program_.kernelLimits(probeKernel_, maxGroup, preferred, privateBytes)) {
                std::cerr << "  OpenCL scan: max group " << maxGroup << ", preferred multiple " << preferred
                          << ", reported private bytes " << privateBytes << " (not a VGPR count)\n";
                if (groupSize_ > maxGroup) {
                    error_ = "scan kernel work-group limit is " + std::to_string(maxGroup) +
                             "; reduce --gpu-group-size";
                    return false;
                }
            }
        }
        if (!openclOptions_.hostSeed && !randBytes(seedBytes_.data(), seedBytes_.size())) {
            error_ = "OS CSPRNG seed generation failed";
            return false;
        }

        stageStarted = beginStage("building the 32-bit secp256k1 offset table");
        auto table = genResidentTable(context_);
        finishStage(stageStarted);

        stageStarted = beginStage("uploading the table and dictionary");
        activeWords_.assign((dictionary_->words.size() + 31) / 32, 0xffffffffU);
        activeOutLen_ = dictionary_->outLen;
        activeOutIds_ = dictionary_->outIds;
        if (!openclOptions_.hostSeed)
            seed_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, seedBytes_.size(), seedBytes_.data(), &error_);
        table_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, table.size(), table.data(), &error_);
        baseSk_ = program_.buffer(ocl::MEM_READ_WRITE, 32, nullptr, &error_);
        basePub_ = program_.buffer(ocl::MEM_READ_WRITE, 64, nullptr, &error_);
        dfa_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                               dictionary_->dfa.size() * sizeof(uint32_t),
                               const_cast<uint32_t*>(dictionary_->dfa.data()), &error_);
        outStart_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                    dictionary_->outStart.size() * sizeof(uint32_t),
                                    const_cast<uint32_t*>(dictionary_->outStart.data()), &error_);
        outLen_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  activeOutLen_.size() * sizeof(uint32_t),
                                  activeOutLen_.data(), &error_);
        outIds_ = program_.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR,
                                  activeOutIds_.size() * sizeof(uint32_t),
                                  activeOutIds_.data(), &error_);
        if ((!openclOptions_.hostSeed && !seed_) || !table_ || !baseSk_ || !basePub_ ||
            !dfa_ || !outStart_ || !outLen_ || !outIds_) {
            failStage(stageStarted);
            return false;
        }
        finishStage(stageStarted);

        stageStarted = beginStage("allocating the device result ring");
        ringSlots_ = (bufferMiB_ * 1024u * 1024u) / kRecordBytes;
        meta_ = program_.buffer(ocl::MEM_READ_WRITE, kMetaWords * sizeof(uint32_t), nullptr, &error_);
        records_ = program_.buffer(ocl::MEM_READ_WRITE,
                                   static_cast<size_t>(ringSlots_) * kRecordBytes, nullptr, &error_);
        std::array<uint32_t, kMetaWords> zero{};
        if (!meta_ || !records_) {
            failStage(stageStarted);
            return false;
        }
        if (!program_.write(meta_, sizeof(zero), zero.data())) {
            error_ = "initializing resident metadata failed";
            failStage(stageStarted);
            return false;
        }
        finishStage(stageStarted);

        stageStarted = beginStage(openclOptions_.hostSeed ? "validating OS CSPRNG and CPU base expansion" :
                                  "validating GPU CSPRNG and CPU base expansion");
        if (!prepareBasePair()) {
            failStage(stageStarted);
            return false;
        }
        finishStage(stageStarted);
        ready_ = true;
        std::cerr << "  resident init: ready (" << elapsedText(initStarted) << " total)\n";
        return true;
    }

    bool bindSeedKernel() {
        uint64_t counter = rngCounter_;
        return program_.setArg(seedKernel_, 0, sizeof(ocl::id), &seed_) &&
               program_.setArg(seedKernel_, 1, sizeof(ocl::id), &baseSk_) &&
               program_.setArg(seedKernel_, 2, sizeof(uint64_t), &counter);
    }

    bool bindProbeKernel() {
        uint64_t sequence = sequenceBase_;
        return program_.setArg(probeKernel_, 0, sizeof(ocl::id), &baseSk_) &&
               program_.setArg(probeKernel_, 1, sizeof(ocl::id), &basePub_) &&
               program_.setArg(probeKernel_, 2, sizeof(ocl::id), &table_) &&
               program_.setArg(probeKernel_, 3, sizeof(ocl::id), &dfa_) &&
               program_.setArg(probeKernel_, 4, sizeof(ocl::id), &outStart_) &&
               program_.setArg(probeKernel_, 5, sizeof(ocl::id), &outLen_) &&
               program_.setArg(probeKernel_, 6, sizeof(ocl::id), &outIds_) &&
               program_.setArg(probeKernel_, 7, sizeof(ocl::id), &meta_) &&
               program_.setArg(probeKernel_, 8, sizeof(ocl::id), &records_) &&
               program_.setArg(probeKernel_, 9, sizeof(uint32_t), &ringSlots_) &&
               program_.setArg(probeKernel_, 10, sizeof(uint64_t), &sequence);
    }

    bool prepareBasePair() {
        PrivateScalar sk;
        if (openclOptions_.hostSeed) {
            bool valid = false;
            for (unsigned attempt = 0; attempt < 4 && !valid; ++attempt) {
                if (!randBytes(sk.data(), sk.size())) { error_ = "OS CSPRNG failed"; return false; }
                valid = secp256k1_ec_seckey_verify(context_, sk.data()) != 0;
            }
            if (!valid) { error_ = "OS CSPRNG rejection sampling failed"; return false; }
            if (!program_.write(baseSk_, sk.size(), sk.data())) {
                error_ = "uploading OS-generated scalar failed"; return false;
            }
        } else {
            if (!bindSeedKernel()) { error_ = "setting resident seed-kernel arguments failed"; return false; }
            if (!program_.run1D(seedKernel_, 1, 1, &error_) || !program_.finish()) {
                if (error_.empty()) error_ = "resident seed-kernel execution failed";
                return false;
            }
            if (!program_.read(baseSk_, sk.size(), sk.data())) {
                error_ = "reading resident GPU-generated scalar failed";
                return false;
            }
        }
        secp256k1_pubkey pubkey;
        unsigned char encoded[65] = {0};
        size_t encodedLen = sizeof(encoded);
        const bool valid = secp256k1_ec_seckey_verify(context_, sk.data()) &&
                           secp256k1_ec_pubkey_create(context_, &pubkey, sk.data()) &&
                           secp256k1_ec_pubkey_serialize(context_, encoded, &encodedLen, &pubkey,
                                                        SECP256K1_EC_UNCOMPRESSED) &&
                           encodedLen == sizeof(encoded);
        if (!valid) {
            error_ = "CSPRNG returned an invalid secp256k1 scalar";
            return false;
        }
        if (!program_.write(basePub_, 64, encoded + 1)) {
            error_ = "uploading the CPU-expanded resident base point failed";
            return false;
        }
        ++rngCounter_;
        return true;
    }

    bool decodeAndVerifyRecord(const unsigned char* rec, FoundKey& key,
                               bool& outputsChanged) {
        secp256k1_pubkey pubkey;
        unsigned char encoded[65] = {0};
        size_t encodedLen = sizeof(encoded);
        if (!secp256k1_ec_seckey_verify(context_, rec) ||
            !secp256k1_ec_pubkey_create(context_, &pubkey, rec) ||
            !secp256k1_ec_pubkey_serialize(context_, encoded, &encodedLen, &pubkey,
                                           SECP256K1_EC_UNCOMPRESSED) ||
            encodedLen != sizeof(encoded)) {
            error_ = "GPU resident returned an invalid private key";
            return false;
        }
        const std::string address = tronAddressFromPubXY(encoded + 1);
        const std::string recorded(reinterpret_cast<const char*>(rec + 32), 34);
        if (address != recorded) {
            error_ = "GPU resident key/address validation mismatch";
            return false;
        }
        auto ids = dictionary_->matchIds(address);
        if (ids.empty()) {
            error_ = "GPU resident returned a false dictionary match";
            return false;
        }
        for (uint32_t id : ids) {
            if (id >= dictionary_->words.size()) continue;
            const uint32_t mask = 1U << (id & 31);
            if (!pruneOutputs_ || (activeWords_[id >> 5] & mask)) {
                if (pruneOutputs_) {
                    activeWords_[id >> 5] &= ~mask;
                    outputsChanged = true;
                }
                key.words.push_back(dictionary_->words[id]);
            }
        }
        if (key.words.empty()) return true;
        key.address = address;
        key.privHex = hexUpper(rec, 32);
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
        if (!program_.write(outLen_, activeOutLen_.size() * sizeof(uint32_t), activeOutLen_.data()) ||
            !program_.write(outIds_, activeOutIds_.size() * sizeof(uint32_t), activeOutIds_.data())) {
            error_ = "updating active resident dictionary outputs failed";
            return false;
        }
        return true;
    }

    bool runChunk(uint32_t* produced, uint32_t* overflow, const ReportFn& report) {
        const auto baseStart = std::chrono::steady_clock::now();
        lastChunkKeys_ = static_cast<uint64_t>(workItems_) * kResidentKpi;
        if (!prepareBasePair() || !bindProbeKernel()) {
            if (error_.empty()) error_ = "setting resident scan-kernel arguments failed";
            return false;
        }
        const auto scanStart = std::chrono::steady_clock::now();
        if (!program_.run1D(probeKernel_, workItems_, groupSize_, &error_) ||
            !program_.finish()) {
            if (error_.empty()) error_ = "resident scan-kernel execution failed";
            return false;
        }
        const auto scanEnd = std::chrono::steady_clock::now();
        const double scanMs = std::chrono::duration<double, std::milli>(scanEnd - scanStart).count();
        if constexpr (!isCuda) {
            if (openclOptions_.profiling) {
                double gpuMs = 0;
                if (program_.lastKernelMilliseconds(gpuMs)) {
                    profile_.gpuSeconds += gpuMs / 1000.0;
                    profile_.maxGpuMs = std::max(profile_.maxGpuMs, gpuMs);
                } else profile_.gpuTimingValid = false;
                profile_.baseSeconds += std::chrono::duration<double>(scanStart - baseStart).count();
                profile_.scanSeconds += scanMs / 1000.0;
            }
        }
        std::array<uint32_t, kMetaWords> meta{};
        if (!program_.read(meta_, sizeof(meta), meta.data())) { error_ = "reading resident metadata failed"; return false; }
        const uint32_t writePos = meta[0];
        const uint32_t available = writePos - readPos_;
        const uint32_t count = std::min(available, ringSlots_);
        if (meta[2] || meta[3] || available > ringSlots_) {
            error_ = "GPU result ring overflow/error; stopping to avoid lost matches";
            return false;
        }
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
            bool outputsChanged = false;
            for (uint32_t i = 0; i < count; ++i) {
                const unsigned char* rec = bytes.data() + static_cast<size_t>(i) * kRecordBytes;
                FoundKey key;
                if (!decodeAndVerifyRecord(rec, key, outputsChanged)) return false;
                if (!key.words.empty()) report(key);
            }
            if (outputsChanged && !refreshActiveOutputs()) return false;
        }
        readPos_ = writePos;
        if (!program_.writeAt(meta_, sizeof(uint32_t), sizeof(uint32_t), &readPos_)) return false;
        sequenceBase_ += lastChunkKeys_;
        profile_.keys += lastChunkKeys_;
        ++profile_.dispatches;
        if (adaptive_ && scanMs > 0) {
            // At most 2x growth, immediate reduction after a slow dispatch.
            // Target is best-effort, not a WDDM hard execution-time guarantee.
            const double ratio = std::min(2.0, chunkMs_ / scanMs);
            const uint32_t next = static_cast<uint32_t>(std::clamp(workItems_ * ratio,
                static_cast<double>(groupSize_), static_cast<double>(1u << 20)));
            workItems_ = std::max(groupSize_, next / groupSize_ * groupSize_);
        }
        return true;
    }
};

} // namespace

std::unique_ptr<Backend> makeResidentGpuBackend(
    const GpuDevice& device, std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng, uint32_t bufferMiB, uint32_t chunkMs, uint32_t pollMs, uint32_t groupSize,
    OpenclResidentOptions options) {
    return std::make_unique<ResidentGpuBackend<ocl::Program>>(device, std::move(dictionary), rng,
                                                bufferMiB, chunkMs, pollMs, groupSize, options);
}

OpenclProfileResult profileOpenclResident(const GpuDevice& device,
    std::shared_ptr<const Dictionary> dictionary, const std::string& rng,
    uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize,
    OpenclResidentOptions options, double seconds) {
    options.profiling = true;
    ResidentGpuBackend<ocl::Program> backend(device, std::move(dictionary), rng,
                                            bufferMiB, chunkMs, 0, groupSize, options);
    return backend.profile(seconds);
}

int openclResidentSelfTest(const GpuDevice& device, const std::string& rng, OpenclResidentOptions options) {
    auto dictionary = std::make_shared<Dictionary>();
    dictionary->words = {"test"};
    dictionary->dfa.assign(Dictionary::Alphabet, 0);
    dictionary->outStart = {0};
    dictionary->outLen = {1};
    dictionary->outIds = {0};
    ResidentGpuBackend<ocl::Program> backend(device, dictionary, rng, 8, 8, 0, 64, options);
    if (!backend.selfTest()) {
        std::cerr << "OpenCL resident self-test failed: " << backend.note() << "\n";
        return 1;
    }
    std::cout << "OpenCL " << (options.hostSeed ? "OS CSPRNG" : rng)
              << ": 1024 address/key pairs verified; no wallet output\n";
    return 0;
}

int diagnoseOpencl(const GpuDevice& device, const std::string& stage,
                   const std::string& rng, OpenclResidentOptions options) {
    if (rng != "chacha12" && rng != "aes-ctr" && rng != "philox") {
        std::cerr << "resident RNG must be chacha12, aes-ctr, or philox\n"; return 1;
    }
    std::string error;
    if (!ocl::load(&error)) { std::cerr << error << std::endl; return 1; }
    std::cout << "OpenCL diagnostic " << stage << ": " << ocl::deviceDescription(device.deviceId)
              << "\nNo wallets or private keys are printed or saved." << std::endl;
    if (stage == "scan" || stage == "full") {
        options.hostSeed = stage == "scan";
        std::cout << (options.hostSeed ? "SCAN ONLY: OS CSPRNG, no RNG kernel in the program\n" :
                                       "COMBINED: GPU RNG and scan in one program\n") << std::flush;
        return openclResidentSelfTest(device, rng, options);
    }
    if (stage != "smoke" && stage != "rng") { std::cerr << "unknown OpenCL diagnostic stage\n"; return 1; }
    ocl::Program program;
    const std::string smoke = "__kernel void diagnostic_smoke(__global uint* out) { out[get_global_id(0)] = 0x13579bdfU ^ (uint)get_global_id(0); }";
    options.hostSeed = false;
    const std::string flags = stage == "smoke" ? "-cl-std=CL1.2" :
        openclResidentBuildOptions(rng, options) + " -D RESIDENT_SEED_ONLY=1";
    if (!program.build(device.platformId, device.deviceId,
                       stage == "smoke" ? smoke : kGpuKernelSource, flags, &error, false, true)) {
        std::cerr << error << std::endl; return 1;
    }
    auto kernel = program.kernel(stage == "smoke" ? "diagnostic_smoke" : "tron_vanity_resident_seed", &error);
    if (!kernel) { std::cerr << error << std::endl; return 1; }
    auto output = program.buffer(ocl::MEM_READ_WRITE, 32, nullptr, &error);
    if (!output) { std::cerr << error << std::endl; return 1; }
    if (stage == "smoke") {
        std::array<uint32_t, 8> values{};
        std::cout << "OpenCL smoke dispatch BEGIN" << std::endl;
        if (!program.setArg(kernel, 0, sizeof(output), &output) ||
            !program.run1D(kernel, values.size(), 1, &error) || !program.finish() ||
            !program.read(output, sizeof(values), values.data())) {
            std::cerr << "OpenCL smoke execution failed: " << error << std::endl; return 1;
        }
        for (uint32_t i = 0; i < values.size(); ++i)
            if (values[i] != (0x13579bdfU ^ i)) { std::cerr << "OpenCL smoke mismatch\n"; return 1; }
    } else {
        // Public deterministic fixture, never used for a real search.
        std::array<unsigned char, 32> seed{};
        for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<unsigned char>(i);
        auto input = program.buffer(ocl::MEM_READ_ONLY | ocl::MEM_COPY_HOST_PTR, seed.size(), seed.data(), &error);
        if (!input || !program.setArg(kernel, 0, sizeof(input), &input) ||
            !program.setArg(kernel, 1, sizeof(output), &output)) {
            std::cerr << "RNG diagnostic argument failure: " << error << std::endl; return 1;
        }
        std::array<std::array<unsigned char, 32>, 3> samples{};
        for (unsigned i = 0; i < samples.size(); ++i) {
            const uint64_t counter = i == 2 ? 43 : 42;
            std::cout << "OpenCL isolated RNG dispatch " << i + 1 << " BEGIN" << std::endl;
            if (!program.setArg(kernel, 2, sizeof(counter), &counter) ||
                !program.run1D(kernel, 1, 1, &error) || !program.finish() ||
                !program.read(output, samples[i].size(), samples[i].data())) {
                std::cerr << "RNG diagnostic execution failed: " << error << std::endl; return 1;
            }
            if (!secp256k1_ec_seckey_verify(secp256k1_context_static, samples[i].data())) {
                std::cerr << "RNG diagnostic scalar validation failed\n"; return 1;
            }
        }
        if (samples[0] != samples[1] || samples[0] == samples[2]) {
            std::cerr << "RNG diagnostic counter/reproducibility mismatch\n"; return 1;
        }
    }
    std::cout << "OpenCL diagnostic " << stage << " PASS" << std::endl;
    return 0;
}

std::unique_ptr<Backend> makeCudaBackend(
    const GpuDevice& device, std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng, uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize) {
    return std::make_unique<ResidentGpuBackend<cuda::Program>>(device, std::move(dictionary), rng,
                                                               bufferMiB, chunkMs, 0, groupSize);
}

int cudaSelfTest(const GpuDevice& device) {
    // A synthetic DFA accepts every address, forcing every result through the
    // independent CPU secp256k1/Keccak/Base58 verifier. Nothing is saved.
    auto dictionary = std::make_shared<Dictionary>();
    dictionary->words = {"test"};
    dictionary->dfa.assign(Dictionary::Alphabet, 0);
    dictionary->outStart = {0};
    dictionary->outLen = {1};
    dictionary->outIds = {0};
    for (const auto& rng : {"chacha12", "aes-ctr", "philox"}) {
        ResidentGpuBackend<cuda::Program> backend(device, dictionary, rng, 8, 8, 0, 64);
        if (!backend.selfTest()) {
            std::cerr << "CUDA self-test " << rng << " failed: " << backend.note() << "\n";
            return 1;
        }
        std::cout << "CUDA " << rng << ": 1024 address/key pairs verified; no wallet output\n";
    }
    return 0;
}
