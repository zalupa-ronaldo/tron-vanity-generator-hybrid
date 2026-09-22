#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"

#include "crypto.h"
#include "rng.h"
#include "resident_protocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <secp256k1.h>

#include "metal_kernel_src.h"

namespace {

constexpr uint32_t kRecordBytes = resident_protocol::kRecordBytes;
constexpr uint32_t kMaxMatches = resident_protocol::kMaxMatches;
constexpr uint32_t kMetaWords = resident_protocol::kMetaWords;
constexpr uint32_t kResidentEcw = 8;
constexpr uint32_t kResidentEcbits = 256;

uint32_t read32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
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

std::vector<unsigned char> genTable(secp256k1_context* c) {
    const uint32_t digits = 1u << kResidentEcw;
    const uint32_t windows = (kResidentEcbits + kResidentEcw - 1) / kResidentEcw;
    std::vector<unsigned char> table(static_cast<size_t>(windows) * digits * 64, 0);
    for (uint32_t w = 0; w < windows; ++w) {
        for (uint32_t d = 1; d < digits; ++d) {
            unsigned char sk[32] = {};
            for (uint32_t bit = 0; bit < kResidentEcw; ++bit) {
                uint32_t absolute = w * kResidentEcw + bit;
                if (absolute < kResidentEcbits && ((d >> bit) & 1u))
                    sk[31 - absolute / 8] |= static_cast<unsigned char>(1u << (absolute & 7));
            }
            secp256k1_pubkey pub;
            if (!secp256k1_ec_pubkey_create(c, &pub, sk)) continue;
            unsigned char encoded[65];
            size_t len = sizeof(encoded);
            secp256k1_ec_pubkey_serialize(c, encoded, &len, &pub, SECP256K1_EC_UNCOMPRESSED);
            std::memcpy(table.data() + (static_cast<size_t>(w) * digits + d) * 64, encoded + 1, 64);
        }
    }
    return table;
}

class MetalResidentBackend final : public Backend {
public:
    MetalResidentBackend(std::shared_ptr<const Dictionary> dictionary,
                         std::string rng, uint32_t bufferMiB,
                         uint32_t chunkMs, uint32_t groupSize,
                         uint32_t keysPerLane, uint32_t profileStage = 0,
                         bool scalarKeccak = false,
                         uint32_t pipelineMaxThreads = 0)
        : dictionary_(std::move(dictionary)), rng_(std::move(rng)),
          bufferMiB_(std::max(8u, bufferMiB)),
          // Keep Metal chunks bounded too: oversized dispatches make the
          // result ring and interactive polling sluggish without improving
          // steady-state throughput on Apple Silicon.
          chunkMs_(std::clamp(chunkMs, 8u, 100u)),
          groupSize_(groupSize == 64 || groupSize == 128 || groupSize == 256 ? groupSize : 256),
          keysPerLane_(keysPerLane), profileStage_(profileStage), scalarKeccak_(scalarKeccak),
          pipelineMaxThreads_(pipelineMaxThreads) {
        context_ = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        // Keep roughly the same number of keys per dispatch when the point-walk
        // batch changes. Without this scaling, KPI=1024 turns a bounded chunk
        // into a multi-second command buffer even though throughput improves.
        const uint64_t desired = (uint64_t{1} << 18) * chunkMs_ / keysPerLane_;
        workItems_ = static_cast<uint32_t>(std::clamp<uint64_t>(desired, 1u << 14, 1u << 20));
        workItems_ = (workItems_ + groupSize_ - 1) / groupSize_ * groupSize_;
    }

    ~MetalResidentBackend() override {
        if (context_) secp256k1_context_destroy(context_);
    }

    std::string name() const override { return "Metal GPU resident"; }

    bool available() const override { return const_cast<MetalResidentBackend*>(this)->ensureReady(); }
    std::string note() const override { return error_; }

    BackendInfo info() const override {
        BackendInfo out;
        out.kind = "GPU-resident";
        out.title = device_ ? std::string([[device_ name] UTF8String]) : "Apple GPU";
        out.lines.push_back("Metal runtime available");
        out.lines.push_back("CSPRNG " + rng_ + ", chunk " + std::to_string(chunkMs_) +
                            " ms, " + std::to_string(keysPerLane_) + " keys/lane");
        out.lines.push_back(scalarKeccak_ ? "Keccak named-lane permutation"
                                         : "Keccak indexed reference permutation");
        out.lines.push_back(std::to_string(bufferMiB_) + " MiB device result ring");
        return out;
    }

    double benchmark(double seconds) override {
        if (!ensureReady()) return 0.0;
        auto start = std::chrono::steady_clock::now();
        uint64_t generated = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
            if (!runChunk(nullptr, nullptr, nullptr)) break;
            generated += static_cast<uint64_t>(workItems_) * keysPerLane_;
        }
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return elapsed > 0 ? generated / elapsed : 0.0;
    }

    MetalProfileResult profile(double seconds) {
        MetalProfileResult result;
        if (!ensureReady()) { result.error = error_; return result; }
        // Exclude shader compilation and first-dispatch warmup from timings.
        if (!runChunk(nullptr, nullptr, nullptr)) { result.error = error_; return result; }
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
            double gpuSeconds = 0.0;
            if (!runChunk(nullptr, nullptr, nullptr, &gpuSeconds)) {
                result.error = error_;
                return result;
            }
            result.keys += static_cast<uint64_t>(workItems_) * keysPerLane_;
            result.gpuSeconds += gpuSeconds;
            ++result.dispatches;
        }
        result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    }

    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        if (!ensureReady()) {
            std::cerr << "Metal resident unavailable: " << error_ << "\n";
            return;
        }
        while (!state.stop.load(std::memory_order_relaxed)) {
            if (cfg.maxAttempts && state.checked.load() >= cfg.maxAttempts) break;
            uint32_t overflow = 0;
            ReportFn counted = [&](const FoundKey& key) {
                state.found.fetch_add(1, std::memory_order_relaxed);
                report(key);
            };
            if (!runChunk(nullptr, &overflow, counted)) {
                std::cerr << "Metal resident chunk failed: " << error_ << "\n";
                state.stop.store(true);
                return;
            }
            state.checked.fetch_add(static_cast<uint64_t>(workItems_) * keysPerLane_, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(static_cast<uint64_t>(workItems_) * keysPerLane_, std::memory_order_relaxed);
            if (overflow) {
                std::cerr << "Metal resident result ring overflow; stopping\n";
                state.stop.store(true);
                return;
            }
        }
    }

private:
    std::shared_ptr<const Dictionary> dictionary_;
    std::string rng_;
    uint32_t bufferMiB_, chunkMs_;
    uint32_t workItems_ = 1u << 18;
    uint32_t ringSlots_ = 0;
    uint32_t groupSize_ = 256;
    uint32_t keysPerLane_ = 32;
    uint32_t profileStage_ = 0;
    bool scalarKeccak_ = false;
    uint32_t pipelineMaxThreads_ = 0;
    uint64_t streamBase_ = 0;
    uint32_t readPos_ = 0;
    secp256k1_context* context_ = nullptr;
    bool tried_ = false, ready_ = false;
    std::string error_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLBuffer> seed_ = nil, table_ = nil, dfa_ = nil, outStart_ = nil, outLen_ = nil, outIds_ = nil;
    id<MTLBuffer> meta_ = nil, records_ = nil;
    std::vector<uint32_t> activeWords_;
    std::array<unsigned char, 32> seedBytes_{};
    uint32_t threadWidth_ = 32;

    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
        if (rng_ != "chacha12" && rng_ != "aes-ctr" && rng_ != "philox") {
            error_ = "Metal resident RNG must be chacha12, aes-ctr, or philox";
            return false;
        }
        if (!dictionary_ || dictionary_->words.empty()) {
            error_ = "missing dictionary";
            return false;
        }
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) { error_ = "MTLCreateSystemDefaultDevice returned nil"; return false; }
        queue_ = [device_ newCommandQueue];
        if (!queue_) { error_ = "cannot create Metal command queue"; return false; }

        int mode = rng_ == "philox" ? 2 : (rng_ == "aes-ctr" ? 3 : 1);
        NSString* prefix = [NSString stringWithFormat:@"#define RESIDENT 1\n#define RESIDENT_RNG %d\n#define ECW %u\n#define ECBITS %u\n#define KPI %u\n#define RESIDENT_PROFILE_STAGE %u\n#define RESIDENT_SCALAR_KECCAK %u\n#define MONT_N 1\n#define METAL_BACKEND 1\n", mode, kResidentEcw, kResidentEcbits, keysPerLane_, profileStage_, scalarKeccak_ ? 1u : 0u];
        NSString* source = [prefix stringByAppendingString:[NSString stringWithUTF8String:kMetalKernelSource]];
        NSError* nsError = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&nsError];
        if (!library) {
            error_ = nsError ? std::string([[nsError localizedDescription] UTF8String]) : "Metal shader compilation failed";
            return false;
        }
        if (scalarKeccak_) {
            id<MTLFunction> validationFunction = [library newFunctionWithName:@"test_keccak_variants"];
            id<MTLComputePipelineState> validationPipeline = validationFunction
                ? [device_ newComputePipelineStateWithFunction:validationFunction error:&nsError] : nil;
            constexpr NSUInteger kValidationCount = 1024;
            id<MTLBuffer> mismatch = [device_ newBufferWithLength:kValidationCount * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
            if (!validationPipeline || !mismatch) {
                error_ = nsError ? std::string([[nsError localizedDescription] UTF8String])
                                 : "Metal scalar Keccak validation setup failed";
                return false;
            }
            std::memset([mismatch contents], 0, kValidationCount * sizeof(uint32_t));
            id<MTLCommandBuffer> validationCommand = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> validationEncoder = [validationCommand computeCommandEncoder];
            [validationEncoder setComputePipelineState:validationPipeline];
            [validationEncoder setBuffer:mismatch offset:0 atIndex:0];
            const NSUInteger width = std::max<NSUInteger>(1, [validationPipeline threadExecutionWidth]);
            NSUInteger groupWidth = std::min<NSUInteger>(256, [validationPipeline maxTotalThreadsPerThreadgroup]);
            groupWidth -= groupWidth % width;
            if (!groupWidth) groupWidth = width;
            [validationEncoder dispatchThreads:MTLSizeMake(kValidationCount, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(groupWidth, 1, 1)];
            [validationEncoder endEncoding];
            [validationCommand commit];
            [validationCommand waitUntilCompleted];
            if ([validationCommand status] != MTLCommandBufferStatusCompleted) {
                error_ = validationCommand.error
                    ? std::string([[validationCommand.error localizedDescription] UTF8String])
                    : "Metal scalar Keccak validation command failed";
                return false;
            }
            auto* differences = static_cast<const uint32_t*>([mismatch contents]);
            for (NSUInteger i = 0; i < kValidationCount; ++i) {
                if (differences[i]) {
                    error_ = "Metal scalar Keccak disagrees with reference permutation";
                    return false;
                }
            }
        }
        id<MTLFunction> function = [library newFunctionWithName:@"tron_vanity_resident"];
        if (!function) { error_ = "tron_vanity_resident not found in Metal library"; return false; }
        if (pipelineMaxThreads_) {
            MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            descriptor.maxTotalThreadsPerThreadgroup = pipelineMaxThreads_;
            pipeline_ = [device_ newComputePipelineStateWithDescriptor:descriptor
                                                                 options:MTLPipelineOptionNone
                                                              reflection:nil error:&nsError];
        } else {
            pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&nsError];
        }
        if (!pipeline_) {
            error_ = nsError ? std::string([[nsError localizedDescription] UTF8String]) : "Metal pipeline creation failed";
            return false;
        }
        threadWidth_ = static_cast<uint32_t>(std::max<NSUInteger>(1, [pipeline_ threadExecutionWidth]));
        if (!randBytes(seedBytes_.data(), seedBytes_.size())) { error_ = "OS CSPRNG seed failed"; return false; }
        auto table = genTable(context_);
        auto make = [&](const void* p, size_t n) -> id<MTLBuffer> {
            return [device_ newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
        };
        seed_ = make(seedBytes_.data(), seedBytes_.size());
        table_ = make(table.data(), table.size());
        dfa_ = make(dictionary_->dfa.data(), dictionary_->dfa.size() * sizeof(uint32_t));
        outStart_ = make(dictionary_->outStart.data(), dictionary_->outStart.size() * sizeof(uint32_t));
        outLen_ = make(dictionary_->outLen.data(), dictionary_->outLen.size() * sizeof(uint32_t));
        outIds_ = make(dictionary_->outIds.data(), dictionary_->outIds.size() * sizeof(uint32_t));
        activeWords_.assign((dictionary_->words.size() + 31) / 32, 0xffffffffU);
        ringSlots_ = (bufferMiB_ * 1024u * 1024u) / kRecordBytes;
        meta_ = [device_ newBufferWithLength:kMetaWords * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        records_ = [device_ newBufferWithLength:static_cast<size_t>(ringSlots_) * kRecordBytes options:MTLResourceStorageModeShared];
        if (!seed_ || !table_ || !dfa_ || !outStart_ || !outLen_ || !outIds_ ||
            !meta_ || !records_) {
            error_ = "Metal buffer allocation failed"; return false;
        }
        std::memset([meta_ contents], 0, kMetaWords * sizeof(uint32_t));
        ready_ = true;
        return true;
    }

    bool runChunk(uint32_t* produced, uint32_t* overflow, const ReportFn& report,
                  double* gpuSeconds = nullptr) {
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) { error_ = "Metal command encoder creation failed"; return false; }
        [encoder setComputePipelineState:pipeline_];
        id<MTLBuffer> buffers[] = {seed_, table_, dfa_, outStart_, outLen_, outIds_, meta_, records_};
        for (NSUInteger i = 0; i < 8; ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        uint32_t cap = ringSlots_;
        uint64_t stream = streamBase_;
        [encoder setBytes:&cap length:sizeof(cap) atIndex:8];
        [encoder setBytes:&stream length:sizeof(stream) atIndex:9];
        MTLSize threads = MTLSizeMake(workItems_, 1, 1);
        NSUInteger width = std::max<NSUInteger>(1, [pipeline_ threadExecutionWidth]);
        NSUInteger requestedGroup = std::min<NSUInteger>(groupSize_, 256);
        if (requestedGroup < width) requestedGroup = width;
        if (requestedGroup % width) requestedGroup = width;
        MTLSize group = MTLSizeMake(requestedGroup, 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted) {
            error_ = command.error ? std::string([[command.error localizedDescription] UTF8String]) : "Metal command failed";
            return false;
        }
        if (gpuSeconds) {
            const double start = [command GPUStartTime];
            const double end = [command GPUEndTime];
            *gpuSeconds = end > start ? end - start : 0.0;
        }
        auto* meta = static_cast<uint32_t*>([meta_ contents]);
        uint32_t writePos = meta[0];
        uint32_t available = writePos - readPos_;
        uint32_t count = std::min(available, ringSlots_);
        if (produced) *produced = count;
        if (overflow) *overflow = meta[2];
        if (report && count) {
            auto* raw = static_cast<unsigned char*>([records_ contents]);
            bool outputsChanged = false;
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t slot = (readPos_ + i) % ringSlots_;
                const unsigned char* rec = raw + static_cast<size_t>(slot) * kRecordBytes;
                FoundKey key;
                uint32_t n = std::min(read32(rec + 68), kMaxMatches);
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t id = read32(rec + 72 + j * 4);
                    uint32_t mask = 1U << (id & 31);
                    if (id < dictionary_->words.size() && (activeWords_[id >> 5] & mask)) {
                        activeWords_[id >> 5] &= ~mask;
                        outputsChanged = true;
                        key.words.push_back(dictionary_->words[id]);
                    }
                }
                if (!key.words.empty()) {
                    key.privHex = hexUpper(rec, 32);
                    key.address.assign(reinterpret_cast<const char*>(rec + 32), 34);
                    report(key);
                }
            }
            if (outputsChanged) refreshActiveOutputs();
        }
        readPos_ = writePos;
        meta[1] = readPos_;
        streamBase_ += workItems_ / groupSize_;
        return true;
    }

    void refreshActiveOutputs() {
        auto* lengths = static_cast<uint32_t*>([outLen_ contents]);
        auto* ids = static_cast<uint32_t*>([outIds_ contents]);
        for (size_t state = 0; state < dictionary_->outLen.size(); ++state) {
            const uint32_t start = dictionary_->outStart[state];
            const uint32_t originalLength = dictionary_->outLen[state];
            uint32_t activeLength = 0;
            for (uint32_t j = 0; j < originalLength; ++j) {
                const uint32_t id = dictionary_->outIds[start + j];
                const uint32_t mask = 1U << (id & 31);
                if (activeWords_[id >> 5] & mask) ids[start + activeLength++] = id;
            }
            lengths[state] = activeLength;
        }
    }
};

} // namespace

bool metalAvailable(std::string* note) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { if (note) *note = "no Metal device"; return false; }
    if (note) *note = std::string([[device name] UTF8String]);
    return true;
}

std::string metalDeviceSummary() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device ? std::string([[device name] UTF8String]) : "Metal unavailable";
}

MetalProfileResult profileMetalResidentStage(
    std::shared_ptr<const Dictionary> dictionary, const std::string& rng,
    uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize,
    uint32_t keysPerLane, uint32_t stage, double seconds, bool scalarKeccak,
    uint32_t pipelineMaxThreads) {
    if (stage > 5 || seconds <= 0.0)
        return MetalProfileResult{0, 0.0, 0.0, 0, "invalid Metal profile stage or duration"};
    MetalResidentBackend backend(std::move(dictionary), rng, bufferMiB, chunkMs,
                                 groupSize, keysPerLane, stage, scalarKeccak, pipelineMaxThreads);
    return backend.profile(seconds);
}

std::unique_ptr<Backend> makeMetalResidentBackend(
    std::shared_ptr<const Dictionary> dictionary, const std::string& rng,
    uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize,
    uint32_t keysPerLane, bool scalarKeccak) {
    return std::make_unique<MetalResidentBackend>(std::move(dictionary), rng, bufferMiB, chunkMs,
                                                  groupSize, keysPerLane, 0, scalarKeccak);
}
