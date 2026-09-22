#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_hardware_profile.h"

#include "metal_hardware_profile_src.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct HwParams {
    uint32_t iterations;
    uint32_t mask;
    uint32_t stride;
    uint32_t seed;
};

enum class MetricKind { Updates, FieldMul, Bytes, Barriers, Dispatch };

struct CaseSpec {
    std::string name;
    std::string category;
    std::string function;
    MetricKind metric = MetricKind::Updates;
    uint32_t grid = 1u << 18;
    uint32_t group = 256;
    uint32_t iterations = 256;
    double operationsPerIteration = 1.0;
    uint64_t bytesPerThread = 0;
    uint32_t threadgroupBytes = 0;
    uint32_t stride = 1;
    uint32_t pipelineHint = 0;
    bool calibrate = true;
    std::string note;
};

struct CaseResult {
    CaseSpec spec;
    uint32_t actualGroup = 0;
    uint32_t iterations = 0;
    uint32_t dispatches = 0;
    uint32_t executionWidth = 0;
    uint32_t maxThreads = 0;
    uint32_t staticThreadgroupBytes = 0;
    double gpuSeconds = 0.0;
    double wallSeconds = 0.0;
    double value = 0.0;
    double gpuMicrosPerDispatch = 0.0;
    std::string unit;
    std::vector<uint32_t> sampleWords;
};

struct DispatchTime {
    double gpu = 0.0;
    double wall = 0.0;
    std::string error;
};

std::string jsonEscape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
                else out << static_cast<char>(c);
        }
    }
    return out.str();
}

bool selected(const CaseSpec& c, const std::string& filter) {
    if (filter.empty() || filter == "all") return true;
    static const char* categories[] = {"dispatch", "arithmetic", "simd", "field", "private",
                                       "memory", "group", "hint", "threadgroup", "banks"};
    for (const char* category : categories)
        if (filter == category) return c.category == filter;
    return c.name.find(filter) != std::string::npos;
}

std::vector<CaseSpec> buildCases() {
    std::vector<CaseSpec> v;
    auto add = [&](CaseSpec c) { v.push_back(std::move(c)); };

    add({"dispatch", "dispatch", "hw_dispatch", MetricKind::Dispatch,
         1, 32, 1, 1.0, 0, 0, 1, 0, false,
         "one-thread command-buffer round trip"});

    add({"u32-mix-dependent", "arithmetic", "hw_u32_mix_dep", MetricKind::Updates,
         1u << 18, 256, 256, 1.0, 0, 0, 1, 0, true, "dependent 32-bit add/rotate/xor"});
    add({"u32-mix-ilp4", "arithmetic", "hw_u32_mix_ilp4", MetricKind::Updates,
         1u << 18, 256, 256, 4.0, 0, 0, 1, 0, true, "four independent 32-bit chains"});
    add({"u32-mul-dependent", "arithmetic", "hw_u32_mul_dep", MetricKind::Updates,
         1u << 18, 256, 256, 1.0, 0, 0, 1, 0, true, "dependent 32-bit multiply-add"});
    add({"u32-mul-ilp4", "arithmetic", "hw_u32_mul_ilp4", MetricKind::Updates,
         1u << 18, 256, 256, 4.0, 0, 0, 1, 0, true, "four independent 32-bit multiply-add chains"});
    add({"u64-mul-dependent", "arithmetic", "hw_u64_mul_dep", MetricKind::Updates,
         1u << 18, 256, 64, 1.0, 0, 0, 1, 0, true, "dependent native ulong multiply-add"});
    add({"u64-mul-ilp4", "arithmetic", "hw_u64_mul_ilp4", MetricKind::Updates,
         1u << 18, 256, 64, 4.0, 0, 0, 1, 0, true, "four independent ulong multiply-add chains"});
    add({"carry-chain-8", "arithmetic", "hw_carry8", MetricKind::Updates,
         1u << 18, 256, 128, 8.0, 0, 0, 1, 0, true, "eight dependent 32-bit limb additions"});
    add({"simd-shuffle-xor", "simd", "hw_simd_shuffle", MetricKind::Updates,
         1u << 18, 256, 128, 5.0, 0, 0, 1, 0, true, "five register shuffles per update"});

    add({"field-10x26", "field", "hw_fe10x26", MetricKind::FieldMul,
         1u << 16, 256, 2, 1.0, 0, 0, 1, 0, true, "production libsecp256k1 representation"});
    add({"field-8x32-upstream", "field", "hw_fe8x32", MetricKind::FieldMul,
         1u << 16, 256, 1, 1.0, 0, 0, 1, 0, true, "mrtozner comparison implementation"});

    for (uint32_t n : {32u, 128u, 512u, 1024u, 2048u}) {
        CaseSpec c;
        c.name = "private-dynamic-" + std::to_string(n);
        c.category = "private";
        c.function = "hw_private_" + std::to_string(n);
        c.grid = 1u << 15;
        c.group = 256;
        c.iterations = std::max(64u, n / 2);
        c.note = std::to_string(n * 4) + " B logical thread-private array";
        add(std::move(c));
    }

    add({"memory-read-64B", "memory", "hw_mem_read64", MetricKind::Bytes,
         1u << 20, 256, 1, 1.0, 80, 0, 1, 0, false, "64 B read + 16 B vector sink"});
    add({"memory-write-64B", "memory", "hw_mem_write64", MetricKind::Bytes,
         1u << 20, 256, 1, 1.0, 64, 0, 1, 0, false, "64 B coalesced write"});
    add({"memory-copy-64B", "memory", "hw_mem_copy64", MetricKind::Bytes,
         1u << 20, 256, 1, 1.0, 128, 0, 1, 0, false, "64 B read + 64 B write"});
    add({"memory-random-16B", "memory", "hw_mem_random", MetricKind::Bytes,
         1u << 20, 256, 1, 1.0, 32, 0, 1, 0, false, "scattered 16 B read + 16 B sink"});

    for (uint32_t group : {32u, 64u, 128u, 256u, 512u, 1024u}) {
        CaseSpec c{"group-u32-" + std::to_string(group), "group", "hw_u32_mul_ilp4",
                   MetricKind::Updates, 1u << 18, group, 256, 4.0};
        c.note = "same ILP4 kernel, group-size sweep";
        add(std::move(c));
    }

    for (uint32_t hint : {0u, 256u, 512u, 1024u}) {
        CaseSpec c{"hint-field10-" + (hint ? std::to_string(hint) : std::string("default")),
                   "hint", "hw_fe10x26", MetricKind::FieldMul,
                   1u << 16, 256, 2, 1.0};
        c.pipelineHint = hint;
        c.note = hint ? "pipeline maxTotalThreads hint" : "descriptor-less pipeline baseline";
        add(std::move(c));
    }

    for (uint32_t kib : {1u, 8u, 16u, 24u, 32u}) {
        CaseSpec c{"threadgroup-capacity-" + std::to_string(kib) + "KiB", "threadgroup",
                   "hw_threadgroup", MetricKind::Barriers,
                   1u << 18, 256, 16, 2.0};
        c.threadgroupBytes = kib * 1024;
        c.stride = 1;
        c.note = "dynamic threadgroup allocation; first 1 KiB touched";
        add(std::move(c));
    }
    for (uint32_t stride : {1u, 2u, 4u, 8u, 16u, 32u}) {
        CaseSpec c{"threadgroup-stride-" + std::to_string(stride), "banks",
                   "hw_threadgroup", MetricKind::Barriers,
                   1u << 18, 256, 16, 2.0};
        c.threadgroupBytes = 32u * 1024;
        c.stride = stride;
        c.note = "32 KiB allocation; neighbor stride sweep";
        add(std::move(c));
    }
    return v;
}

NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

std::string errorText(NSError* error, const std::string& fallback) {
    return error ? std::string([[error localizedDescription] UTF8String]) : fallback;
}

} // namespace

int runMetalHardwareProfile(const MetalHardwareProfileOptions& options) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::cerr << "no Metal device\n"; return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { std::cerr << "cannot create Metal command queue\n"; return 1; }

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kMetalHardwareProfileSource];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "Metal hardware profile compile failed: " << errorText(error, "unknown error") << "\n";
            return 1;
        }

        constexpr NSUInteger kBufferBytes = 64u * 1024u * 1024u;
        id<MTLBuffer> output = [device newBufferWithLength:kBufferBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> input = [device newBufferWithLength:kBufferBytes options:MTLResourceStorageModeShared];
        if (!output || !input) { std::cerr << "cannot allocate Metal profile buffers\n"; return 1; }
        auto* inputWords = static_cast<uint32_t*>([input contents]);
        uint32_t x = 0x12345678U;
        for (NSUInteger i = 0; i < kBufferBytes / sizeof(uint32_t); ++i) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5; inputWords[i] = x;
        }
        std::memset([output contents], 0, kBufferBytes);

        NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = [NSMutableDictionary dictionary];
        auto pipelineFor = [&](const CaseSpec& spec) -> id<MTLComputePipelineState> {
            NSString* key = [NSString stringWithFormat:@"%s#%u", spec.function.c_str(), spec.pipelineHint];
            id<MTLComputePipelineState> cached = [pipelines objectForKey:key];
            if (cached) return cached;
            id<MTLFunction> function = [library newFunctionWithName:ns(spec.function)];
            if (!function) return nil;
            id<MTLComputePipelineState> pipeline = nil;
            NSError* pipelineError = nil;
            if (spec.pipelineHint) {
                MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
                descriptor.computeFunction = function;
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                descriptor.maxTotalThreadsPerThreadgroup = spec.pipelineHint;
                pipeline = [device newComputePipelineStateWithDescriptor:descriptor
                                                                   options:MTLPipelineOptionNone
                                                                reflection:nil error:&pipelineError];
            } else {
                pipeline = [device newComputePipelineStateWithFunction:function error:&pipelineError];
            }
            if (!pipeline) {
                std::cerr << "pipeline " << spec.function << " failed: "
                          << errorText(pipelineError, "unknown error") << "\n";
                return nil;
            }
            [pipelines setObject:pipeline forKey:key];
            return pipeline;
        };

        auto dispatch = [&](const CaseSpec& spec, id<MTLComputePipelineState> pipeline,
                            uint32_t iterations, uint32_t actualGroup) -> DispatchTime {
            DispatchTime result;
            auto wallStart = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder) { result.error = "cannot create Metal command encoder"; return result; }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder setBuffer:input offset:0 atIndex:1];
            HwParams params{iterations,
                            spec.function == "hw_mem_random" ? uint32_t(kBufferBytes / 16 - 1) : actualGroup - 1,
                            spec.stride, 0x6d2b79f5U};
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            if (spec.threadgroupBytes)
                [encoder setThreadgroupMemoryLength:spec.threadgroupBytes atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(spec.grid, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(actualGroup, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            result.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
            if ([command status] != MTLCommandBufferStatusCompleted) {
                result.error = errorText(command.error, "Metal command failed");
                return result;
            }
            double start = [command GPUStartTime], end = [command GPUEndTime];
            result.gpu = end > start ? end - start : 0.0;
            return result;
        };

        std::vector<CaseResult> results;
        const auto cases = buildCases();
        for (const auto& spec : cases) {
            if (!selected(spec, options.caseFilter)) continue;
            id<MTLComputePipelineState> pipeline = pipelineFor(spec);
            if (!pipeline) return 1;
            uint32_t width = static_cast<uint32_t>(std::max<NSUInteger>(1, pipeline.threadExecutionWidth));
            uint32_t maxThreads = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
            uint32_t actualGroup = std::min(spec.group, maxThreads);
            actualGroup -= actualGroup % width;
            if (!actualGroup) actualGroup = width;
            if (spec.threadgroupBytes > device.maxThreadgroupMemoryLength) {
                std::cerr << spec.name << ": requested threadgroup memory exceeds device limit\n";
                return 1;
            }

            uint32_t iterations = spec.iterations;
            DispatchTime warm = dispatch(spec, pipeline, iterations, actualGroup);
            if (!warm.error.empty()) { std::cerr << spec.name << ": " << warm.error << "\n"; return 1; }
            if (spec.calibrate && warm.gpu > 0.0) {
                constexpr double target = 0.025;
                for (int attempt = 0; attempt < 7; ++attempt) {
                    uint32_t next = iterations;
                    if (warm.gpu < target * 0.55 && iterations < (1u << 20)) next = iterations * 2;
                    else if (warm.gpu > target * 2.2 && iterations > 1) next = std::max(1u, iterations / 2);
                    if (next == iterations) break;
                    iterations = next;
                    warm = dispatch(spec, pipeline, iterations, actualGroup);
                    if (!warm.error.empty()) { std::cerr << spec.name << ": " << warm.error << "\n"; return 1; }
                }
            }

            double totalGpu = 0.0, totalWall = 0.0;
            uint32_t runs = 0;
            const double targetSeconds = spec.metric == MetricKind::Dispatch
                ? std::max(0.15, options.secondsPerCase) : std::max(0.05, options.secondsPerCase);
            const uint32_t maxRuns = spec.metric == MetricKind::Dispatch ? 8192 : 256;
            while ((totalWall < targetSeconds || runs < 3) && runs < maxRuns) {
                DispatchTime t = dispatch(spec, pipeline, iterations, actualGroup);
                if (!t.error.empty()) { std::cerr << spec.name << ": " << t.error << "\n"; return 1; }
                totalGpu += t.gpu; totalWall += t.wall; ++runs;
            }

            CaseResult row;
            row.spec = spec;
            row.actualGroup = actualGroup;
            row.iterations = iterations;
            row.dispatches = runs;
            row.executionWidth = width;
            row.maxThreads = maxThreads;
            row.staticThreadgroupBytes = static_cast<uint32_t>(pipeline.staticThreadgroupMemoryLength);
            row.gpuSeconds = totalGpu;
            row.wallSeconds = totalWall;
            row.gpuMicrosPerDispatch = runs ? totalGpu * 1e6 / runs : 0.0;
            if (spec.metric == MetricKind::Dispatch) {
                row.value = runs ? totalWall * 1e6 / runs : 0.0;
                row.unit = "us wall/dispatch";
            } else if (spec.metric == MetricKind::Bytes) {
                row.value = totalGpu > 0 ? double(spec.grid) * spec.bytesPerThread * runs / totalGpu / 1e9 : 0.0;
                row.unit = "GB/s";
            } else {
                double work = double(spec.grid) * iterations * spec.operationsPerIteration * runs;
                if (spec.metric == MetricKind::FieldMul) {
                    row.value = totalGpu > 0 ? work / totalGpu / 1e6 : 0.0;
                    row.unit = "M field-mul/s";
                } else if (spec.metric == MetricKind::Barriers) {
                    row.value = totalGpu > 0 ? work / totalGpu / 1e9 : 0.0;
                    row.unit = "G barriers/s";
                } else {
                    row.value = totalGpu > 0 ? work / totalGpu / 1e9 : 0.0;
                    row.unit = "G updates/s";
                }
            }
            if (spec.metric == MetricKind::FieldMul) {
                const uint32_t limbs = spec.function == "hw_fe10x26" ? 10u : 8u;
                const auto* words = static_cast<const uint32_t*>([output contents]);
                row.sampleWords.assign(words, words + limbs);
            }
            results.push_back(std::move(row));
        }

        if (results.empty()) {
            std::cerr << "no Metal hardware cases match '" << options.caseFilter << "'\nAvailable cases:\n";
            for (const auto& c : cases) std::cerr << "  " << c.name << " (" << c.category << ")\n";
            return 1;
        }

        NSUInteger highestFamily = 0;
        for (NSUInteger n = 1; n <= 10; ++n)
            if ([device supportsFamily:static_cast<MTLGPUFamily>(1000 + n)]) highestFamily = n;
        NSArray<id<MTLCounterSet>>* counterSets = device.counterSets ?: @[];
        std::string architecture = std::string([[device.architecture name] UTF8String]);

        std::cout << "Metal hardware profile (deterministic; no wallet/key data)\n"
                  << "device=" << [device.name UTF8String] << " architecture=" << architecture
                  << " Apple-family=" << highestFamily << " unified-memory=" << (device.hasUnifiedMemory ? "yes" : "no") << "\n"
                  << "device max threads=" << device.maxThreadsPerThreadgroup.width
                  << " threadgroup-memory=" << device.maxThreadgroupMemoryLength / 1024 << " KiB"
                  << " recommended-working-set=" << std::fixed << std::setprecision(2)
                  << double(device.recommendedMaxWorkingSetSize) / (1024.0 * 1024 * 1024) << " GiB"
                  << " counter-sets=" << counterSets.count << "\n\n";
        std::cout << std::left << std::setw(29) << "case"
                  << std::right << std::setw(6) << "group" << std::setw(7) << "SIMD"
                  << std::setw(8) << "maxTG" << std::setw(10) << "iters"
                  << std::setw(11) << "GPU ms" << std::setw(9) << "busy%"
                  << std::setw(14) << "value" << "  unit\n";
        std::cout << std::string(114, '-') << "\n";
        for (const auto& r : results) {
            double busy = r.wallSeconds > 0 ? 100.0 * r.gpuSeconds / r.wallSeconds : 0.0;
            std::cout << std::left << std::setw(29) << r.spec.name
                      << std::right << std::setw(6) << r.actualGroup << std::setw(7) << r.executionWidth
                      << std::setw(8) << r.maxThreads << std::setw(10) << r.iterations
                      << std::fixed << std::setprecision(2) << std::setw(11) << r.gpuSeconds * 1e3
                      << std::setw(9) << busy << std::setprecision(2) << std::setw(14) << r.value
                      << "  " << r.unit << "\n";
        }

        if (!options.jsonPath.empty()) {
            std::ofstream out(options.jsonPath, std::ios::binary);
            if (!out) { std::cerr << "cannot write " << options.jsonPath << "\n"; return 1; }
            out << "{\n  \"device\": \"" << jsonEscape([device.name UTF8String]) << "\",\n"
                << "  \"architecture\": \"" << jsonEscape(architecture) << "\",\n"
                << "  \"apple_family\": " << highestFamily << ",\n"
                << "  \"unified_memory\": " << (device.hasUnifiedMemory ? "true" : "false") << ",\n"
                << "  \"device_max_threads_per_threadgroup\": " << device.maxThreadsPerThreadgroup.width << ",\n"
                << "  \"max_threadgroup_memory_bytes\": " << device.maxThreadgroupMemoryLength << ",\n"
                << "  \"recommended_working_set_bytes\": " << device.recommendedMaxWorkingSetSize << ",\n"
                << "  \"counter_sets\": [";
            for (NSUInteger i = 0; i < counterSets.count; ++i) {
                if (i) out << ',';
                out << "\"" << jsonEscape([[counterSets[i] name] UTF8String]) << "\"";
            }
            out << "],\n  \"seconds_per_case\": " << options.secondsPerCase << ",\n  \"results\": [\n";
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                double busy = r.wallSeconds > 0 ? 100.0 * r.gpuSeconds / r.wallSeconds : 0.0;
                out << "    {\"name\":\"" << jsonEscape(r.spec.name)
                    << "\",\"category\":\"" << jsonEscape(r.spec.category)
                    << "\",\"function\":\"" << jsonEscape(r.spec.function)
                    << "\",\"group_size\":" << r.actualGroup
                    << ",\"pipeline_hint\":" << r.spec.pipelineHint
                    << ",\"simd_width\":" << r.executionWidth
                    << ",\"pipeline_max_threads\":" << r.maxThreads
                    << ",\"static_threadgroup_bytes\":" << r.staticThreadgroupBytes
                    << ",\"dynamic_threadgroup_bytes\":" << r.spec.threadgroupBytes
                    << ",\"iterations\":" << r.iterations
                    << ",\"dispatches\":" << r.dispatches
                    << ",\"gpu_seconds\":" << std::setprecision(12) << r.gpuSeconds
                    << ",\"wall_seconds\":" << r.wallSeconds
                    << ",\"gpu_busy_percent\":" << busy
                    << ",\"gpu_us_per_dispatch\":" << r.gpuMicrosPerDispatch
                    << ",\"value\":" << r.value
                    << ",\"unit\":\"" << jsonEscape(r.unit)
                    << "\",\"note\":\"" << jsonEscape(r.spec.note) << "\",\"sample_words_le\":[";
                for (size_t j = 0; j < r.sampleWords.size(); ++j) {
                    if (j) out << ',';
                    out << r.sampleWords[j];
                }
                out << "]}";
                out << (i + 1 == results.size() ? "\n" : ",\n");
            }
            out << "  ]\n}\n";
            std::cout << "\nJSON: " << options.jsonPath << "\n";
        }
        return 0;
    }
}
