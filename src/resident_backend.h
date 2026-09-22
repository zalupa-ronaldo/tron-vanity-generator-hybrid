#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <string>

#include "backend.h"

struct GpuDevice;

struct OpenclResidentOptions {
    bool pairInverse = false;
    bool compact = true;
    bool profiling = false;
    bool hostSeed = false;
    bool staged = false;
    uint32_t affineBatch = 4; // staged paired inversion; 2 retains the previous kernel, 8 is experimental
    uint32_t curveBatch = 2; // staged consecutive public points per work-item; 4/8 experimental
    uint32_t stageOptMask = 63; // curve, affine, keccak, checksum, Base58, match
    bool shaRing = false; // experimental 16-word SHA-256 message schedule
};

struct OpenclProfileResult {
    uint64_t keys = 0, dispatches = 0, basePairs = 0;
    double wallSeconds = 0, baseSeconds = 0, scanSeconds = 0, gpuSeconds = 0;
    double enqueueSeconds = 0, waitSeconds = 0, eventQuerySeconds = 0;
    double metaReadSeconds = 0, recordsSeconds = 0, metaWriteSeconds = 0;
    double maxGpuMs = 0;
    std::array<double, 6> stageSeconds{};
    bool gpuTimingValid = true;
    std::string error;
};

std::unique_ptr<Backend> makeResidentGpuBackend(
    const GpuDevice& device,
    std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng,
    uint32_t bufferMiB,
    uint32_t chunkMs,
    uint32_t pollMs,
    uint32_t groupSize = 256,
    OpenclResidentOptions options = {});

int openclResidentSelfTest(const GpuDevice& device, const std::string& rng,
                         OpenclResidentOptions options = {});
int diagnoseOpencl(const GpuDevice& device, const std::string& stage,
                   const std::string& rng, OpenclResidentOptions options = {});
OpenclProfileResult profileOpenclResident(const GpuDevice& device,
    std::shared_ptr<const Dictionary> dictionary, const std::string& rng,
    uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize,
    OpenclResidentOptions options, double seconds);

std::unique_ptr<Backend> makeCudaBackend(
    const GpuDevice& device, std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng, uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize = 256);
int cudaSelfTest(const GpuDevice& device);
