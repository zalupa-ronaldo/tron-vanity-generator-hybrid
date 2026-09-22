#pragma once

#include <cstdint>
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
};

struct OpenclProfileResult {
    uint64_t keys = 0, dispatches = 0;
    double wallSeconds = 0, baseSeconds = 0, scanSeconds = 0, gpuSeconds = 0;
    double maxGpuMs = 0;
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
