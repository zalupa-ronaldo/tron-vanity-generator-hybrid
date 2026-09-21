#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "backend.h"

bool metalAvailable(std::string* note = nullptr);
std::string metalDeviceSummary();

struct MetalProfileResult {
    uint64_t keys = 0;
    double wallSeconds = 0.0;
    double gpuSeconds = 0.0;
    uint32_t dispatches = 0;
    std::string error;
};

MetalProfileResult profileMetalResidentStage(
    std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng,
    uint32_t bufferMiB,
    uint32_t chunkMs,
    uint32_t groupSize,
    uint32_t keysPerLane,
    uint32_t stage,
    double seconds,
    bool scalarKeccak = true);

std::unique_ptr<Backend> makeMetalResidentBackend(
    std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng,
    uint32_t bufferMiB,
    uint32_t chunkMs,
    uint32_t groupSize = 256,
    uint32_t keysPerLane = 32,
    bool scalarKeccak = true);
