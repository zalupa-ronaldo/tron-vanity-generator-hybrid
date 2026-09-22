#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "backend.h"

struct GpuDevice;

std::unique_ptr<Backend> makeResidentGpuBackend(
    const GpuDevice& device,
    std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng,
    uint32_t bufferMiB,
    uint32_t chunkMs,
    uint32_t pollMs,
    uint32_t groupSize = 256);

std::unique_ptr<Backend> makeCudaBackend(
    const GpuDevice& device, std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng, uint32_t bufferMiB, uint32_t chunkMs, uint32_t groupSize = 256);
int cudaSelfTest(const GpuDevice& device);
