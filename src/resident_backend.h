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
    uint32_t pollMs);
