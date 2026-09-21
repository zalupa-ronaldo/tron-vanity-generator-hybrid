#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "backend.h"

bool metalAvailable(std::string* note = nullptr);
std::string metalDeviceSummary();

std::unique_ptr<Backend> makeMetalResidentBackend(
    std::shared_ptr<const Dictionary> dictionary,
    const std::string& rng,
    uint32_t bufferMiB,
    uint32_t chunkMs,
    uint32_t pollMs,
    uint32_t groupSize = 256);
