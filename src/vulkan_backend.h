#pragma once

#include <memory>
#include <array>
#include <cstdint>
#include <string>

#include "backend.h"

// Explicit native Vulkan backend. Never silently substitutes a CPU path.
std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary);

// Bounded, no-wallet runtime verification on the selected Vulkan device.
int vulkanResidentSelfTest();

struct VulkanProfileResult {
    std::string device;
    std::string error;
    uint64_t keys = 0;
    uint64_t dispatches = 0;
    double wallSeconds = 0.0;
    double gpuSeconds = 0.0;
    std::array<double, 5> stageSeconds{};
    bool timestampsSupported = false;
    bool deviceLocalHostVisible = false;
};

// Full-address, no-wallet wall/GPU-stage profile. No CPU or software fallback.
VulkanProfileResult profileVulkanResident(std::shared_ptr<const Dictionary> dictionary,
                                          double seconds);
