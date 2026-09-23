#pragma once

#include <memory>
#include <array>
#include <cstdint>
#include <string>

#include "backend.h"

// Explicit native Vulkan backend. Never silently substitutes a CPU path.
std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary, uint32_t curveBatch = 1,
    uint32_t batchKeys = 131072);

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
    // Host intervals are inclusive wall time, not disjoint CPU utilization.
    // Fence wait includes GPU execution and queue scheduling.
    std::array<double, 5> hostSeconds{}; // setup, record, submit, fence, collect
    uint64_t candidateRecords = 0;
    bool timestampsSupported = false;
    bool deviceLocalHostVisible = false;
    uint32_t curveBatch = 1;
    uint32_t batchKeys = 131072;
};

// Full-address, no-wallet wall/GPU-stage profile. No CPU or software fallback.
VulkanProfileResult profileVulkanResident(std::shared_ptr<const Dictionary> dictionary,
                                          double seconds, uint32_t curveBatch = 1,
                                          uint32_t batchKeys = 131072);
