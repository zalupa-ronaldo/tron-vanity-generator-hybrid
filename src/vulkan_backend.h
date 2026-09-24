#pragma once

#include <memory>
#include <array>
#include <cstdint>
#include <string>

#include "backend.h"

// Explicit native Vulkan backend. Never silently substitutes a CPU path.
std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary, uint32_t curveBatch = 4,
    uint32_t batchKeys = 131072, uint32_t affineBatch = 8,
    uint32_t residentDispatches = 16, bool field8 = false);

// Bounded, no-wallet runtime verification on the selected Vulkan device.
int vulkanResidentSelfTest(bool field8 = false);

struct VulkanProfileResult {
    std::string device;
    std::string error;
    uint64_t keys = 0;
    uint64_t dispatches = 0;
    uint64_t queueSubmits = 0;
    uint32_t gpuStageDispatchesPerSubmit = 6;
    double wallSeconds = 0.0;
    double gpuSeconds = 0.0;
    std::array<double, 6> stageSeconds{};
    // Host intervals are inclusive wall time, not disjoint CPU utilization.
    // Fence wait includes GPU execution and queue scheduling.
    std::array<double, 5> hostSeconds{}; // setup, record, submit, fence, collect
    uint64_t candidateRecords = 0;
    bool timestampsSupported = false;
    bool deviceLocalHostVisible = false;
    bool gpuScratchDeviceLocal = false;
    uint32_t curveBatch = 4;
    uint32_t batchKeys = 131072;
    uint32_t residentDispatches = 16;
    uint32_t affineBatch = 8;
    std::string fieldRepresentation = "10x26";
};

// Full-address, no-wallet wall/GPU-stage profile. No CPU or software fallback.
VulkanProfileResult profileVulkanResident(std::shared_ptr<const Dictionary> dictionary,
                                          double seconds, uint32_t curveBatch = 4,
                                          uint32_t batchKeys = 131072, uint32_t affineBatch = 8,
                                          uint32_t residentDispatches = 16,
                                          bool field8 = false);
