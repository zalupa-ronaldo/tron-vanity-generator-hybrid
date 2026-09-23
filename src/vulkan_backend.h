#pragma once

#include <memory>

#include "backend.h"

// Explicit native Vulkan backend. Never silently substitutes a CPU path.
std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary);

// Bounded, no-wallet runtime verification on the selected Vulkan device.
int vulkanResidentSelfTest();
