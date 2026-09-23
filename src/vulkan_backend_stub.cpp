#include "vulkan_backend.h"

#include <iostream>

std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary>, uint32_t) {
    return nullptr;
}

int vulkanResidentSelfTest() {
    std::cerr << "native Vulkan backend is not enabled in this build\n";
    return 77;
}

VulkanProfileResult profileVulkanResident(std::shared_ptr<const Dictionary>, double, uint32_t) {
    VulkanProfileResult result;
    result.error = "native Vulkan backend is not enabled in this build";
    return result;
}
