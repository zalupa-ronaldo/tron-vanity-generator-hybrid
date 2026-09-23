#include "vulkan_backend.h"

#include <iostream>

std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary>) {
    return nullptr;
}

int vulkanResidentSelfTest() {
    std::cerr << "native Vulkan backend is not enabled in this build\n";
    return 77;
}
