#include "vulkan_probe.h"

#if !defined(TRON_ENABLE_VULKAN)
#include <iostream>

int vulkanComputeSelfTest() {
    std::cerr << "Vulkan compute was not built. Reconfigure with -DTRON_ENABLE_VULKAN=ON and a Vulkan SDK.\n";
    return 77;
}

int vulkanKeccakSelfTest() {
    std::cerr << "Vulkan Keccak stage was not built. Reconfigure with -DTRON_ENABLE_VULKAN=ON and a Vulkan SDK.\n";
    return 77;
}

std::string vulkanDeviceSummary() { return "Vulkan compute not built"; }
#endif
