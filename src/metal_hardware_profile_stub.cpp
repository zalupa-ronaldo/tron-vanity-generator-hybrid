#include "metal_hardware_profile.h"

#if !defined(__APPLE__)
#include <iostream>

int runMetalHardwareProfile(const MetalHardwareProfileOptions&) {
    std::cerr << "--metal-hw-profile requires macOS with Metal\n";
    return 1;
}
#endif
