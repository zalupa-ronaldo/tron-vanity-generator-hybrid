#include "hwdetect.h"
#include "ocl.h"
#include "resident_backend.h"
#include <iostream>

int main() {
    std::string error;
    if (!ocl::load(&error)) { std::cout << error << "\n"; return 77; }
    const auto devices = ocl::enumerateTestDevices();
    if (devices.empty()) { std::cout << "No OpenCL test device\n"; return 77; }
    // May be CPU OpenCL (PoCL), not evidence of GPU performance. The test
    // actually builds OpenCL C and exercises arguments, queues and transfers.
    const auto& d = devices.front();
    std::cout << "OpenCL runtime integration test on " << d.name << " / " << d.platformName << "\n";
    GpuDevice device;
    device.name = d.name;
    device.platform = d.platformName;
    device.platformId = d.platform;
    device.deviceId = d.device;
    if (openclResidentSelfTest(device, "chacha12", {false, true, true})) return 1;
    for (const auto& rng : {"chacha12", "aes-ctr", "philox"})
        if (openclResidentSelfTest(device, rng, {true, true, true})) return 1;
    auto dictionary = std::make_shared<Dictionary>();
    dictionary->words = {"benchmark"};
    dictionary->dfa.assign(Dictionary::Alphabet, 0);
    dictionary->outStart = {0};
    dictionary->outLen = {0};
    dictionary->outIds = {0};
    auto profile = profileOpenclResident(device, dictionary, "chacha12", 8, 8, 64,
                                         {true, true, true}, 0.05);
    if (!profile.error.empty() || !profile.keys || !profile.dispatches ||
        profile.wallSeconds <= 0 || profile.keys % 128 != 0) {
        std::cerr << "OpenCL profile/count integration failed: " << profile.error << "\n";
        return 1;
    }
    if (!profile.gpuTimingValid || profile.gpuSeconds <= 0) {
        std::cerr << "OpenCL profiling queue did not return valid timestamps\n";
        return 1;
    }
    // Each self-test creates/destroys its context, programs, events and memory.
    return 0;
}
