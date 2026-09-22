#include "hwdetect.h"
#include "ocl.h"
#include "resident_backend.h"
#include <cmath>
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
    if (diagnoseOpencl(device, "smoke", "chacha12")) return 1;
    for (const auto& rng : {"chacha12", "aes-ctr", "philox"})
        if (diagnoseOpencl(device, "rng", rng)) return 1;
    for (const auto& stage : {"build-curve", "build-affine", "build-keccak", "build-checksum",
                              "build-base58", "build-match"})
        if (diagnoseOpencl(device, stage, "chacha12", {true, true, false})) return 1;
    if (diagnoseOpencl(device, "scan", "chacha12", {false, true, false})) return 1;
    if (diagnoseOpencl(device, "scan", "chacha12", {true, true, false})) return 1;
    if (openclResidentSelfTest(device, "chacha12", {false, true, true})) return 1;
    for (const auto& rng : {"chacha12", "aes-ctr", "philox"})
        if (openclResidentSelfTest(device, rng, {true, true, true})) return 1;
    if (openclResidentSelfTest(device, "chacha12", {false, true, true, true, true})) return 1;
    OpenclResidentOptions pairedTwo{false, true, false, true, true};
    pairedTwo.affineBatch = 2;
    if (openclResidentSelfTest(device, "chacha12", pairedTwo)) return 1;
    OpenclResidentOptions pairedEight{false, true, false, true, true};
    pairedEight.affineBatch = 8;
    if (openclResidentSelfTest(device, "chacha12", pairedEight)) return 1;
    for (uint32_t batch : {4U, 8U}) {
        OpenclResidentOptions curveBatch{false, true, false, true, true};
        curveBatch.curveBatch = batch;
        if (openclResidentSelfTest(device, "chacha12", curveBatch)) return 1;
    }
    for (uint32_t mask : {0U, 1U, 2U, 4U, 8U, 16U, 32U}) {
        OpenclResidentOptions isolated{false, true, false, true, true};
        isolated.stageOptMask = mask;
        if (openclResidentSelfTest(device, "chacha12", isolated)) {
            std::cerr << "Staged optimization mask " << mask << " failed\n";
            return 1;
        }
    }
    if (openclResidentSelfTest(device, "chacha12", {false, false, true, true, true})) return 1;
    if (openclResidentSelfTest(device, "chacha12", {true, true, true, true, true})) return 1;
    for (const auto& rng : {"chacha12", "aes-ctr", "philox"})
        if (openclResidentSelfTest(device, rng, {true, true, true, false, true})) return 1;
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
    auto hostProfile = profileOpenclResident(device, dictionary, "chacha12", 8, 8, 64,
                                             {true, true, true, true}, 0.05);
    if (!hostProfile.error.empty() || !hostProfile.keys || !hostProfile.dispatches ||
        hostProfile.wallSeconds <= 0 || hostProfile.keys % 128 != 0 || !hostProfile.gpuTimingValid) {
        std::cerr << "OS-seeded OpenCL profile failed: " << hostProfile.error << "\n";
        return 1;
    }
    // Each self-test creates/destroys its context, programs, events and memory.
    auto stagedProfile = profileOpenclResident(device, dictionary, "chacha12", 8, 8, 64,
                                               {true, true, true, true, true}, 0.05);
    double stageSeconds = 0;
    for (double seconds : stagedProfile.stageSeconds) stageSeconds += seconds;
    if (!stagedProfile.error.empty() || !stagedProfile.keys || !stagedProfile.dispatches ||
        !stagedProfile.gpuTimingValid || stagedProfile.gpuSeconds <= 0 ||
        stageSeconds <= 0 || std::abs(stageSeconds - stagedProfile.gpuSeconds) > 1e-6 ||
        stagedProfile.keys > stagedProfile.dispatches * (1u << 17)) {
        std::cerr << "Staged profile/event/cap integration failed: " << stagedProfile.error << "\n";
        return 1;
    }
    return 0;
}
