#pragma once

#include <cstdint>
#include <string>

struct MetalHardwareProfileOptions {
    double secondsPerCase = 0.35;
    std::string caseFilter = "all";
    std::string jsonPath;
};

// Runs deterministic, non-secret microbenchmarks on the default Metal device.
// Returns zero on success and prints a compact table to stdout.
int runMetalHardwareProfile(const MetalHardwareProfileOptions& options);
