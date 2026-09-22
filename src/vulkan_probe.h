#pragma once

#include <string>

// A deterministic API/compute smoke test, not a wallet-generating backend.
int vulkanComputeSelfTest();
std::string vulkanDeviceSummary();
