#include "metal_backend.h"

#if !defined(__APPLE__)
bool metalAvailable(std::string* note) {
    if (note) *note = "Metal is available only on Apple platforms";
    return false;
}

std::string metalDeviceSummary() { return "Metal unavailable"; }

std::unique_ptr<Backend> makeMetalResidentBackend(
    std::shared_ptr<const Dictionary>, const std::string&, uint32_t, uint32_t, uint32_t) {
    return nullptr;
}
#endif
