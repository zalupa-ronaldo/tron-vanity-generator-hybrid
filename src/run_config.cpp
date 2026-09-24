#include "run_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#else
#  include <unistd.h>
#endif

namespace {
std::string trim(std::string s) {
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}
}

std::filesystem::path executableDirectory(const char* argv0) {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const DWORD count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (count && count < path.size()) return std::filesystem::path(path.substr(0, count)).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) == 0)
        return std::filesystem::weakly_canonical(std::filesystem::path(path.c_str())).parent_path();
#else
    std::string path(4096, '\0');
    const auto count = readlink("/proc/self/exe", path.data(), path.size());
    if (count > 0 && static_cast<size_t>(count) < path.size())
        return std::filesystem::path(path.substr(0, static_cast<size_t>(count))).parent_path();
#endif
    return std::filesystem::absolute(argv0).parent_path();
}

bool loadRunConfig(const std::filesystem::path& path,
                   std::vector<std::string>& arguments, std::string& error) {
    static const std::set<std::string> values = {
        "backend", "seconds", "threads", "words", "out", "output", "gpu-rng",
        "gpu-buffer-mb", "gpu-chunk-ms", "gpu-group-size", "metal-keys-per-lane",
        "vulkan-curve-batch", "vulkan-affine-batch", "vulkan-batch-keys",
        "vulkan-resident-group",
        "opencl-pipeline", "opencl-inverse", "opencl-affine-batch", "opencl-curve-batch",
        "opencl-compiler", "opencl-opt-mask"
    };
    static const std::set<std::string> flags = {
        "gpu-resident", "opencl-async-meta-read", "opencl-sha-ring", "opencl-host-seed",
        "case-sensitive", "strict-backend", "verbose"
    };
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open config: " + path.string(); return false; }
    std::set<std::string> seen;
    std::vector<std::string> parsed;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (lineNo == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        line = trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';') continue;
        const auto equals = line.find('=');
        const std::string key = trim(line.substr(0, equals));
        const std::string value = equals == std::string::npos ? "" : trim(line.substr(equals + 1));
        auto fail = [&](const std::string& why) {
            error = path.string() + ":" + std::to_string(lineNo) + ": " + why;
            return false;
        };
        if (equals == std::string::npos || key.empty() || value.empty()) return fail("expected key=value");
        if (!seen.insert(key).second) return fail("duplicate key: " + key);
        if (flags.count(key)) {
            if (value == "true" || value == "1" || value == "yes") parsed.push_back("--" + key);
            else if (value != "false" && value != "0" && value != "no") return fail("expected true or false for " + key);
            continue;
        }
        if (!values.count(key)) return fail("unknown key: " + key);
        parsed.push_back("--" + key);
        if (key == "words" || key == "out" || key == "output") {
            std::filesystem::path target(value);
            if (target.is_relative()) target = path.parent_path() / target;
            parsed.push_back(target.lexically_normal().string());
        } else parsed.push_back(value);
    }
    if (in.bad()) { error = "error reading config: " + path.string(); return false; }
    arguments.insert(arguments.end(), parsed.begin(), parsed.end());
    return true;
}
