#include "run_config.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("tron-config-test-" + std::to_string(suffix));
    std::filesystem::create_directory(dir);
    const auto config = dir / "tron-vanity.conf";
    auto check = [&](const std::string& content, bool expected) {
        std::ofstream out(config, std::ios::binary | std::ios::trunc);
        out << content;
        out.close();
        std::vector<std::string> args;
        std::string error;
        if (loadRunConfig(config, args, error) != expected) {
            std::cerr << "Config outcome mismatch: " << error << "\n";
            return false;
        }
        if (!expected) return args.empty();
        const auto batch = std::find(args.begin(), args.end(), "--vulkan-curve-batch");
        if (content.find("vulkan-curve-batch") != std::string::npos &&
            (batch == args.end() || std::next(batch) == args.end() || *std::next(batch) != "4"))
            return false;
        const auto batchKeys = std::find(args.begin(), args.end(), "--vulkan-batch-keys");
        if (content.find("vulkan-batch-keys") != std::string::npos &&
            (batchKeys == args.end() || std::next(batchKeys) == args.end() ||
             *std::next(batchKeys) != "131072"))
            return false;
        const auto affineBatch = std::find(args.begin(), args.end(), "--vulkan-affine-batch");
        if (content.find("vulkan-affine-batch") != std::string::npos &&
            (affineBatch == args.end() || std::next(affineBatch) == args.end() ||
             *std::next(affineBatch) != "4"))
            return false;
        const auto residentGroup = std::find(args.begin(), args.end(), "--vulkan-resident-group");
        if (content.find("vulkan-resident-group") != std::string::npos &&
            (residentGroup == args.end() || std::next(residentGroup) == args.end() ||
             *std::next(residentGroup) != "8"))
            return false;
        return std::find(args.begin(), args.end(), (dir / "words.txt").string()) != args.end() &&
               std::find(args.begin(), args.end(), "--strict-backend") != args.end() &&
               std::find(args.begin(), args.end(), "--opencl-sha-ring") == args.end();
    };
    const bool okay =
        check("\xEF\xBB\xBF# config\nwords=words.txt\nstrict-backend=true\nopencl-sha-ring=false\n", true) &&
        check("backend=vulkan\nvulkan-curve-batch=4\nvulkan-affine-batch=4\nvulkan-batch-keys=131072\nvulkan-resident-group=8\nwords=words.txt\nstrict-backend=true\n", true) &&
        check("words=words.txt\nwords=other.txt\n", false) &&
        check("words=words.txt\nsecret-key=oops\n", false) &&
        check("strict-backend=perhaps\n", false);
    std::filesystem::remove(config);
    std::filesystem::remove(dir);
    return okay ? 0 : 1;
}
