#include "run_config.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

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
        return std::find(args.begin(), args.end(), (dir / "words.txt").string()) != args.end() &&
               std::find(args.begin(), args.end(), "--strict-backend") != args.end() &&
               std::find(args.begin(), args.end(), "--opencl-sha-ring") == args.end();
    };
    const bool okay =
        check("\xEF\xBB\xBF# config\nwords=words.txt\nstrict-backend=true\nopencl-sha-ring=false\n", true) &&
        check("words=words.txt\nwords=other.txt\n", false) &&
        check("words=words.txt\nsecret-key=oops\n", false) &&
        check("strict-backend=perhaps\n", false);
    std::filesystem::remove(config);
    std::filesystem::remove(dir);
    return okay ? 0 : 1;
}
