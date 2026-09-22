#pragma once

#include <filesystem>
#include <string>
#include <vector>

// Config is deliberately a small, non-executable key=value file. It may
// contain paths and performance settings, but never wallet secrets.
std::filesystem::path executableDirectory(const char* argv0);
bool loadRunConfig(const std::filesystem::path& path,
                   std::vector<std::string>& arguments, std::string& error);
