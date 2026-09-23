#pragma once

#include "dictionary.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// Keep the no-wallet Vulkan self-test portable in downloaded CI artifacts.
// This is the exact 30 one-character words from tests/vulkan_words.txt.
inline std::shared_ptr<Dictionary> vulkanTestDictionary() {
    std::vector<std::string> words;
    for (char c : std::string("123456789ABCDEFGHJKLMNPQRSTUVW"))
        words.emplace_back(1, c);
    return Dictionary::fromWords(std::move(words), true, nullptr);
}
