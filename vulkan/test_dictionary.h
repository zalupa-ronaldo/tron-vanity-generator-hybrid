#pragma once

#include "dictionary.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// Keep the no-wallet Vulkan self-test portable in downloaded CI artifacts.
// The 30-word subset keeps both bounded and overflowing match-ID vectors in
// the per-stage tests. This is the exact list in tests/vulkan_words.txt.
inline std::shared_ptr<Dictionary> vulkanTestDictionary() {
    std::vector<std::string> words;
    for (char c : std::string("123456789ABCDEFGHJKLMNPQRSTUVW"))
        words.emplace_back(1, c);
    return Dictionary::fromWords(std::move(words), true, nullptr);
}

// The full Base58 alphabet guarantees one result per address in the reusable
// backend test, filling its 32,768-slot ring before the base rollover.
inline std::shared_ptr<Dictionary> vulkanFullAlphabetTestDictionary() {
    std::vector<std::string> words;
    for (char c : std::string("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"))
        words.emplace_back(1, c);
    return Dictionary::fromWords(std::move(words), true, nullptr);
}
