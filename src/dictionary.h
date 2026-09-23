#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Dictionary {
    static constexpr uint32_t Alphabet = 58;

    std::vector<std::string> words;
    std::vector<uint32_t> dfa;
    std::vector<uint32_t> outStart;
    std::vector<uint32_t> outLen;
    std::vector<uint32_t> outIds;

    static std::shared_ptr<Dictionary> load(const std::string& path, bool caseSensitive,
                                            std::string* error);
    static std::shared_ptr<Dictionary> fromWords(std::vector<std::string> source,
                                                 bool caseSensitive, std::string* error);
    std::vector<uint32_t> matchIds(const std::string& address) const;
    std::vector<std::string> matchWords(const std::string& address) const;
    static int alphabetIndex(char c);
};
