#include "dictionary.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <utility>

namespace {
struct Node {
    std::array<uint32_t, Dictionary::Alphabet> next{};
    uint32_t fail = 0;
    std::vector<uint32_t> outputs;
    Node() { next.fill(UINT32_MAX); }
};

std::string upperAscii(std::string s) {
    for (char& c : s) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return s;
}
}

int Dictionary::alphabetIndex(char c) {
    static constexpr char b58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (int i = 0; i < 58; ++i) if (b58[i] == c) return i;
    return -1;
}

std::shared_ptr<Dictionary> Dictionary::load(const std::string& path, bool caseSensitive,
                                              std::string* error) {
    std::ifstream in(path);
    if (!in) { if (error) *error = "cannot read dictionary: " + path; return nullptr; }
    std::vector<std::string> source;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        line = line.substr(first);
        if (line.empty() || line.size() > 33) continue;
        source.push_back(line);
    }
    return fromWords(std::move(source), caseSensitive, error);
}

std::shared_ptr<Dictionary> Dictionary::fromWords(std::vector<std::string> source,
                                                   bool caseSensitive, std::string* error) {
    for (const auto& word : source) {
        for (char c : word) if (alphabetIndex(c) < 0 &&
            (caseSensitive || alphabetIndex(static_cast<char>(c +
                ('a' <= c && c <= 'z' ? 'A' - 'a' : 'a' - 'A'))) < 0)) {
            if (error) *error = "dictionary word is not Base58-compatible: " + word;
            return nullptr;
        }
    }
    auto d = std::make_shared<Dictionary>();
    std::sort(source.begin(), source.end());
    source.erase(std::unique(source.begin(), source.end()), source.end());
    d->words = source;

    std::vector<Node> nodes(1);
    auto addPattern = [&](const std::string& pattern, uint32_t id) {
        uint32_t state = 0;
        for (char c : pattern) {
            int a = alphabetIndex(c);
            if (a < 0) return;
            if (nodes[state].next[a] == UINT32_MAX) {
                nodes[state].next[a] = static_cast<uint32_t>(nodes.size());
                nodes.emplace_back();
            }
            state = nodes[state].next[a];
        }
        nodes[state].outputs.push_back(id);
    };
    for (uint32_t i = 0; i < d->words.size(); ++i) {
        addPattern(d->words[i], i);
        if (!caseSensitive) addPattern(upperAscii(d->words[i]), i);
    }

    std::queue<uint32_t> q;
    for (uint32_t a = 0; a < Alphabet; ++a) {
        uint32_t child = nodes[0].next[a];
        if (child == UINT32_MAX) nodes[0].next[a] = 0;
        else { nodes[child].fail = 0; q.push(child); }
    }
    while (!q.empty()) {
        uint32_t v = q.front(); q.pop();
        uint32_t f = nodes[v].fail;
        nodes[v].outputs.insert(nodes[v].outputs.end(), nodes[f].outputs.begin(), nodes[f].outputs.end());
        for (uint32_t a = 0; a < Alphabet; ++a) {
            uint32_t child = nodes[v].next[a];
            if (child == UINT32_MAX) nodes[v].next[a] = nodes[f].next[a];
            else { nodes[child].fail = nodes[f].next[a]; q.push(child); }
        }
    }
    d->dfa.resize(nodes.size() * Alphabet);
    d->outStart.resize(nodes.size());
    d->outLen.resize(nodes.size());
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        std::copy(nodes[i].next.begin(), nodes[i].next.end(), d->dfa.begin() + i * Alphabet);
        d->outStart[i] = static_cast<uint32_t>(d->outIds.size());
        std::sort(nodes[i].outputs.begin(), nodes[i].outputs.end());
        nodes[i].outputs.erase(std::unique(nodes[i].outputs.begin(), nodes[i].outputs.end()), nodes[i].outputs.end());
        d->outLen[i] = static_cast<uint32_t>(nodes[i].outputs.size());
        d->outIds.insert(d->outIds.end(), nodes[i].outputs.begin(), nodes[i].outputs.end());
    }
    return d;
}

std::vector<uint32_t> Dictionary::matchIds(const std::string& address) const {
    std::vector<uint32_t> out;
    std::vector<uint8_t> seen(words.size(), 0);
    uint32_t state = 0;
    for (size_t i = 1; i < address.size(); ++i) {
        int a = alphabetIndex(address[i]);
        if (a < 0) { state = 0; continue; }
        state = dfa[state * Alphabet + static_cast<uint32_t>(a)];
        for (uint32_t j = 0; j < outLen[state]; ++j) {
            uint32_t id = outIds[outStart[state] + j];
            if (id < seen.size() && !seen[id]) { seen[id] = 1; out.push_back(id); }
        }
    }
    return out;
}

std::vector<std::string> Dictionary::matchWords(const std::string& address) const {
    std::vector<std::string> out;
    for (uint32_t id : matchIds(address)) out.push_back(words[id]);
    return out;
}
