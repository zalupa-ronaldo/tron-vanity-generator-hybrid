#include "keccak.h"
#include <cstdint>

static inline uint64_t rotl64(uint64_t x, int n) {
    return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

static const uint64_t kRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// kRho[x][y] —— 标准 Keccak 旋转偏移（lane = x + 5*y）
static const int kRho[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14}
};

static void keccakF(uint64_t s[25]) {
    for (int round = 0; round < 24; ++round) {
        uint64_t c[5], d[5];
        for (int x = 0; x < 5; ++x)
            c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
        for (int x = 0; x < 5; ++x)
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                s[x + 5 * y] ^= d[x];

        uint64_t b[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(s[x + 5 * y], kRho[x][y]);

        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                s[x + 5 * y] = b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);

        s[0] ^= kRC[round];
    }
}

void keccak256(const unsigned char* data, size_t len, unsigned char output[32]) {
    uint64_t s[25] = {0};
    const size_t rate = 136;
    size_t off = 0;
    for (size_t i = 0; i < len; ++i) {
        s[off / 8] ^= static_cast<uint64_t>(data[i]) << ((off % 8) * 8);
        if (++off == rate) { keccakF(s); off = 0; }
    }
    s[off / 8] ^= static_cast<uint64_t>(0x01) << ((off % 8) * 8);
    s[(rate / 8) - 1] ^= static_cast<uint64_t>(0x80) << 56;
    keccakF(s);
    for (size_t i = 0; i < 32; ++i)
        output[i] = static_cast<unsigned char>(s[i / 8] >> ((i % 8) * 8));
}
