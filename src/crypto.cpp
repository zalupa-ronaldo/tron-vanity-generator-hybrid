#include "crypto.h"
#include "keccak.h"

#include <cstdint>
#include <cstring>
#include <cstdio>

const char kBase58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// ---------------- SHA-256（精简自公有领域实现） ----------------

namespace {

struct Sha256 {
    uint32_t s[8];
    uint64_t bitlen = 0;
    unsigned char buf[64];
    size_t idx = 0;

    Sha256() {
        static const uint32_t iv[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(s, iv, sizeof(s));
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) |
                   (uint32_t(p[4*i+2]) << 8) | uint32_t(p[4*i+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
    }

    void update(const unsigned char* p, size_t n) {
        bitlen += uint64_t(n) * 8;
        while (n) {
            size_t take = 64 - idx;
            if (take > n) take = n;
            std::memcpy(buf + idx, p, take);
            idx += take; p += take; n -= take;
            if (idx == 64) { block(buf); idx = 0; }
        }
    }

    void final(unsigned char out[32]) {
        unsigned char pad = 0x80;
        uint64_t bl = bitlen;
        update(&pad, 1);
        unsigned char zero = 0;
        while (idx != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bl >> (56 - 8*i));
        // update() 会改 bitlen，这里直接写块
        std::memcpy(buf + 56, len, 8);
        block(buf);
        for (int i = 0; i < 8; ++i) {
            out[4*i]   = static_cast<unsigned char>(s[i] >> 24);
            out[4*i+1] = static_cast<unsigned char>(s[i] >> 16);
            out[4*i+2] = static_cast<unsigned char>(s[i] >> 8);
            out[4*i+3] = static_cast<unsigned char>(s[i]);
        }
    }
};

void sha256(const unsigned char* data, size_t len, unsigned char out[32]) {
    Sha256 h;
    h.update(data, len);
    h.final(out);
}

std::string base58Encode(const unsigned char* input, size_t len) {
    size_t zeros = 0;
    while (zeros < len && input[zeros] == 0) ++zeros;

    std::vector<unsigned char> num(input + zeros, input + len);
    std::string digits;
    digits.reserve(len * 138 / 100 + 1);
    while (!num.empty()) {
        int carry = 0;
        for (size_t i = 0; i < num.size(); ++i) {
            int v = (carry << 8) + num[i];
            num[i] = static_cast<unsigned char>(v / 58);
            carry = v % 58;
        }
        digits.push_back(kBase58[carry]);
        size_t nz = 0;
        while (nz < num.size() && num[nz] == 0) ++nz;
        num.erase(num.begin(), num.begin() + nz);
    }
    std::string out(zeros, '1');
    out.append(digits.rbegin(), digits.rend());
    return out;
}

}  // namespace

std::string bytesToHexUpper(const unsigned char* data, size_t len) {
    static const char* h = "0123456789ABCDEF";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[2 * i] = h[data[i] >> 4];
        out[2 * i + 1] = h[data[i] & 0xF];
    }
    return out;
}

void tronFullFromPubXY(const unsigned char pubXY[64], unsigned char full[25]) {
    unsigned char hash[32];
    keccak256(pubXY, 64, hash);

    unsigned char payload[21];
    payload[0] = 0x41;
    std::memcpy(payload + 1, hash + 12, 20);

    unsigned char h1[32], h2[32];
    sha256(payload, 21, h1);
    sha256(h1, 32, h2);

    std::memcpy(full, payload, 21);
    std::memcpy(full + 21, h2, 4);
}

std::string tronAddressFromPubXY(const unsigned char pubXY[64]) {
    unsigned char full[25];
    tronFullFromPubXY(pubXY, full);
    return base58Encode(full, 25);
}

int hashtest() {
    unsigned char h[32];
    sha256(reinterpret_cast<const unsigned char*>("abc"), 3, h);
    std::printf("sha256(abc)  = %s\n", bytesToHexUpper(h, 32).c_str());
    std::printf("  expect     = BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD\n");
    keccak256(reinterpret_cast<const unsigned char*>(""), 0, h);
    std::printf("keccak256()  = %s\n", bytesToHexUpper(h, 32).c_str());
    std::printf("  expect     = C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470\n");
    return 0;
}
