// CPU execution of the exact shared CUDA kernel body, checked against
// libsecp256k1 and the independent host address encoder. This is a correctness
// test, not a substitute for NVIDIA device execution or performance testing.
#include "crypto.h"
#include <secp256k1.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel {
struct Dimension { unsigned x = 0; } blockIdx, blockDim, threadIdx;
unsigned atomicAdd(unsigned* p, unsigned value) { unsigned old = *p; *p += value; return old; }
unsigned __umulhi(unsigned a, unsigned b) { return (static_cast<uint64_t>(a) * b) >> 32; }
#define __device__
#define __forceinline__ inline
#define __global__
#define __constant__
#include "../kernels/tron_vanity_cuda.cu"
#undef inline
#undef __global
#undef __private
#undef __constant
#undef __kernel
#undef __device__
#undef __forceinline__
#undef __global__
#undef __constant__
#undef atomic_add
#undef get_global_id
#undef mul_hi
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::array<unsigned char, 64> pubXY(secp256k1_context* context, const unsigned char* sk) {
    secp256k1_pubkey pub;
    require(secp256k1_ec_pubkey_create(context, &pub, sk), "reference public key failed");
    unsigned char encoded[65];
    size_t size = sizeof(encoded);
    require(secp256k1_ec_pubkey_serialize(context, encoded, &size, &pub, SECP256K1_EC_UNCOMPRESSED), "serialize failed");
    std::array<unsigned char, 64> out{};
    std::memcpy(out.data(), encoded + 1, out.size());
    return out;
}

int main() {
    try {
        auto* context = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        std::vector<unsigned char> table(4 * 256 * 64, 0);
        for (unsigned window = 0; window < 4; ++window) for (unsigned digit = 1; digit < 256; ++digit) {
            std::array<unsigned char, 32> scalar{};
            scalar[31 - window] = static_cast<unsigned char>(digit);
            auto point = pubXY(context, scalar.data());
            std::memcpy(table.data() + (window * 256 + digit) * 64, point.data(), 64);
        }
        std::array<unsigned char, 32> seed{}, base{};
        for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<unsigned char>(i);
        kernel::blockDim.x = 1;
        kernel::tron_vanity_resident_seed(seed.data(), base.data(), 42);
        require(secp256k1_ec_seckey_verify(context, base.data()), "seed scalar invalid");
        std::array<unsigned char, 32> repeated{}, different{};
        kernel::tron_vanity_resident_seed(seed.data(), repeated.data(), 42);
        kernel::tron_vanity_resident_seed(seed.data(), different.data(), 43);
        require(base == repeated && base != different, "RNG reproducibility/counter mismatch");

        // The synthetic DFA accepts all addresses so no broken candidate can
        // hide behind the rarity of a real dictionary match.
        std::array<uint32_t, 58> dfa{};
        uint32_t start = 0, length = 1, id = 0;
        const std::array<unsigned char, 32> orderMinusOne = {
            0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,
            0xba,0xae,0xdc,0xe6,0xaf,0x48,0xa0,0x3b,0xbf,0xd2,0x5e,0x8c,0xd0,0x36,0x41,0x40};
        size_t checked = 0;
        for (const auto& baseKey : {base, orderMinusOne}) {
            auto pub = pubXY(context, baseKey.data());
            for (uint32_t gid : {0u, 1u, 127u, 128u, 32767u, 32768u, 524287u, 524288u, 0x7fffffffu}) {
                uint32_t meta[5]{};
                unsigned char records[2 * 160]{};
                kernel::threadIdx.x = gid;
                kernel::tron_vanity_resident_probe(baseKey.data(), pub.data(), table.data(), dfa.data(),
                    &start, &length, &id, meta, records, 2, 100);
                unsigned expectedCount = 0;
                for (uint32_t item = 0; item < 2; ++item) {
                    const uint32_t offset = gid * 2 + item;
                    auto expected = baseKey;
                    uint64_t carry = offset;
                    for (int i = 31; i >= 0; --i) {
                        carry += expected[i];
                        expected[i] = static_cast<unsigned char>(carry);
                        carry >>= 8;
                    }
                    if (carry || !secp256k1_ec_seckey_verify(context, expected.data())) continue;
                    const auto* record = records + expectedCount * 160;
                    require(std::memcmp(record, expected.data(), 32) == 0, "offset scalar mismatch");
                    auto expectedPub = pubXY(context, expected.data());
                    auto address = tronAddressFromPubXY(expectedPub.data());
                    require(address == std::string(reinterpret_cast<const char*>(record + 32), 34), "address mismatch");
                    require(record[68] == 1 && record[72] == 0, "DFA match record mismatch");
                    uint64_t sequence = 0;
                    for (unsigned j = 0; j < 8; ++j) sequence |= uint64_t(record[140 + j]) << (8 * j);
                    require(sequence == 100ULL + offset, "sequence mismatch");
                    ++expectedCount;
                    ++checked;
                }
                require(meta[0] == expectedCount && meta[2] == 0, "ring count mismatch");
            }
        }
        kernel::threadIdx.x = 0;
        auto pub = pubXY(context, base.data());
        uint32_t meta[5]{};
        std::array<unsigned char, 2 * 160> records{};
        records.fill(0xa5);
        kernel::tron_vanity_resident_probe(base.data(), pub.data(), table.data(), dfa.data(),
            &start, &length, &id, meta, records.data(), 1, 0);
        require(meta[0] == 2 && meta[2] == 1 && meta[3] == 1, "overflow not flagged");
        for (size_t i = 160; i < records.size(); ++i) require(records[i] == 0xa5, "ring wrote past capacity");
        length = 0;
        std::memset(meta, 0, sizeof(meta));
        kernel::tron_vanity_resident_probe(base.data(), pub.data(), table.data(), dfa.data(),
            &start, &length, &id, meta, records.data(), 2, 0);
        require(meta[0] == 0, "disabled output still emitted");
        secp256k1_context_destroy(context);
        std::cout << "CUDA shared kernel RNG " << RESIDENT_RNG << ": " << checked
                  << " boundary vectors, RNG counters, overflow and disabled-output checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
