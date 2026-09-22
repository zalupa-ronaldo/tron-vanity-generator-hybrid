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
#define RESIDENT_SPLIT_TEST 1
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

// CPU-execute all six exact staged kernels with tiny buffers. offset_base
// exercises the full 32-bit offset space without allocating huge test arrays.
void checkedProbe(const unsigned char* base, const unsigned char* pub, const unsigned char* table,
                  const uint32_t* dfa, const uint32_t* start, const uint32_t* length, const uint32_t* id,
                  uint32_t* meta, unsigned char* records, uint32_t capacity, uint64_t sequence) {
    const uint32_t gid = kernel::threadIdx.x;
    uint32_t stagedMeta[5];
    unsigned char stagedRecords[320];
    std::memcpy(stagedMeta, meta, sizeof(stagedMeta));
    std::memcpy(stagedRecords, records, sizeof(stagedRecords));
    kernel::tron_vanity_resident_probe(base, pub, table, dfa, start, length, id, meta, records, capacity, sequence);
    uint32_t points[64]{};
    unsigned char pubs[128]{}, payloads[42]{}, fulls[50]{}, addresses[68]{};
    kernel::threadIdx.x = 0;
    kernel::resident_stage_curve(pub, table, points, gid * 2U);
    kernel::resident_stage_affine(points, pubs);
    for (unsigned item = 0; item < 2; ++item) {
        kernel::threadIdx.x = item;
        kernel::resident_stage_keccak(pubs, payloads);
        kernel::resident_stage_checksum(payloads, fulls);
        kernel::resident_stage_base58(fulls, addresses);
        kernel::resident_stage_match(base, addresses, dfa, start, length, id,
                                     stagedMeta, stagedRecords, capacity, sequence, gid * 2U);
    }
    kernel::threadIdx.x = gid;
    require(std::memcmp(meta, stagedMeta, sizeof(stagedMeta)) == 0, "staged metadata mismatch");
    require(std::memcmp(records, stagedRecords, sizeof(stagedRecords)) == 0, "staged record mismatch");
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
        auto orderMinusTwo = orderMinusOne;
        --orderMinusTwo[31];
        std::vector<std::array<unsigned char, 32>> bases{base, orderMinusOne, orderMinusTwo};
        for (unsigned counter = 100; counter < 108; ++counter) {
            std::array<unsigned char, 32> nextBase{};
            kernel::tron_vanity_resident_seed(seed.data(), nextBase.data(), counter);
            require(secp256k1_ec_seckey_verify(context, nextBase.data()), "test scalar invalid");
            bases.push_back(nextBase);
        }
        size_t checked = 0;
        for (const auto& baseKey : bases) {
            auto pub = pubXY(context, baseKey.data());
            for (uint32_t gid : {0u, 1u, 127u, 128u, 32767u, 32768u, 524287u, 524288u, 0x7fffffffu}) {
                uint32_t meta[5]{};
                unsigned char records[2 * 160]{};
                kernel::threadIdx.x = gid;
                checkedProbe(baseKey.data(), pub.data(), table.data(), dfa.data(),
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
        // Exercise a single range with 1,024 consecutive offsets, independent
        // of the device runtime test and including both lanes of every pair.
        auto rangePub = pubXY(context, base.data());
        for (uint32_t gid = 0; gid < 512; ++gid) {
            kernel::threadIdx.x = gid;
            uint32_t rangeMeta[5]{};
            unsigned char rangeRecords[2 * 160]{};
            checkedProbe(base.data(), rangePub.data(), table.data(), dfa.data(),
                &start, &length, &id, rangeMeta, rangeRecords, 2, 0);
            require(rangeMeta[0] == 2 && rangeMeta[2] == 0, "range count mismatch");
            for (unsigned item = 0; item < 2; ++item) {
                const auto* record = rangeRecords + item * 160;
                require(secp256k1_ec_seckey_verify(context, record), "range scalar invalid");
                auto expectedPub = pubXY(context, record);
                require(tronAddressFromPubXY(expectedPub.data()) ==
                    std::string(reinterpret_cast<const char*>(record + 32), 34), "range address mismatch");
                ++checked;
            }
        }
        kernel::threadIdx.x = 0;
        auto pub = pubXY(context, base.data());
        uint32_t meta[5]{};
        std::array<unsigned char, 2 * 160> records{};
        records.fill(0xa5);
        checkedProbe(base.data(), pub.data(), table.data(), dfa.data(),
            &start, &length, &id, meta, records.data(), 1, 0);
        require(meta[0] == 2 && meta[2] == 1 && meta[3] == 1, "overflow not flagged");
        for (size_t i = 160; i < records.size(); ++i) require(records[i] == 0xa5, "ring wrote past capacity");
        length = 0;
        std::memset(meta, 0, sizeof(meta));
        checkedProbe(base.data(), pub.data(), table.data(), dfa.data(),
            &start, &length, &id, meta, records.data(), 2, 0);
        require(meta[0] == 0, "disabled output still emitted");
        secp256k1_context_destroy(context);
        std::cout << "Shared kernel RNG " << RESIDENT_RNG << ", pair inverse " << RESIDENT_PAIR_INVERSE << ": " << checked
                  << " boundary vectors (monolithic + staged), RNG counters, overflow and disabled-output checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
