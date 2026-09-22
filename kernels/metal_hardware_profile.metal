#include <metal_stdlib>
using namespace metal;

/*
 * Deterministic Apple-GPU microbenchmarks. No keys or wallet data enter these
 * kernels. A runtime iteration count prevents the compiler from replacing a
 * benchmark with a precomputed constant; every result is written to device
 * memory so the measured dependency chains remain observable.
 */

struct HwParams {
    uint iterations;
    uint mask;
    uint stride;
    uint seed;
};

static inline uint hw_rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

kernel void hw_dispatch(device uint *out [[buffer(0)]],
                        constant HwParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = p.seed ^ 0x6a09e667U;
}

kernel void hw_u32_mix_dep(device uint *out [[buffer(0)]],
                           constant HwParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint x = gid ^ p.seed;
    for (uint i = 0; i < p.iterations; ++i)
        x = hw_rotl32(x + (0x9e3779b9U ^ i), 7) ^ 0x85ebca6bU;
    out[gid] = x;
}

kernel void hw_u32_mix_ilp4(device uint *out [[buffer(0)]],
                            constant HwParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    uint a = gid ^ p.seed, b = a ^ 0x243f6a88U;
    uint c = a ^ 0x85a308d3U, d = a ^ 0x13198a2eU;
    for (uint i = 0; i < p.iterations; ++i) {
        a = hw_rotl32(a + (0x9e3779b9U ^ i), 5) ^ 0xa4093822U;
        b = hw_rotl32(b + (0x7f4a7c15U ^ i), 7) ^ 0x299f31d0U;
        c = hw_rotl32(c + (0x94d049bbU ^ i), 11) ^ 0x082efa98U;
        d = hw_rotl32(d + (0x5bd1e995U ^ i), 13) ^ 0xec4e6c89U;
    }
    out[gid] = a ^ b ^ c ^ d;
}

kernel void hw_u32_mul_dep(device uint *out [[buffer(0)]],
                           constant HwParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint x = gid ^ p.seed;
    for (uint i = 0; i < p.iterations; ++i)
        x = x * 1664525U + (1013904223U ^ i);
    out[gid] = x;
}

kernel void hw_u32_mul_ilp4(device uint *out [[buffer(0)]],
                            constant HwParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    uint a = gid ^ p.seed, b = a ^ 0x243f6a88U;
    uint c = a ^ 0x85a308d3U, d = a ^ 0x13198a2eU;
    for (uint i = 0; i < p.iterations; ++i) {
        a = a * 1664525U + (1013904223U ^ i);
        b = b * 22695477U + (1U ^ i);
        c = c * 1103515245U + (12345U ^ i);
        d = d * 214013U + (2531011U ^ i);
    }
    out[gid] = a ^ b ^ c ^ d;
}

kernel void hw_u64_mul_dep(device ulong *out [[buffer(0)]],
                           constant HwParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    ulong x = ((ulong)gid << 32) ^ p.seed;
    for (uint i = 0; i < p.iterations; ++i)
        x = x * 0x9e3779b97f4a7c15UL + (0xbf58476d1ce4e5b9UL ^ (ulong)i);
    out[gid] = x;
}

kernel void hw_u64_mul_ilp4(device ulong *out [[buffer(0)]],
                            constant HwParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    ulong a = ((ulong)gid << 32) ^ p.seed;
    ulong b = a ^ 0x243f6a8885a308d3UL;
    ulong c = a ^ 0x13198a2e03707344UL;
    ulong d = a ^ 0xa4093822299f31d0UL;
    for (uint i = 0; i < p.iterations; ++i) {
        a = a * 0x9e3779b97f4a7c15UL + (0xbf58476d1ce4e5b9UL ^ (ulong)i);
        b = b * 0x94d049bb133111ebUL + (0xd6e8feb86659fd93UL ^ (ulong)i);
        c = c * 0xd1342543de82ef95UL + (0xa5a3564e27f8862bUL ^ (ulong)i);
        d = d * 0x8cb92baa3f3d8dd7UL + (0xdb4f0b9175ae2165UL ^ (ulong)i);
    }
    out[gid] = a ^ b ^ c ^ d;
}

kernel void hw_carry8(device uint *out [[buffer(0)]],
                      constant HwParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint a[8], b[8];
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
        a[j] = (gid + 1U) * (0x9e3779b9U + j * 0x1000193U);
        b[j] = p.seed ^ (0x7f4a7c15U + j * 0x85ebca6bU);
    }
    for (uint i = 0; i < p.iterations; ++i) {
        uint carry = i & 1U;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            ulong sum = (ulong)a[j] + b[j] + carry;
            a[j] = (uint)sum;
            carry = (uint)(sum >> 32);
        }
        b[i & 7U] ^= a[(i + 3U) & 7U] + i;
    }
    uint x = 0;
    #pragma unroll
    for (uint j = 0; j < 8; ++j) x ^= hw_rotl32(a[j], j + 1U);
    out[gid] = x;
}

kernel void hw_simd_shuffle(device uint *out [[buffer(0)]],
                            constant HwParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    uint x = gid ^ p.seed;
    for (uint i = 0; i < p.iterations; ++i) {
        x ^= simd_shuffle_xor(x + i, 1);
        x ^= simd_shuffle_xor(x, 2);
        x ^= simd_shuffle_xor(x, 4);
        x ^= simd_shuffle_xor(x, 8);
        x ^= simd_shuffle_xor(x, 16);
    }
    out[gid] = x;
}

#define HW_PRIVATE_KERNEL(NAME, N) \
kernel void NAME(device uint *out [[buffer(0)]], \
                 constant HwParams& p [[buffer(2)]], \
                 uint gid [[thread_position_in_grid]]) { \
    uint a[N]; \
    for (uint j = 0; j < N; ++j) \
        a[j] = (gid + 1U) * (0x9e3779b9U ^ (j * 0x85ebca6bU)); \
    uint x = gid ^ p.seed; \
    for (uint i = 0; i < p.iterations; ++i) { \
        uint index = (x + i) & (N - 1U); \
        uint v = a[index]; \
        v = hw_rotl32(v + x + 0x7f4a7c15U, 9) ^ i; \
        a[index] = v; \
        x = hw_rotl32(x ^ v, 5); \
    } \
    for (uint j = 0; j < N; ++j) x ^= a[j]; \
    out[gid] = x; \
}

HW_PRIVATE_KERNEL(hw_private_32, 32)
HW_PRIVATE_KERNEL(hw_private_128, 128)
HW_PRIVATE_KERNEL(hw_private_512, 512)
HW_PRIVATE_KERNEL(hw_private_1024, 1024)
HW_PRIVATE_KERNEL(hw_private_2048, 2048)
#undef HW_PRIVATE_KERNEL

kernel void hw_threadgroup(device uint *out [[buffer(0)]],
                           constant HwParams& p [[buffer(2)]],
                           threadgroup uint *scratch [[threadgroup(0)]],
                           uint gid [[thread_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
    uint x = gid ^ p.seed;
    scratch[lid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < p.iterations; ++i) {
        uint other = scratch[(lid + p.stride) & p.mask];
        x = hw_rotl32(x + other + i, 7);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[gid] = x;
}

kernel void hw_mem_read64(device uint4 *out [[buffer(0)]],
                          device const uint4 *in [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    uint base = gid * 4U;
    uint4 x = in[base] ^ in[base + 1] ^ in[base + 2] ^ in[base + 3];
    out[gid] = x;
}

kernel void hw_mem_write64(device uint4 *out [[buffer(0)]],
                           constant HwParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint base = gid * 4U;
    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
        uint x = gid ^ p.seed ^ (j * 0x9e3779b9U);
        out[base + j] = uint4(x, x + 1U, x + 2U, x + 3U);
    }
}

kernel void hw_mem_copy64(device uint4 *out [[buffer(0)]],
                          device const uint4 *in [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    uint base = gid * 4U;
    #pragma unroll
    for (uint j = 0; j < 4; ++j) out[base + j] = in[base + j];
}

kernel void hw_mem_random(device uint4 *out [[buffer(0)]],
                          device const uint4 *in [[buffer(1)]],
                          constant HwParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    uint index = (gid * 0x9e3779b9U + p.seed) & p.mask;
    out[gid] = in[index];
}

/* Exact snapshot of the production field_10x26 multiplication. The source
 * originates in Bitcoin Core libsecp256k1 (MIT); see NOTICE. */
struct hw_fe10 { uint n[10]; };

static inline void hw_fe10_mul(thread hw_fe10& r, thread const hw_fe10& a,
                               thread const hw_fe10& b) {
    ulong c, d;
    ulong u0, u1, u2, u3, u4, u5, u6, u7, u8;
    uint t9, t1, t0, t2, t3, t4, t5, t6, t7;
    const uint M = 0x3FFFFFFU, R0 = 0x3D10U, R1 = 0x400U;

    d  = (ulong)a.n[0] * b.n[9] + (ulong)a.n[1] * b.n[8] + (ulong)a.n[2] * b.n[7] + (ulong)a.n[3] * b.n[6]
       + (ulong)a.n[4] * b.n[5] + (ulong)a.n[5] * b.n[4] + (ulong)a.n[6] * b.n[3] + (ulong)a.n[7] * b.n[2]
       + (ulong)a.n[8] * b.n[1] + (ulong)a.n[9] * b.n[0];
    t9 = d & M; d >>= 26;
    c  = (ulong)a.n[0] * b.n[0];
    d += (ulong)a.n[1] * b.n[9] + (ulong)a.n[2] * b.n[8] + (ulong)a.n[3] * b.n[7] + (ulong)a.n[4] * b.n[6]
       + (ulong)a.n[5] * b.n[5] + (ulong)a.n[6] * b.n[4] + (ulong)a.n[7] * b.n[3] + (ulong)a.n[8] * b.n[2]
       + (ulong)a.n[9] * b.n[1];
    u0 = d & M; d >>= 26; c += u0 * R0;
    t0 = c & M; c >>= 26; c += u0 * R1;
    c += (ulong)a.n[0] * b.n[1] + (ulong)a.n[1] * b.n[0];
    d += (ulong)a.n[2] * b.n[9] + (ulong)a.n[3] * b.n[8] + (ulong)a.n[4] * b.n[7] + (ulong)a.n[5] * b.n[6]
       + (ulong)a.n[6] * b.n[5] + (ulong)a.n[7] * b.n[4] + (ulong)a.n[8] * b.n[3] + (ulong)a.n[9] * b.n[2];
    u1 = d & M; d >>= 26; c += u1 * R0;
    t1 = c & M; c >>= 26; c += u1 * R1;
    c += (ulong)a.n[0] * b.n[2] + (ulong)a.n[1] * b.n[1] + (ulong)a.n[2] * b.n[0];
    d += (ulong)a.n[3] * b.n[9] + (ulong)a.n[4] * b.n[8] + (ulong)a.n[5] * b.n[7] + (ulong)a.n[6] * b.n[6]
       + (ulong)a.n[7] * b.n[5] + (ulong)a.n[8] * b.n[4] + (ulong)a.n[9] * b.n[3];
    u2 = d & M; d >>= 26; c += u2 * R0;
    t2 = c & M; c >>= 26; c += u2 * R1;
    c += (ulong)a.n[0] * b.n[3] + (ulong)a.n[1] * b.n[2] + (ulong)a.n[2] * b.n[1] + (ulong)a.n[3] * b.n[0];
    d += (ulong)a.n[4] * b.n[9] + (ulong)a.n[5] * b.n[8] + (ulong)a.n[6] * b.n[7] + (ulong)a.n[7] * b.n[6]
       + (ulong)a.n[8] * b.n[5] + (ulong)a.n[9] * b.n[4];
    u3 = d & M; d >>= 26; c += u3 * R0;
    t3 = c & M; c >>= 26; c += u3 * R1;
    c += (ulong)a.n[0] * b.n[4] + (ulong)a.n[1] * b.n[3] + (ulong)a.n[2] * b.n[2] + (ulong)a.n[3] * b.n[1]
       + (ulong)a.n[4] * b.n[0];
    d += (ulong)a.n[5] * b.n[9] + (ulong)a.n[6] * b.n[8] + (ulong)a.n[7] * b.n[7] + (ulong)a.n[8] * b.n[6]
       + (ulong)a.n[9] * b.n[5];
    u4 = d & M; d >>= 26; c += u4 * R0;
    t4 = c & M; c >>= 26; c += u4 * R1;
    c += (ulong)a.n[0] * b.n[5] + (ulong)a.n[1] * b.n[4] + (ulong)a.n[2] * b.n[3] + (ulong)a.n[3] * b.n[2]
       + (ulong)a.n[4] * b.n[1] + (ulong)a.n[5] * b.n[0];
    d += (ulong)a.n[6] * b.n[9] + (ulong)a.n[7] * b.n[8] + (ulong)a.n[8] * b.n[7] + (ulong)a.n[9] * b.n[6];
    u5 = d & M; d >>= 26; c += u5 * R0;
    t5 = c & M; c >>= 26; c += u5 * R1;
    c += (ulong)a.n[0] * b.n[6] + (ulong)a.n[1] * b.n[5] + (ulong)a.n[2] * b.n[4] + (ulong)a.n[3] * b.n[3]
       + (ulong)a.n[4] * b.n[2] + (ulong)a.n[5] * b.n[1] + (ulong)a.n[6] * b.n[0];
    d += (ulong)a.n[7] * b.n[9] + (ulong)a.n[8] * b.n[8] + (ulong)a.n[9] * b.n[7];
    u6 = d & M; d >>= 26; c += u6 * R0;
    t6 = c & M; c >>= 26; c += u6 * R1;
    c += (ulong)a.n[0] * b.n[7] + (ulong)a.n[1] * b.n[6] + (ulong)a.n[2] * b.n[5] + (ulong)a.n[3] * b.n[4]
       + (ulong)a.n[4] * b.n[3] + (ulong)a.n[5] * b.n[2] + (ulong)a.n[6] * b.n[1] + (ulong)a.n[7] * b.n[0];
    d += (ulong)a.n[8] * b.n[9] + (ulong)a.n[9] * b.n[8];
    u7 = d & M; d >>= 26; c += u7 * R0;
    t7 = c & M; c >>= 26; c += u7 * R1;
    c += (ulong)a.n[0] * b.n[8] + (ulong)a.n[1] * b.n[7] + (ulong)a.n[2] * b.n[6] + (ulong)a.n[3] * b.n[5]
       + (ulong)a.n[4] * b.n[4] + (ulong)a.n[5] * b.n[3] + (ulong)a.n[6] * b.n[2] + (ulong)a.n[7] * b.n[1]
       + (ulong)a.n[8] * b.n[0];
    d += (ulong)a.n[9] * b.n[9];
    u8 = d & M; d >>= 26; c += u8 * R0;
    r.n[3] = t3; r.n[4] = t4; r.n[5] = t5; r.n[6] = t6; r.n[7] = t7;
    r.n[8] = c & M; c >>= 26; c += u8 * R1;
    c += d * R0 + t9;
    r.n[9] = c & (M >> 4); c >>= 22; c += d * (R1 << 4);
    d = c * (R0 >> 4) + t0;
    r.n[0] = d & M; d >>= 26;
    d += c * (R1 >> 4) + t1;
    r.n[1] = d & M; d >>= 26;
    d += t2;
    r.n[2] = d;
}

kernel void hw_fe10x26(device uint *out [[buffer(0)]],
                       constant HwParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    hw_fe10 a, b, r;
    for (uint j = 0; j < 10; ++j) {
        a.n[j] = ((gid + 1U) * (0x1f123bb5U + j * 0x1021U)) & 0x3ffffffU;
        b.n[j] = (0x02c1b3c7U + j * 0x12345U) & 0x3ffffffU;
    }
    a.n[9] &= 0x03fffffU; b.n[9] &= 0x03fffffU;
    for (uint i = 0; i < p.iterations; ++i) {
        hw_fe10_mul(r, a, b);
        a = r;
    }
    uint base = gid * 10U;
    for (uint j = 0; j < 10; ++j) out[base + j] = a.n[j];
}

/* The comparison implementation is the 8x32 field representation used by
 * mrtozner/tron-vanity-metal (MIT), adapted only with hw_ name prefixes. */
struct hw_u256 { uint d[8]; };
constant hw_u256 HW_P = {{0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
                          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}};

static inline uint hw_add8(thread hw_u256& r, thread const hw_u256& a,
                           thread const hw_u256& b) {
    uint carry = 0;
    for (int i = 0; i < 8; ++i) {
        ulong sum = (ulong)a.d[i] + b.d[i] + carry;
        r.d[i] = (uint)sum; carry = (uint)(sum >> 32);
    }
    return carry;
}
static inline uint hw_sub8p(thread hw_u256& r, thread const hw_u256& a) {
    uint borrow = 0;
    for (int i = 0; i < 8; ++i) {
        long diff = (long)a.d[i] - HW_P.d[i] - borrow;
        r.d[i] = (uint)diff; borrow = diff < 0;
    }
    return borrow;
}
static inline bool hw_gte8p(thread const hw_u256& a) {
    for (int i = 7; i >= 0; --i) {
        if (a.d[i] > HW_P.d[i]) return true;
        if (a.d[i] < HW_P.d[i]) return false;
    }
    return true;
}
static inline void hw_fe8_mul(thread hw_u256& r, thread const hw_u256& a,
                              thread const hw_u256& b) {
    uint c[16] = {0};
    for (int i = 0; i < 8; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 8; ++j) {
            ulong prod = (ulong)a.d[i] * b.d[j] + c[i + j] + carry;
            c[i + j] = (uint)prod; carry = prod >> 32;
        }
        c[i + 8] = (uint)carry;
    }
    hw_u256 upper, lower, times977;
    for (int i = 0; i < 8; ++i) { upper.d[i] = c[i + 8]; lower.d[i] = c[i]; times977.d[i] = 0; }
    ulong carry977 = 0;
    for (int i = 0; i < 8; ++i) {
        ulong prod = (ulong)upper.d[i] * 977UL + carry977;
        times977.d[i] = (uint)prod; carry977 = prod >> 32;
    }
    uint carryMain = hw_add8(r, lower, times977);
    ulong shiftCarry = 0;
    for (int i = 1; i < 8; ++i) {
        ulong sum = (ulong)r.d[i] + upper.d[i - 1] + shiftCarry;
        r.d[i] = (uint)sum; shiftCarry = sum >> 32;
    }
    uint overflow = (uint)carry977 + carryMain + (uint)shiftCarry + upper.d[7];
    ulong sum = (ulong)r.d[0] + (ulong)overflow * 977UL;
    r.d[0] = (uint)sum; ulong carry = sum >> 32;
    for (int i = 1; i < 8; ++i) {
        sum = (ulong)r.d[i] + carry; r.d[i] = (uint)sum; carry = sum >> 32;
    }
    sum = (ulong)r.d[1] + overflow; r.d[1] = (uint)sum; carry = sum >> 32;
    for (int i = 2; i < 8; ++i) {
        sum = (ulong)r.d[i] + carry; r.d[i] = (uint)sum; carry = sum >> 32;
    }
    if (hw_gte8p(r)) { hw_u256 t; hw_sub8p(t, r); r = t; }
    if (hw_gte8p(r)) { hw_u256 t; hw_sub8p(t, r); r = t; }
}

kernel void hw_fe8x32(device uint *out [[buffer(0)]],
                      constant HwParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    hw_u256 a, b, r;
    for (uint j = 0; j < 8; ++j) {
        a.d[j] = (gid + 1U) * (0x1f123bb5U + j * 0x1021U);
        b.d[j] = 0x02c1b3c7U + j * 0x12345U;
    }
    a.d[7] &= 0x7fffffffU; b.d[7] &= 0x7fffffffU;
    for (uint i = 0; i < p.iterations; ++i) {
        hw_fe8_mul(r, a, b);
        a = r;
    }
    uint base = gid * 8U;
    for (uint j = 0; j < 8; ++j) out[base + j] = a.d[j];
}
