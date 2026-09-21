#include <metal_stdlib>
using namespace metal;

#define METAL_BACKEND 1


#ifndef KPI
#define KPI 8            /* 每个 work-item 连续处理的私钥数（host 用 -D KPI=n 覆盖） */
#endif
#ifndef ECW
#define ECW 1
#endif
#ifndef ECBITS
#define ECBITS 20
#endif
#define ECW_DIGITS (1 << ECW)
#define ECW_WINDOWS ((ECBITS + ECW - 1) / ECW)

typedef struct { uint n[10]; } fe;
typedef struct { fe x, y; int inf; } ge;
typedef struct { fe x, y, z; int inf; } gej;

inline void fe_set_int(thread fe *r, uint v);
inline void ge_load_g(thread ge *p, device const uchar *xy);
inline void gej_add_ge(thread gej *r, thread const gej *a, thread const ge *b);
inline void gej_to_pub(thread uchar *out, thread gej *a);
inline void keccak256_64(thread uchar *out, thread const uchar *in);
inline void sha256_short(thread uchar *out, thread const uchar *msg, int len);

#ifdef RESIDENT
/* ---------------- GPU-resident generator ----------------
 * The resident path uses a device-side CSPRNG and returns only complete
 * matches through a fixed-size device ring. It intentionally uses bounded
 * launches: Windows WDDM must be able to preempt the queue.
 */
#define RESIDENT_RECORD_BYTES 160
#define RESIDENT_KEY_BYTES 32
#define RESIDENT_ADDR_BYTES 34
#define RESIDENT_MAX_MATCHES 16
#ifndef RESIDENT_RNG
#define RESIDENT_RNG 1
#endif

#ifdef METAL_BACKEND
static inline uint metal_atomic_add(volatile device uint *p, uint v) {
    return atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
#define resident_atomic_add metal_atomic_add
#else

#endif

static inline uint resident_load32(const device uchar *p) {
    return ((uint)p[0]) | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

static inline uint resident_rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

static inline void resident_qr(thread uint *a, thread uint *b, thread uint *c, thread uint *d) {
    *a += *b; *d ^= *a; *d = resident_rotl(*d, 16);
    *c += *d; *b ^= *c; *b = resident_rotl(*b, 12);
    *a += *b; *d ^= *a; *d = resident_rotl(*d, 8);
    *c += *d; *b ^= *c; *b = resident_rotl(*b, 7);
}

static inline void resident_chacha12(device const uchar *seed, ulong counter,
                                     uint domain, thread thread uchar *out) {
    uint x[16], orig[16];
    x[0] = 0x61707865U; x[1] = 0x3320646eU; x[2] = 0x79622d32U; x[3] = 0x6b206574U;
    for (int i = 0; i < 8; ++i) x[4 + i] = resident_load32(&seed[i * 4]);
    x[12] = (uint)counter; x[13] = domain ^ (uint)(counter >> 32); x[14] = resident_load32(&seed[16]) ^ 0x9e3779b9U;
    x[15] = resident_load32(&seed[20]) ^ 0x243f6a88U;
    for (int i = 0; i < 16; ++i) orig[i] = x[i];
    for (int r = 0; r < 6; ++r) {
        resident_qr(&x[0], &x[4], &x[8], &x[12]);
        resident_qr(&x[1], &x[5], &x[9], &x[13]);
        resident_qr(&x[2], &x[6], &x[10], &x[14]);
        resident_qr(&x[3], &x[7], &x[11], &x[15]);
        resident_qr(&x[0], &x[5], &x[10], &x[15]);
        resident_qr(&x[1], &x[6], &x[11], &x[12]);
        resident_qr(&x[2], &x[7], &x[8], &x[13]);
        resident_qr(&x[3], &x[4], &x[9], &x[14]);
    }
    for (int i = 0; i < 16; ++i) {
        uint v = x[i] + orig[i];
        out[i * 4 + 0] = (uchar)v; out[i * 4 + 1] = (uchar)(v >> 8);
        out[i * 4 + 2] = (uchar)(v >> 16); out[i * 4 + 3] = (uchar)(v >> 24);
    }
}

static inline void resident_philox4(device const uchar *seed, ulong counter,
                                    uint domain, thread thread uchar *out) {
    uint c0 = (uint)counter, c1 = (uint)(counter >> 32), c2 = domain, c3 = 0;
    uint k0 = resident_load32(&seed[0]), k1 = resident_load32(&seed[4]);
    for (int round = 0; round < 10; ++round) {
        uint hi0 = ((uint)(((ulong)0xD2511F53U * (ulong)c0) >> 32)), lo0 = 0xD2511F53U * c0;
        uint hi1 = ((uint)(((ulong)0xCD9E8D57U * (ulong)c2) >> 32)), lo1 = 0xCD9E8D57U * c2;
        uint n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
    }
    uint v[4] = {c0, c1, c2, c3};
    for (int i = 0; i < 4; ++i) {
        out[i * 4 + 0] = (uchar)v[i]; out[i * 4 + 1] = (uchar)(v[i] >> 8);
        out[i * 4 + 2] = (uchar)(v[i] >> 16); out[i * 4 + 3] = (uchar)(v[i] >> 24);
    }
}

constant uchar resident_aes_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static inline uchar resident_aes_xtime(uchar x) { return (uchar)((x << 1) ^ ((x >> 7) * 0x1b)); }

static inline void resident_aes128(device const uchar *seed, ulong counter,
                                   uint domain, thread thread uchar *out) {
    uchar rk[176], s[16];
    for (int i = 0; i < 16; ++i) rk[i] = seed[i];
    uchar rc = 1;
    int bytes = 16;
    while (bytes < 176) {
        uchar t0 = rk[bytes - 4], t1 = rk[bytes - 3];
        uchar t2 = rk[bytes - 2], t3 = rk[bytes - 1];
        if ((bytes & 15) == 0) {
            uchar q0 = resident_aes_sbox[t1];
            uchar q1 = resident_aes_sbox[t2];
            uchar q2 = resident_aes_sbox[t3];
            uchar q3 = resident_aes_sbox[t0];
            t0 = q0 ^ rc; t1 = q1; t2 = q2; t3 = q3;
            rc = resident_aes_xtime(rc);
        }
        for (int j = 0; j < 4; ++j) {
            uchar v = rk[bytes - 16] ^ (j == 0 ? t0 : j == 1 ? t1 : j == 2 ? t2 : t3);
            rk[bytes++] = v;
        }
    }
    for (int i = 0; i < 16; ++i) s[i] = seed[16 + (i & 7)] ^ (uchar)(i < 8 ? (counter >> (i * 8)) : (domain >> ((i - 8) * 4)));
    for (int i = 0; i < 16; ++i) s[i] ^= rk[i];
    for (int round = 1; round <= 10; ++round) {
        uchar t[16];
        for (int i = 0; i < 16; ++i) t[i] = resident_aes_sbox[s[i]];
        s[0]=t[0]; s[1]=t[5]; s[2]=t[10]; s[3]=t[15];
        s[4]=t[4]; s[5]=t[9]; s[6]=t[14]; s[7]=t[3];
        s[8]=t[8]; s[9]=t[13]; s[10]=t[2]; s[11]=t[7];
        s[12]=t[12]; s[13]=t[1]; s[14]=t[6]; s[15]=t[11];
        if (round != 10) for (int c = 0; c < 4; ++c) {
            uchar a=s[c*4], b=s[c*4+1], d=s[c*4+2], e=s[c*4+3], q=a^b^d^e;
            s[c*4]=a^q^resident_aes_xtime(a^b);
            s[c*4+1]=b^q^resident_aes_xtime(b^d);
            s[c*4+2]=d^q^resident_aes_xtime(d^e);
            s[c*4+3]=e^q^resident_aes_xtime(e^a);
        }
        for (int i = 0; i < 16; ++i) s[i] ^= rk[round * 16 + i];
    }
    for (int i = 0; i < 16; ++i) out[i] = s[i];
}

static inline void resident_rng32(device const uchar *seed, ulong counter,
                                  thread thread uchar *out) {
#if RESIDENT_RNG == 2
    resident_philox4(seed, counter, 0x47505552U, out);
    resident_philox4(seed, counter, 0x47505553U, &out[16]);
#elif RESIDENT_RNG == 3
    resident_aes128(seed, counter, 0x47505552U, out);
    resident_aes128(seed, counter + 1, 0x47505553U, &out[16]);
#else
    uchar block[64];
    resident_chacha12(seed, counter, 0x47505552U, block);
    for (int i = 0; i < 32; ++i) out[i] = block[i];
#endif
}

static inline int resident_lt_order(thread const uchar *sk) {
    /* secp256k1 order, big endian. */
    const uchar n[32] = {
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
        0xbc,0xe6,0xfa,0xda,0x71,0x48,0x9d,0x7e,
        0xa3,0x0b,0x8c,0xd0,0x36,0x41,0x41,0x21
    };
    int nonzero = 0, cmp = 0;
    for (int i = 0; i < 32; ++i) {
        nonzero |= sk[i] != 0;
        if (cmp == 0 && sk[i] != n[i]) cmp = sk[i] < n[i] ? -1 : 1;
    }
    return nonzero && cmp < 0;
}

static inline uint resident_scalar_bit(thread const uchar *sk, int bit) {
    return (sk[31 - bit / 8] >> (bit & 7)) & 1U;
}

static inline void resident_ec_mul(thread gej *acc, thread const uchar *sk,
                                   device const uchar *table_b32) {
    acc->x.n[0] = 0; acc->x.n[1] = 0; acc->x.n[2] = 0; acc->x.n[3] = 0;
    acc->x.n[4] = 0; acc->x.n[5] = 0; acc->x.n[6] = 0; acc->x.n[7] = 0;
    acc->x.n[8] = 0; acc->x.n[9] = 0;
    acc->y = acc->x; fe_set_int(&acc->z, 1); acc->inf = 1;
    #if ECW == 1
    for (int bit = 0; bit < 256; ++bit) {
        if (resident_scalar_bit(sk, bit)) {
            ge g; ge_load_g(&g, &table_b32[bit * 64]);
            gej_add_ge(acc, acc, &g);
        }
    }
    #else
    for (int w = 0; w < ECW_WINDOWS; ++w) {
        uint d = 0;
        for (int b = 0; b < ECW; ++b)
            d |= resident_scalar_bit(sk, w * ECW + b) << b;
        if (d) {
            ge g;
            ge_load_g(&g, &table_b32[(w * ECW_DIGITS + d) * 64]);
            gej_add_ge(acc, acc, &g);
        }
    }
    #endif
}

static inline void resident_u32(device uchar *p, uint v) {
    p[0] = (uchar)v; p[1] = (uchar)(v >> 8); p[2] = (uchar)(v >> 16); p[3] = (uchar)(v >> 24);
}

static inline void resident_u64(device uchar *p, ulong v) {
    for (int i = 0; i < 8; ++i) p[i] = (uchar)(v >> (i * 8));
}

constant uchar resident_b58_map[128] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 255, 255, 255, 255, 255,
    255, 9, 10, 11, 12, 13, 14, 15, 16, 255, 17, 18, 19, 20, 21, 255,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 255, 255, 255, 255, 255,
    255, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 255, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 255, 255, 255, 255, 255,
};

static inline uint resident_b58_index(uchar c) {
    return c < 128 ? resident_b58_map[c] : 255;
}

static inline void resident_base58_address(thread uchar *addr, thread const uchar *full25) {
    const char b58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    // Big-endian base-65536 limbs: 13 limbs instead of 25 byte divisions.
    // The first limb is one byte because the payload is 25 bytes long.
    ushort num[13];
    num[0] = full25[0];
    for (int i = 1; i < 13; ++i)
        num[i] = ((ushort)full25[1 + (i - 1) * 2] << 8) | full25[2 + (i - 1) * 2];
    for (int i = 0; i < 34; ++i) addr[i] = '1';
    int start = 0;
    for (int it = 0; it < 34; ++it) {
        uint rem = 0;
        for (int i = start; i < 13; ++i) {
            uint acc = (rem << 16) | num[i];
            num[i] = (ushort)(acc / 58);
            rem = acc % 58;
        }
        addr[33 - it] = (uchar)b58[rem];
        while (start < 13 && num[start] == 0) ++start;
    }
}

static inline void resident_emit(thread const uchar *sk, thread const uchar *pub,
                                 device const uint *dfa,
                                 device const uint *out_start,
                                 device const uint *out_len,
                                 device const uint *out_ids,
                                 volatile device uint *meta,
                                 device uchar *records, constant uint& cap, ulong seq) {
    uchar h[32], payload[21], d1[32], d2[32], full[25], addr[34];
    keccak256_64(h, pub);
    payload[0] = 0x41;
    for (int i = 0; i < 20; ++i) payload[1 + i] = h[12 + i];
    sha256_short(d1, payload, 21); sha256_short(d2, d1, 32);
    for (int i = 0; i < 21; ++i) full[i] = payload[i];
    for (int i = 0; i < 4; ++i) full[21 + i] = d2[i];
    resident_base58_address(addr, full);

    uint ids[RESIDENT_MAX_MATCHES];
    uint count = 0, state = 0, flags = 0;
    for (int i = 1; i < 34; ++i) {
        uchar c = addr[i];
        uint ai = resident_b58_index(c);
        if (ai >= 58) { state = 0; continue; }
        state = dfa[state * 58 + ai];
        uint start = out_start[state], len = out_len[state];
        for (uint j = 0; j < len; ++j) {
            uint id = out_ids[start + j], seen = 0;
            for (uint k = 0; k < count; ++k) if (ids[k] == id) seen = 1;
            if (!seen) {
                if (count < RESIDENT_MAX_MATCHES) ids[count++] = id;
                else flags |= 1U;
            }
        }
    }
    if (!count) return;

    uint pos = resident_atomic_add((volatile device uint*)&meta[0], 1U);
    uint read_pos = meta[1];
    if (pos - read_pos >= cap) {
        resident_atomic_add((volatile device uint*)&meta[2], 1U);
        /* Metal's OpenCL-compatible front-end does not expose atomic_or in
         * the final AIR linker; the flag is boolean, so an atomic increment
         * preserves the protocol invariant (non-zero means error). */
        resident_atomic_add((volatile device uint*)&meta[3], 1U);
        return;
    }
    device uchar *dst = &records[(pos % cap) * RESIDENT_RECORD_BYTES];
    for (int i = 0; i < 32; ++i) dst[i] = sk[i];
    for (int i = 0; i < RESIDENT_ADDR_BYTES; ++i) dst[32 + i] = addr[i];
    resident_u32(&dst[68], count);
    for (uint i = 0; i < RESIDENT_MAX_MATCHES; ++i) resident_u32(&dst[72 + i * 4], i < count ? ids[i] : 0);
    resident_u32(&dst[136], flags);
    resident_u64(&dst[140], seq);
}

kernel void tron_vanity_resident(
        device const uchar *seed,
        device const uchar *table_b32,
        device const uint *dfa,
        device const uint *out_start,
        device const uint *out_len,
        device const uint *out_ids,
        volatile device uint *meta,
        device uchar *records,
        constant uint& cap,
        constant ulong& stream_base,
        uint3 tid [[thread_position_in_grid]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint3 tlid [[thread_position_in_threadgroup]]) {
    uint lid = tlid.x;
    uint group = tgid.x;
    threadgroup uchar base_sk[32];
    threadgroup gej base_acc;
    if (lid == 0) {
        uchar base_private[32];
        resident_rng32(seed, stream_base + group, base_private);
        for (int i = 0; i < 32; ++i) base_sk[i] = base_private[i];
        if (resident_lt_order(base_private)) {
            gej base_tmp;
            resident_ec_mul(&base_tmp, base_private, table_b32);
            base_acc = base_tmp;
        }
        else {
            for (int i = 0; i < 32; ++i) base_sk[i] = 0;
            base_acc.inf = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = 0; item < KPI; ++item) {
        uint offset = lid * KPI + item;
        // Reserve the maximum supported 64-lane threadgroup span so group
        // sequence ranges never overlap even on wider Apple GPU groups.
        ulong seq = (stream_base + group) * (uint)(KPI * 128) + offset;
        uchar sk[32], pub[64];
        for (int i = 0; i < 32; ++i) sk[i] = base_sk[i];
        uint carry = offset;
        for (int i = 31; i >= 0 && carry; --i) {
            uint sum = sk[i] + (carry & 255U);
            sk[i] = (uchar)sum;
            carry = (carry >> 8) + (sum >> 8);
        }
        resident_atomic_add((volatile device uint*)&meta[4], 1U);
        if (base_acc.inf) continue;
        if (!resident_lt_order(sk)) continue;
        gej acc = base_acc;
        if (offset) {
            ge g;
            ge_load_g(&g, &table_b32[offset * 64]);
            gej_add_ge(&acc, &acc, &g);
        }
        gej_to_pub(pub, &acc);
        resident_emit(sk, pub, dfa, out_start, out_len, out_ids, meta, records, cap, seq);
    }
}
#endif

/* TRON 靓号 OpenCL 批处理内核
 * secp256k1 域/群运算移植自 bitcoin-core/libsecp256k1 (field_10x26 / group_impl，MIT)。
 * keccak-256 / sha-256 与 CPU 侧 src/ 实现同参数。
 *
 * 每个 work-item：
 *   s   = batch_offset + gid
 *   Q   = P0 + s*G           （P0 = k0*G 由 host 上传；用 table[j]=2^j*G 做 <=32 次点加）
 *   pub = X(Q)||Y(Q)  (BE)
 *   h   = keccak256(pub);  payload = 0x41 || h[12:32]
 *   full= payload || sha256d(payload)[0:4]              (25 字节)
 *   将完整 TRON Base58 地址送入扁平 Aho-Corasick DFA；命中则回传 (s, word_id)
 */

/* ---------------- 域运算 ---------------- */

inline void fe_mul_inner(thread uint *r, thread const uint *a, thread const uint *b) {
    ulong c, d;
    ulong u0, u1, u2, u3, u4, u5, u6, u7, u8;
    uint t9, t1, t0, t2, t3, t4, t5, t6, t7;
    const uint M = 0x3FFFFFFU, R0 = 0x3D10U, R1 = 0x400U;

    d  = (ulong)a[0] * b[9] + (ulong)a[1] * b[8] + (ulong)a[2] * b[7] + (ulong)a[3] * b[6]
       + (ulong)a[4] * b[5] + (ulong)a[5] * b[4] + (ulong)a[6] * b[3] + (ulong)a[7] * b[2]
       + (ulong)a[8] * b[1] + (ulong)a[9] * b[0];
    t9 = d & M; d >>= 26;

    c  = (ulong)a[0] * b[0];
    d += (ulong)a[1] * b[9] + (ulong)a[2] * b[8] + (ulong)a[3] * b[7] + (ulong)a[4] * b[6]
       + (ulong)a[5] * b[5] + (ulong)a[6] * b[4] + (ulong)a[7] * b[3] + (ulong)a[8] * b[2]
       + (ulong)a[9] * b[1];
    u0 = d & M; d >>= 26; c += u0 * R0;
    t0 = c & M; c >>= 26; c += u0 * R1;

    c += (ulong)a[0] * b[1] + (ulong)a[1] * b[0];
    d += (ulong)a[2] * b[9] + (ulong)a[3] * b[8] + (ulong)a[4] * b[7] + (ulong)a[5] * b[6]
       + (ulong)a[6] * b[5] + (ulong)a[7] * b[4] + (ulong)a[8] * b[3] + (ulong)a[9] * b[2];
    u1 = d & M; d >>= 26; c += u1 * R0;
    t1 = c & M; c >>= 26; c += u1 * R1;

    c += (ulong)a[0] * b[2] + (ulong)a[1] * b[1] + (ulong)a[2] * b[0];
    d += (ulong)a[3] * b[9] + (ulong)a[4] * b[8] + (ulong)a[5] * b[7] + (ulong)a[6] * b[6]
       + (ulong)a[7] * b[5] + (ulong)a[8] * b[4] + (ulong)a[9] * b[3];
    u2 = d & M; d >>= 26; c += u2 * R0;
    t2 = c & M; c >>= 26; c += u2 * R1;

    c += (ulong)a[0] * b[3] + (ulong)a[1] * b[2] + (ulong)a[2] * b[1] + (ulong)a[3] * b[0];
    d += (ulong)a[4] * b[9] + (ulong)a[5] * b[8] + (ulong)a[6] * b[7] + (ulong)a[7] * b[6]
       + (ulong)a[8] * b[5] + (ulong)a[9] * b[4];
    u3 = d & M; d >>= 26; c += u3 * R0;
    t3 = c & M; c >>= 26; c += u3 * R1;

    c += (ulong)a[0] * b[4] + (ulong)a[1] * b[3] + (ulong)a[2] * b[2] + (ulong)a[3] * b[1]
       + (ulong)a[4] * b[0];
    d += (ulong)a[5] * b[9] + (ulong)a[6] * b[8] + (ulong)a[7] * b[7] + (ulong)a[8] * b[6]
       + (ulong)a[9] * b[5];
    u4 = d & M; d >>= 26; c += u4 * R0;
    t4 = c & M; c >>= 26; c += u4 * R1;

    c += (ulong)a[0] * b[5] + (ulong)a[1] * b[4] + (ulong)a[2] * b[3] + (ulong)a[3] * b[2]
       + (ulong)a[4] * b[1] + (ulong)a[5] * b[0];
    d += (ulong)a[6] * b[9] + (ulong)a[7] * b[8] + (ulong)a[8] * b[7] + (ulong)a[9] * b[6];
    u5 = d & M; d >>= 26; c += u5 * R0;
    t5 = c & M; c >>= 26; c += u5 * R1;

    c += (ulong)a[0] * b[6] + (ulong)a[1] * b[5] + (ulong)a[2] * b[4] + (ulong)a[3] * b[3]
       + (ulong)a[4] * b[2] + (ulong)a[5] * b[1] + (ulong)a[6] * b[0];
    d += (ulong)a[7] * b[9] + (ulong)a[8] * b[8] + (ulong)a[9] * b[7];
    u6 = d & M; d >>= 26; c += u6 * R0;
    t6 = c & M; c >>= 26; c += u6 * R1;

    c += (ulong)a[0] * b[7] + (ulong)a[1] * b[6] + (ulong)a[2] * b[5] + (ulong)a[3] * b[4]
       + (ulong)a[4] * b[3] + (ulong)a[5] * b[2] + (ulong)a[6] * b[1] + (ulong)a[7] * b[0];
    d += (ulong)a[8] * b[9] + (ulong)a[9] * b[8];
    u7 = d & M; d >>= 26; c += u7 * R0;
    t7 = c & M; c >>= 26; c += u7 * R1;

    c += (ulong)a[0] * b[8] + (ulong)a[1] * b[7] + (ulong)a[2] * b[6] + (ulong)a[3] * b[5]
       + (ulong)a[4] * b[4] + (ulong)a[5] * b[3] + (ulong)a[6] * b[2] + (ulong)a[7] * b[1]
       + (ulong)a[8] * b[0];
    d += (ulong)a[9] * b[9];
    u8 = d & M; d >>= 26; c += u8 * R0;

    r[3] = t3; r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;

    r[8] = c & M; c >>= 26; c += u8 * R1;
    c   += d * R0 + t9;
    r[9] = c & (M >> 4); c >>= 22; c += d * (R1 << 4);

    d    = c * (R0 >> 4) + t0;
    r[0] = d & M; d >>= 26;
    d   += c * (R1 >> 4) + t1;
    r[1] = d & M; d >>= 26;
    d   += t2;
    r[2] = d;
}

inline void fe_sqr_inner(thread uint *r, thread const uint *a) {
    ulong c, d;
    ulong u0, u1, u2, u3, u4, u5, u6, u7, u8;
    uint t9, t0, t1, t2, t3, t4, t5, t6, t7;
    const uint M = 0x3FFFFFFU, R0 = 0x3D10U, R1 = 0x400U;

    d  = (ulong)(a[0]*2) * a[9] + (ulong)(a[1]*2) * a[8] + (ulong)(a[2]*2) * a[7]
       + (ulong)(a[3]*2) * a[6] + (ulong)(a[4]*2) * a[5];
    t9 = d & M; d >>= 26;

    c  = (ulong)a[0] * a[0];
    d += (ulong)(a[1]*2) * a[9] + (ulong)(a[2]*2) * a[8] + (ulong)(a[3]*2) * a[7]
       + (ulong)(a[4]*2) * a[6] + (ulong)a[5] * a[5];
    u0 = d & M; d >>= 26; c += u0 * R0;
    t0 = c & M; c >>= 26; c += u0 * R1;

    c += (ulong)(a[0]*2) * a[1];
    d += (ulong)(a[2]*2) * a[9] + (ulong)(a[3]*2) * a[8] + (ulong)(a[4]*2) * a[7]
       + (ulong)(a[5]*2) * a[6];
    u1 = d & M; d >>= 26; c += u1 * R0;
    t1 = c & M; c >>= 26; c += u1 * R1;

    c += (ulong)(a[0]*2) * a[2] + (ulong)a[1] * a[1];
    d += (ulong)(a[3]*2) * a[9] + (ulong)(a[4]*2) * a[8] + (ulong)(a[5]*2) * a[7]
       + (ulong)a[6] * a[6];
    u2 = d & M; d >>= 26; c += u2 * R0;
    t2 = c & M; c >>= 26; c += u2 * R1;

    c += (ulong)(a[0]*2) * a[3] + (ulong)(a[1]*2) * a[2];
    d += (ulong)(a[4]*2) * a[9] + (ulong)(a[5]*2) * a[8] + (ulong)(a[6]*2) * a[7];
    u3 = d & M; d >>= 26; c += u3 * R0;
    t3 = c & M; c >>= 26; c += u3 * R1;

    c += (ulong)(a[0]*2) * a[4] + (ulong)(a[1]*2) * a[3] + (ulong)a[2] * a[2];
    d += (ulong)(a[5]*2) * a[9] + (ulong)(a[6]*2) * a[8] + (ulong)a[7] * a[7];
    u4 = d & M; d >>= 26; c += u4 * R0;
    t4 = c & M; c >>= 26; c += u4 * R1;

    c += (ulong)(a[0]*2) * a[5] + (ulong)(a[1]*2) * a[4] + (ulong)(a[2]*2) * a[3];
    d += (ulong)(a[6]*2) * a[9] + (ulong)(a[7]*2) * a[8];
    u5 = d & M; d >>= 26; c += u5 * R0;
    t5 = c & M; c >>= 26; c += u5 * R1;

    c += (ulong)(a[0]*2) * a[6] + (ulong)(a[1]*2) * a[5] + (ulong)(a[2]*2) * a[4]
       + (ulong)a[3] * a[3];
    d += (ulong)(a[7]*2) * a[9] + (ulong)a[8] * a[8];
    u6 = d & M; d >>= 26; c += u6 * R0;
    t6 = c & M; c >>= 26; c += u6 * R1;

    c += (ulong)(a[0]*2) * a[7] + (ulong)(a[1]*2) * a[6] + (ulong)(a[2]*2) * a[5]
       + (ulong)(a[3]*2) * a[4];
    d += (ulong)(a[8]*2) * a[9];
    u7 = d & M; d >>= 26; c += u7 * R0;
    t7 = c & M; c >>= 26; c += u7 * R1;

    c += (ulong)(a[0]*2) * a[8] + (ulong)(a[1]*2) * a[7] + (ulong)(a[2]*2) * a[6]
       + (ulong)(a[3]*2) * a[5] + (ulong)a[4] * a[4];
    d += (ulong)a[9] * a[9];
    u8 = d & M; d >>= 26; c += u8 * R0;

    r[3] = t3; r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;

    r[8] = c & M; c >>= 26; c += u8 * R1;
    c   += d * R0 + t9;
    r[9] = c & (M >> 4); c >>= 22; c += d * (R1 << 4);

    d    = c * (R0 >> 4) + t0;
    r[0] = d & M; d >>= 26;
    d   += c * (R1 >> 4) + t1;
    r[1] = d & M; d >>= 26;
    d   += t2;
    r[2] = d;
}

inline void fe_mul(thread fe *r, thread const fe *a, thread const fe *b) { fe_mul_inner(r->n, a->n, b->n); }
inline void fe_sqr(thread fe *r, thread const fe *a) { fe_sqr_inner(r->n, a->n); }

inline void fe_set_int(thread fe *r, uint v) {
    r->n[0] = v;
    for (int i = 1; i < 10; i++) r->n[i] = 0;
}

inline void fe_add(thread fe *r, thread const fe *a) {
    for (int i = 0; i < 10; i++) r->n[i] += a->n[i];
}

inline void fe_add_int(thread fe *r, uint a) { r->n[0] += a; }

inline void fe_mul_int(thread fe *r, uint a) {
    for (int i = 0; i < 10; i++) r->n[i] *= a;
}

inline void fe_negate(thread fe *r, thread const fe *a, int m) {
    r->n[0] = 0x3FFFC2FU * 2 * (m + 1) - a->n[0];
    r->n[1] = 0x3FFFFBFU * 2 * (m + 1) - a->n[1];
    r->n[2] = 0x3FFFFFFU * 2 * (m + 1) - a->n[2];
    r->n[3] = 0x3FFFFFFU * 2 * (m + 1) - a->n[3];
    r->n[4] = 0x3FFFFFFU * 2 * (m + 1) - a->n[4];
    r->n[5] = 0x3FFFFFFU * 2 * (m + 1) - a->n[5];
    r->n[6] = 0x3FFFFFFU * 2 * (m + 1) - a->n[6];
    r->n[7] = 0x3FFFFFFU * 2 * (m + 1) - a->n[7];
    r->n[8] = 0x3FFFFFFU * 2 * (m + 1) - a->n[8];
    r->n[9] = 0x03FFFFFU * 2 * (m + 1) - a->n[9];
}

inline void fe_cmov(thread fe *r, thread const fe *a, int flag) {
    uint mask0 = (uint)flag + ~((uint)0);
    uint mask1 = ~mask0;
    for (int i = 0; i < 10; i++) r->n[i] = (r->n[i] & mask0) | (a->n[i] & mask1);
}

inline void fe_half(thread fe *r) {
    uint t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4],
         t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    uint one = 1U;
    uint mask = (uint)(-(int)(t0 & one)) >> 6;
    t0 += 0x3FFFC2FU & mask; t1 += 0x3FFFFBFU & mask;
    t2 += mask; t3 += mask; t4 += mask; t5 += mask; t6 += mask; t7 += mask; t8 += mask;
    t9 += mask >> 4;
    r->n[0] = (t0 >> 1) + ((t1 & one) << 25);
    r->n[1] = (t1 >> 1) + ((t2 & one) << 25);
    r->n[2] = (t2 >> 1) + ((t3 & one) << 25);
    r->n[3] = (t3 >> 1) + ((t4 & one) << 25);
    r->n[4] = (t4 >> 1) + ((t5 & one) << 25);
    r->n[5] = (t5 >> 1) + ((t6 & one) << 25);
    r->n[6] = (t6 >> 1) + ((t7 & one) << 25);
    r->n[7] = (t7 >> 1) + ((t8 & one) << 25);
    r->n[8] = (t8 >> 1) + ((t9 & one) << 25);
    r->n[9] = (t9 >> 1);
}

inline void fe_normalize(thread fe *r) {
    uint t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4],
         t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    uint m;
    uint x = t9 >> 22; t9 &= 0x03FFFFFU;
    t0 += x * 0x3D1U; t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= 0x3FFFFFFU;
    t2 += (t1 >> 26); t1 &= 0x3FFFFFFU;
    t3 += (t2 >> 26); t2 &= 0x3FFFFFFU; m = t2;
    t4 += (t3 >> 26); t3 &= 0x3FFFFFFU; m &= t3;
    t5 += (t4 >> 26); t4 &= 0x3FFFFFFU; m &= t4;
    t6 += (t5 >> 26); t5 &= 0x3FFFFFFU; m &= t5;
    t7 += (t6 >> 26); t6 &= 0x3FFFFFFU; m &= t6;
    t8 += (t7 >> 26); t7 &= 0x3FFFFFFU; m &= t7;
    t9 += (t8 >> 26); t8 &= 0x3FFFFFFU; m &= t8;

    x = (t9 >> 22) | ((t9 == 0x03FFFFFU) & (m == 0x3FFFFFFU)
        & ((t1 + 0x40U + ((t0 + 0x3D1U) >> 26)) > 0x3FFFFFFU));

    t0 += x * 0x3D1U; t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= 0x3FFFFFFU;
    t2 += (t1 >> 26); t1 &= 0x3FFFFFFU;
    t3 += (t2 >> 26); t2 &= 0x3FFFFFFU;
    t4 += (t3 >> 26); t3 &= 0x3FFFFFFU;
    t5 += (t4 >> 26); t4 &= 0x3FFFFFFU;
    t6 += (t5 >> 26); t5 &= 0x3FFFFFFU;
    t7 += (t6 >> 26); t6 &= 0x3FFFFFFU;
    t8 += (t7 >> 26); t7 &= 0x3FFFFFFU;
    t9 += (t8 >> 26); t8 &= 0x3FFFFFFU;
    t9 &= 0x03FFFFFU;

    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
    r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}

inline int fe_normalizes_to_zero(thread const fe *r) {
    uint t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4],
         t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];
    uint z0, z1;
    uint x = t9 >> 22; t9 &= 0x03FFFFFU;
    t0 += x * 0x3D1U; t1 += (x << 6);
    t1 += (t0 >> 26); t0 &= 0x3FFFFFFU; z0  = t0; z1  = t0 ^ 0x3D0U;
    t2 += (t1 >> 26); t1 &= 0x3FFFFFFU; z0 |= t1; z1 &= t1 ^ 0x40U;
    t3 += (t2 >> 26); t2 &= 0x3FFFFFFU; z0 |= t2; z1 &= t2;
    t4 += (t3 >> 26); t3 &= 0x3FFFFFFU; z0 |= t3; z1 &= t3;
    t5 += (t4 >> 26); t4 &= 0x3FFFFFFU; z0 |= t4; z1 &= t4;
    t6 += (t5 >> 26); t5 &= 0x3FFFFFFU; z0 |= t5; z1 &= t5;
    t7 += (t6 >> 26); t6 &= 0x3FFFFFFU; z0 |= t6; z1 &= t6;
    t8 += (t7 >> 26); t7 &= 0x3FFFFFFU; z0 |= t7; z1 &= t7;
    t9 += (t8 >> 26); t8 &= 0x3FFFFFFU; z0 |= t8; z1 &= t8;
    z0 |= t9; z1 &= t9 ^ 0x3C00000U;
    return (z0 == 0) | (z1 == 0x3FFFFFFU);
}

inline void fe_set_b32(thread fe *r, thread const uchar *a) {
    r->n[0] = (uint)a[31] | ((uint)a[30] << 8) | ((uint)a[29] << 16) | ((uint)(a[28] & 0x3) << 24);
    r->n[1] = (uint)((a[28] >> 2) & 0x3f) | ((uint)a[27] << 6) | ((uint)a[26] << 14) | ((uint)(a[25] & 0xf) << 22);
    r->n[2] = (uint)((a[25] >> 4) & 0xf) | ((uint)a[24] << 4) | ((uint)a[23] << 12) | ((uint)(a[22] & 0x3f) << 20);
    r->n[3] = (uint)((a[22] >> 6) & 0x3) | ((uint)a[21] << 2) | ((uint)a[20] << 10) | ((uint)a[19] << 18);
    r->n[4] = (uint)a[18] | ((uint)a[17] << 8) | ((uint)a[16] << 16) | ((uint)(a[15] & 0x3) << 24);
    r->n[5] = (uint)((a[15] >> 2) & 0x3f) | ((uint)a[14] << 6) | ((uint)a[13] << 14) | ((uint)(a[12] & 0xf) << 22);
    r->n[6] = (uint)((a[12] >> 4) & 0xf) | ((uint)a[11] << 4) | ((uint)a[10] << 12) | ((uint)(a[9] & 0x3f) << 20);
    r->n[7] = (uint)((a[9] >> 6) & 0x3) | ((uint)a[8] << 2) | ((uint)a[7] << 10) | ((uint)a[6] << 18);
    r->n[8] = (uint)a[5] | ((uint)a[4] << 8) | ((uint)a[3] << 16) | ((uint)(a[2] & 0x3) << 24);
    r->n[9] = (uint)((a[2] >> 2) & 0x3f) | ((uint)a[1] << 6) | ((uint)a[0] << 14);
}

inline void wbe32(thread uchar *p, uint x) { p[0] = x >> 24; p[1] = x >> 16; p[2] = x >> 8; p[3] = x; }

inline void fe_get_b32(thread uchar *r, thread const fe *a) {
    wbe32(&r[0],  (a->n[9] << 10) | (a->n[8] >> 16));
    wbe32(&r[4],  (a->n[8] << 16) | (a->n[7] >> 10));
    wbe32(&r[8],  (a->n[7] << 22) | (a->n[6] >> 4));
    wbe32(&r[12], (a->n[6] << 28) | (a->n[5] << 2) | (a->n[4] >> 24));
    wbe32(&r[16], (a->n[4] << 8) | (a->n[3] >> 18));
    wbe32(&r[20], (a->n[3] << 14) | (a->n[2] >> 12));
    wbe32(&r[24], (a->n[2] << 20) | (a->n[1] >> 6));
    wbe32(&r[28], (a->n[1] << 26) | a->n[0]);
}

/* a^(p-2) mod p —— 费马求逆（libsecp256k1 加法链）*/
inline void fe_inv(thread fe *r, thread const fe *a) {
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;
    int j;
    fe_sqr(&x2, a);       fe_mul(&x2, &x2, a);
    fe_sqr(&x3, &x2);      fe_mul(&x3, &x3, a);
    x6 = x3;  for (j = 0; j < 3; j++) fe_sqr(&x6, &x6);   fe_mul(&x6, &x6, &x3);
    x9 = x6;  for (j = 0; j < 3; j++) fe_sqr(&x9, &x9);   fe_mul(&x9, &x9, &x3);
    x11 = x9; for (j = 0; j < 2; j++) fe_sqr(&x11, &x11); fe_mul(&x11, &x11, &x2);
    x22 = x11; for (j = 0; j < 11; j++) fe_sqr(&x22, &x22); fe_mul(&x22, &x22, &x11);
    x44 = x22; for (j = 0; j < 22; j++) fe_sqr(&x44, &x44); fe_mul(&x44, &x44, &x22);
    x88 = x44; for (j = 0; j < 44; j++) fe_sqr(&x88, &x88); fe_mul(&x88, &x88, &x44);
    x176 = x88; for (j = 0; j < 88; j++) fe_sqr(&x176, &x176); fe_mul(&x176, &x176, &x88);
    x220 = x176; for (j = 0; j < 44; j++) fe_sqr(&x220, &x220); fe_mul(&x220, &x220, &x44);
    x223 = x220; for (j = 0; j < 3; j++) fe_sqr(&x223, &x223); fe_mul(&x223, &x223, &x3);
    t1 = x223; for (j = 0; j < 23; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, &x22);
    for (j = 0; j < 5; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, a);
    for (j = 0; j < 3; j++) fe_sqr(&t1, &t1); fe_mul(&t1, &t1, &x2);
    for (j = 0; j < 2; j++) fe_sqr(&t1, &t1); fe_mul(r, &t1, a);
}

/* ---------------- 群运算 ---------------- */

/* 统一加法/倍点：r = a + b，b 为仿射点 (b.inf 必须为 0)。移植自 secp256k1_gej_add_ge。 */
inline void gej_add_ge(thread gej *r, thread const gej *a, thread const ge *b) {
    fe zz, u1, u2, s1, s2, t, tt, m, n, q, rr, m_alt, rr_alt;
    fe fe_one; fe_set_int(&fe_one, 1);
    int degenerate;
    const int GEJ_X_M = 4, GEJ_Y_M = 4;

    fe_sqr(&zz, &a->z);
    u1 = a->x;
    fe_mul(&u2, &b->x, &zz);
    s1 = a->y;
    fe_mul(&s2, &b->y, &zz);
    fe_mul(&s2, &s2, &a->z);
    t = u1; fe_add(&t, &u2);
    m = s1; fe_add(&m, &s2);
    fe_sqr(&rr, &t);
    fe_negate(&m_alt, &u2, 1);
    fe_mul(&tt, &u1, &m_alt);
    fe_add(&rr, &tt);
    degenerate = fe_normalizes_to_zero(&m);
    rr_alt = s1;
    fe_mul_int(&rr_alt, 2);
    fe_add(&m_alt, &u1);
    fe_cmov(&rr_alt, &rr, !degenerate);
    fe_cmov(&m_alt, &m, !degenerate);
    fe_sqr(&n, &m_alt);
    fe_negate(&q, &t, GEJ_X_M + 1);
    fe_mul(&q, &q, &n);
    fe_sqr(&n, &n);
    fe_cmov(&n, &m, degenerate);
    fe_sqr(&t, &rr_alt);
    fe_mul(&r->z, &a->z, &m_alt);
    fe_add(&t, &q);
    r->x = t;
    fe_mul_int(&t, 2);
    fe_add(&t, &q);
    fe_mul(&t, &t, &rr_alt);
    fe_add(&t, &n);
    fe_negate(&r->y, &t, GEJ_Y_M + 2);
    fe_half(&r->y);
    fe_cmov(&r->x, &b->x, a->inf);
    fe_cmov(&r->y, &b->y, a->inf);
    fe_cmov(&r->z, &fe_one, a->inf);
    r->inf = fe_normalizes_to_zero(&r->z);
}

inline void gej_from_ge(thread gej *r, thread const ge *a) {
    r->x = a->x; r->y = a->y; fe_set_int(&r->z, 1); r->inf = a->inf;
}

/* 转仿射并输出 X||Y (各 32 字节大端，私有内存) */
inline void gej_to_pub(thread uchar *out, thread gej *a) {
    fe zi, z2, z3, x, y;
    fe_inv(&zi, &a->z);
    fe_sqr(&z2, &zi);
    fe_mul(&z3, &zi, &z2);
    fe_mul(&x, &a->x, &z2);
    fe_mul(&y, &a->y, &z3);
    fe_normalize(&x);
    fe_normalize(&y);
    fe_get_b32(&out[0], &x);
    fe_get_b32(&out[32], &y);
}

/* 从 global 大端 64 字节载入 ge (仿射) */
inline void ge_load_g(thread ge *p, device const uchar *xy) {
    uchar t[64];
    for (int i = 0; i < 64; i++) t[i] = xy[i];
    fe_set_b32(&p->x, &t[0]);
    fe_set_b32(&p->y, &t[32]);
    p->inf = 0;
}

/* ---------------- 固定基点标量乘 ----------------
 * ECW = 窗口位宽；ECBITS = base 的有效位数（= log2(每次内核扫描的私钥数）
 *   ECW==1 : table[j] (j=0..31) = 2^j * G，逐 bit（baseline）
 *   ECW>=2 : table[w*(1<<ECW) + d] = d * 2^(w*ECW) * G，comb 窗口法
 * 窗口法把点加次数从 ~ECBITS/2 降到 ceil(ECBITS/ECW)。
 */
inline void ec_base_mul(thread gej *acc, thread const ge *P0, device const uchar *table_b32, uint base) {
    gej_from_ge(acc, P0);
#if ECW == 1
    for (int j = 0; j < 32; j++) {
        if ((base >> j) & 1u) {
            ge tj; ge_load_g(&tj, &table_b32[j * 64]);
            gej_add_ge(acc, acc, &tj);
        }
    }
#else
    for (int w = 0; w < ECW_WINDOWS; w++) {
        uint d = (base >> (w * ECW)) & (ECW_DIGITS - 1);
        if (d) {
            ge tj; ge_load_g(&tj, &table_b32[(w * ECW_DIGITS + d) * 64]);
            gej_add_ge(acc, acc, &tj);
        }
    }
#endif
}

/* 生成器 G 在表中的位置 */
inline void ec_load_G(thread ge *G, device const uchar *table_b32) {
#if ECW == 1
    ge_load_g(G, &table_b32[0 * 64]);          /* 2^0 * G */
#else
    ge_load_g(G, &table_b32[1 * 64]);          /* w=0,d=1 -> 1 * G */
#endif
}

/* ---------------- keccak-256 ---------------- */

constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL,0x0000000000008082UL,0x800000000000808AUL,0x8000000080008000UL,
    0x000000000000808BUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
    0x000000000000008AUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000AUL,
    0x000000008000808BUL,0x800000000000008BUL,0x8000000000008089UL,0x8000000000008003UL,
    0x8000000000008002UL,0x8000000000000080UL,0x000000000000800AUL,0x800000008000000AUL,
    0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL };
constant int KECCAK_RHO[25] = {
    0,1,62,28,27, 36,44,6,55,20, 3,10,43,25,39, 41,45,15,21,8, 18,2,61,56,14 };

inline ulong rotl64(ulong x, int n) { return n == 0 ? x : ((x << n) | (x >> (64 - n))); }

inline void keccakf(thread ulong *s) {
    for (int rnd = 0; rnd < 24; rnd++) {
        ulong c[5], d[5];
        for (int x = 0; x < 5; x++)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        for (int x = 0; x < 5; x++)
            d[x] = c[(x+4)%5] ^ rotl64(c[(x+1)%5], 1);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                s[x + 5*y] ^= d[x];
        ulong b[25];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                b[y + 5*((2*x + 3*y) % 5)] = rotl64(s[x + 5*y], KECCAK_RHO[x + 5*y]);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                s[x + 5*y] = b[x + 5*y] ^ ((~b[(x+1)%5 + 5*y]) & b[(x+2)%5 + 5*y]);
        s[0] ^= KECCAK_RC[rnd];
    }
}

/* keccak256 of exactly 64 bytes */
inline void keccak256_64(thread uchar *out, thread const uchar *in) {
    ulong s[25];
    for (int i = 0; i < 25; i++) s[i] = 0;
    for (int i = 0; i < 64; i++)
        s[i >> 3] ^= (ulong)in[i] << ((i & 7) * 8);
    s[8] ^= (ulong)0x01UL << ((64 & 7) * 8);   /* pad at offset 64 -> lane 8, byte 0 */
    s[16] ^= (ulong)0x80UL << 56;              /* rate 136 -> last lane index 16 */
    keccakf(s);
    for (int i = 0; i < 32; i++)
        out[i] = (uchar)(s[i >> 3] >> ((i & 7) * 8));
}

/* ---------------- sha-256 ---------------- */

constant uint SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

inline uint shr(uint x, int n) { return x >> n; }
inline uint rotr(uint x, int n) { return (x >> n) | (x << (32 - n)); }

inline void sha256_block(thread uint *st, thread const uchar *p) {
    uint w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint)p[4*i] << 24) | ((uint)p[4*i+1] << 16) | ((uint)p[4*i+2] << 8) | (uint)p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ shr(w[i-15],3);
        uint s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ shr(w[i-2],10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA_K[i] + w[i];
        uint S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint mj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

/* sha256 of a short message (< 56 bytes), single block */
inline void sha256_short(thread uchar *out, thread const uchar *msg, int len) {
    uint st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uchar blk[64];
    for (int i = 0; i < 64; i++) blk[i] = 0;
    for (int i = 0; i < len; i++) blk[i] = msg[i];
    blk[len] = 0x80;
    ulong bits = (ulong)len * 8;
    for (int i = 0; i < 8; i++) blk[63 - i] = (uchar)(bits >> (8 * i));
    sha256_block(st, blk);
    for (int i = 0; i < 8; i++) {
        out[4*i]   = st[i] >> 24;
        out[4*i+1] = st[i] >> 16;
        out[4*i+2] = st[i] >> 8;
        out[4*i+3] = st[i];
    }
}

/* ---------------- base58 尾部 + 匹配 ---------------- */

#define TAIL 12
#define ADDR_LEN 34

constant char B58[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z' };

/* 字符类别：1=数字 2=小写 3=大写 0=其它（与 src/matcher.h 一致）*/
inline int char_class(uchar c) {
    if (c >= '0' && c <= '9') return 1;
    if (c >= 'a' && c <= 'z') return 2;
    if (c >= 'A' && c <= 'Z') return 3;
    return 0;
}

/* full[25] 大端；取地址最后 TAIL 个 base58 字符，tc[0] = 最末字符。 */
inline void base58_tail(thread uchar *tc, thread const uchar *full25) {
    uchar num[25];
    for (int i = 0; i < 25; i++) num[i] = full25[i];
    for (int it = 0; it < TAIL; it++) {
        uint rem = 0;
        for (int i = 0; i < 25; i++) {
            uint acc = (rem << 8) | num[i];
            num[i] = acc / 58;
            rem = acc % 58;
        }
        tc[it] = B58[rem];
    }
}

constant uchar B58_INDEX[128] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,0,1,2,3,4,5,6,7,8,255,255,255,255,255,255,
    255,9,10,11,12,13,14,15,16,255,17,18,19,20,21,255,
    22,23,24,25,26,27,28,29,30,31,32,255,255,255,255,255,
    33,34,35,36,37,38,39,40,41,42,43,255,44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,255,255,255,255,255,255
};

inline void base58_address(thread uchar *addr, thread const uchar *full25) {
    uchar num[25];
    for (int i = 0; i < 25; ++i) num[i] = full25[i];
    for (int i = 0; i < ADDR_LEN; ++i) addr[i] = '1';
    for (int it = 0; it < ADDR_LEN; ++it) {
        uint rem = 0;
        for (int i = 0; i < 25; ++i) {
            uint acc = (rem << 8) | num[i];
            num[i] = acc / 58;
            rem = acc % 58;
        }
        addr[ADDR_LEN - 1 - it] = B58[rem];
    }
}

/* 返回结尾「相同」或「连续」尾段的最大长度。
 * 连续 = 只认升序：每位 ASCII +1，且整段同一类别（全数字/全小写/全大写）。
 * tc[0] 是地址最末字符，所以升序要求 tc[0] = tc[1] + 1。*/
inline int tail_match_len(thread const uchar *tc) {
    int rep = 1;
    for (int i = 1; i < TAIL; i++) { if (tc[i] == tc[0]) rep++; else break; }

    int seq = 1;
    int cls = char_class(tc[0]);
    if (cls != 0 && char_class(tc[1]) == cls && (int)tc[0] - (int)tc[1] == 1) {
        seq = 2;
        for (int i = 2; i < TAIL; i++) {
            if (char_class(tc[i]) != cls) break;
            if ((int)tc[i - 1] - (int)tc[i] != 1) break;
            seq++;
        }
    }
    return rep > seq ? rep : seq;
}

/* ---------------- 主内核 ---------------- */

/* pub = X||Y (64B 大端)。算 TRON 地址结尾 TAIL 个 base58 位置。 */
inline void pub_to_tail(thread const uchar *pub, thread uchar *tp) {
    uchar h[32];
    keccak256_64(h, pub);
    uchar payload[21];
    payload[0] = 0x41;
    for (int i = 0; i < 20; i++) payload[1 + i] = h[12 + i];
    uchar d1[32], d2[32];
    sha256_short(d1, payload, 21);
    sha256_short(d2, d1, 32);
    uchar full[25];
    for (int i = 0; i < 21; i++) full[i] = payload[i];
    for (int i = 0; i < 4; i++) full[21 + i] = d2[i];
    base58_tail(tp, full);
}

inline void emit_if_match(thread const uchar *pub, uint s,
                          device const uint *dfa,
                          device const uint *out_start,
                          device const uint *out_len,
                          device const uint *out_ids,
                          volatile device uint *out_count,
                          device uint *out_hits, constant uint& out_cap) {
    uchar h[32];
    keccak256_64(h, pub);
    uchar payload[21];
    payload[0] = 0x41;
    for (int i = 0; i < 20; ++i) payload[1 + i] = h[12 + i];
    uchar d1[32], d2[32];
    sha256_short(d1, payload, 21);
    sha256_short(d2, d1, 32);
    uchar full[25];
    for (int i = 0; i < 21; ++i) full[i] = payload[i];
    for (int i = 0; i < 4; ++i) full[21 + i] = d2[i];
    uchar addr[ADDR_LEN];
    base58_address(addr, full);
    uint state = 0;
    for (int i = 1; i < ADDR_LEN; ++i) {
        uchar c = addr[i];
        uint ai = c < 128 ? (uint)B58_INDEX[c] : 255;
        if (ai >= 58) { state = 0; continue; }
        state = dfa[state * 58 + ai];
        uint start = out_start[state];
        uint len = out_len[state];
        for (uint j = 0; j < len; ++j) {
            uint idx = resident_atomic_add(out_count, 1U);
            if (idx < out_cap) {
                out_hits[idx * 2] = s;
                out_hits[idx * 2 + 1] = out_ids[start + j];
            }
        }
    }
}

static inline void probe_one(thread gej *acc, uint s,
                      device const uint *dfa,
                      device const uint *out_start,
                      device const uint *out_len,
                      device const uint *out_ids,
                      volatile device uint *out_count,
                      device uint *out_hits, constant uint& out_cap) {
    uchar pub[64];
    gej_to_pub(pub, acc);
    emit_if_match(pub, s, dfa, out_start, out_len, out_ids, out_count, out_hits, out_cap);
}

/* fe <-> threadgroup uint[10] */
inline void fe_ld_l(thread fe *r, threadgroup const uint *p) { for (int i = 0; i < 10; i++) r->n[i] = p[i]; }
inline void fe_st_l(threadgroup uint *p, thread const fe *a) { for (int i = 0; i < 10; i++) p[i] = a->n[i]; }

/* Jacobian (X,Y) + 已求逆的 z^-1 -> 仿射 pub 64B 大端 */
inline void jac_to_pub(thread uchar *pub, thread const fe *X, thread const fe *Y, thread const fe *zi) {
    fe z2, z3, x, y;
    fe_sqr(&z2, zi);
    fe_mul(&z3, zi, &z2);
    fe_mul(&x, X, &z2);
    fe_mul(&y, Y, &z3);
    fe_normalize(&x);
    fe_normalize(&y);
    fe_get_b32(&pub[0], &x);
    fe_get_b32(&pub[32], &y);
}

#ifndef MONT_N
#define MONT_N 1
#endif

/* Montgomery 批量求逆：zbuf 存 N 个 Z，出口 zinv 存 N 个 z^-1。所有 WI 必须都调用。 */
inline void mont_batch_invert(threadgroup uint *zbuf, threadgroup uint *zinv, int lid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        fe prod, zt;
        fe_ld_l(&prod, &zbuf[0]);
        fe_st_l(&zinv[0], &prod);
        for (int i = 1; i < MONT_N; i++) {
            fe_ld_l(&zt, &zbuf[i * 10]);
            fe_mul(&prod, &prod, &zt);
            fe_st_l(&zinv[i * 10], &prod);
        }
        fe inv;
        fe_inv(&inv, &prod);
        for (int i = MONT_N - 1; i >= 1; i--) {
            fe pre, t;
            fe_ld_l(&pre, &zinv[(i - 1) * 10]);
            fe_mul(&t, &inv, &pre);
            fe_ld_l(&zt, &zbuf[i * 10]);
            fe_mul(&inv, &inv, &zt);
            fe_st_l(&zinv[i * 10], &t);
        }
        fe_st_l(&zinv[0], &inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#if MONT_N <= 1

kernel void tron_vanity_probe(
        device const uchar *P0_b32,
        device const uchar *table_b32,
        device const uint *dfa,
        device const uint *out_start,
        device const uint *out_len,
        device const uint *out_ids,
        volatile device uint *out_count,
        device uint *out_hits,
        constant uint& out_cap) {
    uint gid = 0;
    uint base = gid * KPI;

    ge Gpt; ec_load_G(&Gpt, table_b32);
    ge P0;  ge_load_g(&P0, &P0_b32[0]);

    gej acc;
    ec_base_mul(&acc, &P0, table_b32, base);

    #pragma unroll 1
    for (uint i = 0; i < KPI; i++) {
        probe_one(&acc, base + i, dfa, out_start, out_len, out_ids, out_count, out_hits, out_cap);
        gej_add_ge(&acc, &acc, &Gpt);
    }
}

#else  /* Montgomery 批量求逆：1 work-item = 1 私钥 = 1 Jacobian 点，work-group 内 N 个一起求逆 */

kernel __attribute__((reqd_work_group_size(MONT_N, 1, 1)))
void tron_vanity_probe(
        device const uchar *P0_b32,
        device const uchar *table_b32,
        device const uint *dfa,
        device const uint *out_start,
        device const uint *out_len,
        device const uint *out_ids,
        volatile device uint *out_count,
        device uint *out_hits,
        constant uint& out_cap) {
    uint gid = 0;
    uint lid = get_local_id(0);
    uint s = gid;

    ge P0; ge_load_g(&P0, &P0_b32[0]);
    gej acc;
    ec_base_mul(&acc, &P0, table_b32, s);        /* 本 WI 唯一的 Jacobian 点 */

    threadgroup uint zbuf[MONT_N * 10];
    threadgroup uint zinv[MONT_N * 10];
    fe_st_l(&zbuf[lid * 10], &acc.z);
    mont_batch_invert(zbuf, zinv, lid);

    fe zi;
    fe_ld_l(&zi, &zinv[lid * 10]);
    uchar pub[64];
    jac_to_pub(pub, &acc.x, &acc.y, &zi);
    emit_if_match(pub, s, dfa, out_start, out_len, out_ids, out_count, out_hits, out_cap);
}

#endif

/* ---------------- 分阶段 profile 内核 ----------------
 * host 用 -D PROF_STAGE=1..6 各编译一次，逐阶段消融测吞吐；
 * 相邻两阶段的 ns/key 之差 ≈ 该阶段耗时。每阶段都把结果 xor 进 sink 防止被优化掉。
 *  1 EC 标量乘   2 +模逆/转仿射   3 +keccak   4 +sha256d   5 +base58   6 +尾号匹配
 */
#ifndef PROF_STAGE
#define PROF_STAGE 0
#endif
#if PROF_STAGE > 0
kernel void prof(device const uchar *P0_b32,
                   device const uchar *table_b32,
                   device uint *sink) {
    uint gid = 0;
    uint base = gid * KPI;

    ge Gpt; ec_load_G(&Gpt, table_b32);
    ge P0;  ge_load_g(&P0, &P0_b32[0]);
    gej acc; ec_base_mul(&acc, &P0, table_b32, base);

    uint sv = 0;
    #pragma unroll 1
    for (uint i = 0; i < KPI; i++) {
#if PROF_STAGE >= 2
        uchar pub[64]; gej_to_pub(pub, &acc); sv ^= pub[0] ^ pub[63];
#endif
#if PROF_STAGE >= 3
        uchar h[32]; keccak256_64(h, pub); sv ^= h[0];
#endif
#if PROF_STAGE >= 4
        uchar payload[21]; payload[0] = 0x41;
        for (int k = 0; k < 20; k++) payload[1+k] = h[12+k];
        uchar d1[32], d2[32]; sha256_short(d1, payload, 21); sha256_short(d2, d1, 32); sv ^= d2[0];
#endif
#if PROF_STAGE >= 5
        uchar full[25];
        for (int k = 0; k < 21; k++) full[k] = payload[k];
        for (int k = 0; k < 4; k++) full[21+k] = d2[k];
        uchar tp[TAIL]; base58_tail(tp, full); sv ^= tp[0];
#endif
#if PROF_STAGE >= 6
        sv ^= (uint)tail_match_len(tp);
#endif
        gej_add_ge(&acc, &acc, &Gpt);
    }
#if PROF_STAGE == 1
    sv ^= acc.x.n[0];
#endif
    sink[gid] = sv;
}
#endif

/* ---------------- 自检内核（供 host 交叉验证） ---------------- */

/* 输入 n 个私钥标量偏移，输出各自 P0 + s*G 的 pub(64B)；P0 由 host 给 */
kernel void test_pub(
        device const uchar *P0_b32,
        device const uchar *table_b32,
        device const uint *scalars,
        device uchar *pubout,
        constant uint& n) {
    uint gid = 0;
    if (gid >= n) return;
    uint s = scalars[gid];

    ge P0;
    ge_load_g(&P0, &P0_b32[0]);
    gej acc; gej_from_ge(&acc, &P0);
    for (int j = 0; j < 32; j++)
        if ((s >> j) & 1u) {
            ge tj;
            ge_load_g(&tj, &table_b32[j * 64]);
            gej_add_ge(&acc, &acc, &tj);
        }
    uchar pub[64];
    gej_to_pub(pub, &acc);
    for (int i = 0; i < 64; i++) pubout[gid * 64 + i] = pub[i];
}

#if MONT_N > 1
/* 批量求逆版：work-group=MONT_N，用 ec_base_mul(ECW) 算点，Montgomery 一次求逆，输出 pub。
 * host 对照 test_pub（逐点单独求逆）与 libsecp256k1，验证 batch inverse 数学一致。
 * scalars[gid] 必须 < 2^ECBITS。 */
kernel __attribute__((reqd_work_group_size(MONT_N, 1, 1)))
void test_mont(device const uchar *P0_b32,
               device const uchar *table_b32,
               device const uint *scalars,
               device uchar *pubout,
               constant uint& n) {
    uint gid = 0;
    uint lid = get_local_id(0);
    uint s = (gid < n) ? scalars[gid] : 0;

    ge P0; ge_load_g(&P0, &P0_b32[0]);
    gej acc; ec_base_mul(&acc, &P0, table_b32, s);

    threadgroup uint zbuf[MONT_N * 10];
    threadgroup uint zinv[MONT_N * 10];
    fe_st_l(&zbuf[lid * 10], &acc.z);
    mont_batch_invert(zbuf, zinv, lid);

    fe zi; fe_ld_l(&zi, &zinv[lid * 10]);
    uchar pub[64];
    jac_to_pub(pub, &acc.x, &acc.y, &zi);
    if (gid < n) for (int i = 0; i < 64; i++) pubout[gid * 64 + i] = pub[i];
}
#endif
