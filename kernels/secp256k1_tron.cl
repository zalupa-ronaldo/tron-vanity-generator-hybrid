#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef KPI
#define KPI 8            /* 每个 work-item 连续处理的私钥数（host 用 -D KPI=n 覆盖） */
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

typedef struct { uint n[10]; } fe;
typedef struct { fe x, y; int inf; } ge;
typedef struct { fe x, y, z; int inf; } gej;

/* ---------------- 域运算 ---------------- */

inline void fe_mul_inner(uint *r, const uint *a, const uint *b) {
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

inline void fe_sqr_inner(uint *r, const uint *a) {
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

inline void fe_mul(fe *r, const fe *a, const fe *b) { fe_mul_inner(r->n, a->n, b->n); }
inline void fe_sqr(fe *r, const fe *a) { fe_sqr_inner(r->n, a->n); }

inline void fe_set_int(fe *r, uint v) {
    r->n[0] = v;
    for (int i = 1; i < 10; i++) r->n[i] = 0;
}

inline void fe_add(fe *r, const fe *a) {
    for (int i = 0; i < 10; i++) r->n[i] += a->n[i];
}

inline void fe_add_int(fe *r, uint a) { r->n[0] += a; }

inline void fe_mul_int(fe *r, uint a) {
    for (int i = 0; i < 10; i++) r->n[i] *= a;
}

inline void fe_negate(fe *r, const fe *a, int m) {
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

inline void fe_cmov(fe *r, const fe *a, int flag) {
    uint mask0 = (uint)flag + ~((uint)0);
    uint mask1 = ~mask0;
    for (int i = 0; i < 10; i++) r->n[i] = (r->n[i] & mask0) | (a->n[i] & mask1);
}

inline void fe_half(fe *r) {
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

inline void fe_normalize(fe *r) {
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

inline int fe_normalizes_to_zero(const fe *r) {
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

inline void fe_set_b32(fe *r, const uchar *a) {
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

inline void wbe32(uchar *p, uint x) { p[0] = x >> 24; p[1] = x >> 16; p[2] = x >> 8; p[3] = x; }

inline void fe_get_b32(uchar *r, const fe *a) {
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
inline void fe_inv(fe *r, const fe *a) {
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
inline void gej_add_ge(gej *r, const gej *a, const ge *b) {
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

inline void gej_from_ge(gej *r, const ge *a) {
    r->x = a->x; r->y = a->y; fe_set_int(&r->z, 1); r->inf = a->inf;
}

/* 转仿射并输出 X||Y (各 32 字节大端，私有内存) */
inline void gej_to_pub(uchar *out, gej *a) {
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
inline void ge_load_g(ge *p, __global const uchar *xy) {
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
#ifndef ECW
#define ECW 1
#endif
#ifndef ECBITS
#define ECBITS 20
#endif
#define ECW_DIGITS (1 << ECW)
#define ECW_WINDOWS ((ECBITS + ECW - 1) / ECW)

inline void ec_base_mul(gej *acc, const ge *P0, __global const uchar *table_b32, uint base) {
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
inline void ec_load_G(ge *G, __global const uchar *table_b32) {
#if ECW == 1
    ge_load_g(G, &table_b32[0 * 64]);          /* 2^0 * G */
#else
    ge_load_g(G, &table_b32[1 * 64]);          /* w=0,d=1 -> 1 * G */
#endif
}

/* ---------------- keccak-256 ---------------- */

__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL,0x0000000000008082UL,0x800000000000808AUL,0x8000000080008000UL,
    0x000000000000808BUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
    0x000000000000008AUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000AUL,
    0x000000008000808BUL,0x800000000000008BUL,0x8000000000008089UL,0x8000000000008003UL,
    0x8000000000008002UL,0x8000000000000080UL,0x000000000000800AUL,0x800000008000000AUL,
    0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL };
__constant int KECCAK_RHO[25] = {
    0,1,62,28,27, 36,44,6,55,20, 3,10,43,25,39, 41,45,15,21,8, 18,2,61,56,14 };

inline ulong rotl64(ulong x, int n) { return n == 0 ? x : ((x << n) | (x >> (64 - n))); }

inline void keccakf(ulong *s) {
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
inline void keccak256_64(uchar *out, const uchar *in) {
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

__constant uint SHA_K[64] = {
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

inline void sha256_block(uint *st, const uchar *p) {
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
inline void sha256_short(uchar *out, const uchar *msg, int len) {
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

__constant char B58[58] = {
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
inline void base58_tail(uchar *tc, const uchar *full25) {
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

__constant uchar B58_INDEX[128] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,0,1,2,3,4,5,6,7,8,255,255,255,255,255,255,
    255,9,10,11,12,13,14,15,16,255,17,18,19,20,21,255,
    22,23,24,25,26,27,28,29,30,31,32,255,255,255,255,255,
    33,34,35,36,37,38,39,40,41,42,43,255,44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,255,255,255,255,255,255
};

inline void base58_address(uchar *addr, const uchar *full25) {
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
inline int tail_match_len(const uchar *tc) {
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
inline void pub_to_tail(const uchar *pub, uchar *tp) {
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

inline void emit_if_match(const uchar *pub, uint s,
                          __global const uint *dfa,
                          __global const uint *out_start,
                          __global const uint *out_len,
                          __global const uint *out_ids,
                          volatile __global uint *out_count,
                          __global uint *out_hits, uint out_cap) {
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
            uint idx = atomic_inc(out_count);
            if (idx < out_cap) {
                out_hits[idx * 2] = s;
                out_hits[idx * 2 + 1] = out_ids[start + j];
            }
        }
    }
}

inline void probe_one(gej *acc, uint s,
                      __global const uint *dfa,
                      __global const uint *out_start,
                      __global const uint *out_len,
                      __global const uint *out_ids,
                      volatile __global uint *out_count,
                      __global uint *out_hits, uint out_cap) {
    uchar pub[64];
    gej_to_pub(pub, acc);
    emit_if_match(pub, s, dfa, out_start, out_len, out_ids, out_count, out_hits, out_cap);
}

/* fe <-> __local uint[10] */
inline void fe_ld_l(fe *r, __local const uint *p) { for (int i = 0; i < 10; i++) r->n[i] = p[i]; }
inline void fe_st_l(__local uint *p, const fe *a) { for (int i = 0; i < 10; i++) p[i] = a->n[i]; }

/* Jacobian (X,Y) + 已求逆的 z^-1 -> 仿射 pub 64B 大端 */
inline void jac_to_pub(uchar *pub, const fe *X, const fe *Y, const fe *zi) {
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
inline void mont_batch_invert(__local uint *zbuf, __local uint *zinv, int lid) {
    barrier(CLK_LOCAL_MEM_FENCE);
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
    barrier(CLK_LOCAL_MEM_FENCE);
}

#if MONT_N <= 1

__kernel void tron_vanity_probe(
        __global const uchar *P0_b32,
        __global const uchar *table_b32,
        __global const uint *dfa,
        __global const uint *out_start,
        __global const uint *out_len,
        __global const uint *out_ids,
        volatile __global uint *out_count,
        __global uint *out_hits,
        const uint out_cap) {
    uint gid = get_global_id(0);
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

__kernel __attribute__((reqd_work_group_size(MONT_N, 1, 1)))
void tron_vanity_probe(
        __global const uchar *P0_b32,
        __global const uchar *table_b32,
        __global const uint *dfa,
        __global const uint *out_start,
        __global const uint *out_len,
        __global const uint *out_ids,
        volatile __global uint *out_count,
        __global uint *out_hits,
        const uint out_cap) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint s = gid;

    ge P0; ge_load_g(&P0, &P0_b32[0]);
    gej acc;
    ec_base_mul(&acc, &P0, table_b32, s);        /* 本 WI 唯一的 Jacobian 点 */

    __local uint zbuf[MONT_N * 10];
    __local uint zinv[MONT_N * 10];
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
__kernel void prof(__global const uchar *P0_b32,
                   __global const uchar *table_b32,
                   __global uint *sink) {
    uint gid = get_global_id(0);
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
__kernel void test_pub(
        __global const uchar *P0_b32,
        __global const uchar *table_b32,
        __global const uint *scalars,
        __global uchar *pubout,
        const uint n) {
    uint gid = get_global_id(0);
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
__kernel __attribute__((reqd_work_group_size(MONT_N, 1, 1)))
void test_mont(__global const uchar *P0_b32,
               __global const uchar *table_b32,
               __global const uint *scalars,
               __global uchar *pubout,
               const uint n) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint s = (gid < n) ? scalars[gid] : 0;

    ge P0; ge_load_g(&P0, &P0_b32[0]);
    gej acc; ec_base_mul(&acc, &P0, table_b32, s);

    __local uint zbuf[MONT_N * 10];
    __local uint zinv[MONT_N * 10];
    fe_st_l(&zbuf[lid * 10], &acc.z);
    mont_batch_invert(zbuf, zinv, lid);

    fe zi; fe_ld_l(&zi, &zinv[lid * 10]);
    uchar pub[64];
    jac_to_pub(pub, &acc.x, &acc.y, &zi);
    if (gid < n) for (int i = 0; i < 64; i++) pubout[gid * 64 + i] = pub[i];
}
#endif
