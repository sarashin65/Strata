// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <sycl/ext/intel/math.hpp>

#include <cmath>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    sycl::half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
struct IQ4XSBlock {
    sycl::half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    sycl::half d;
};
struct Q40Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q50Block {
    sycl::half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        /*
        DPCT1096:518: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset);
    }
    return x;
}

// V (power of two, 2..32) values reduced at once. Step o pairs lane L with
// lane L^o exactly as warp_sum does. The kept value is own + the partner's
// value of the same index, so every value sees the same pairs in the same
// order as warp_sum(v[i]). After the transposed steps, lane L holds
// warp_sum_multi_index<V>(lane) in v[0]; the remaining offsets are a plain
// butterfly.
template <int V>
__dpct_inline__ void warp_sum_multi(float (&v)[V], int lane) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int o = 16 >> st, w = V >> st;
        if (w > 1) {
            const bool hi = (lane & o) != 0;
#pragma unroll
            for (int i = 0; i < w / 2; ++i) {
                const float send = hi ? v[i] : v[i + w / 2];
                const float keep = hi ? v[i + w / 2] : v[i];
                v[i] = keep + dpct::permute_sub_group_by_xor(sg, send, o);
            }
        } else {
            v[0] += dpct::permute_sub_group_by_xor(sg, v[0], o);
        }
    }
}

template <int V>
__dpct_inline__ int warp_sum_multi_index(int lane) {
    int idx = 0;
#pragma unroll
    for (int st = 0; (V >> st) > 1; ++st)
        idx = 2 * idx + ((lane >> (4 - st)) & 1);
    return idx;
}

bool mmvq_rot_on() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_MMVQ_ROT");
        const bool on = value != nullptr && std::strcmp(value, "1") == 0;
        if (on)
            std::fprintf(stderr, "strata mmvq: rotated rows ON (STRATA_MMVQ_ROT)\n");
        return on;
    }();
    return enabled;
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = sycl::fmax(
            x,
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset));
    }
    return x;
}


void native_quantize_q8_1_kernel(const float* __restrict__ x,
                                           Q81Block* __restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS +
                  int(item_ct1.get_local_id(2));
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = sycl::half2(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q5_q8_dot_impl(const int *__restrict__ vl, const int *__restrict__ vh,
               const int *__restrict__ u, const uint8_t *__restrict__ sc,
               const uint8_t *__restrict__ m, const sycl::half2 &dm5,
               const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 =
            dpct::dp4a(v0i, u[2 * i], dpct::dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = dpct::dp4a(0x01010101, u[2 * i],
                                    dpct::dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        sycl::float2(sycl::ext::intel::math::half2float(dm5.x()),
                     sycl::ext::intel::math::half2float(dm5.y()));
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

// Q5_K cannot absorb its independent minimum term into q-minus.  This
// candidate keeps that term exact, but hoists the q-minus words out of the
// column loop.  q0s/q1s are q-16 in signed bytes; adding 16*dot2 restores
// the original integer dot before the original scale/min float sequence.
__dpct_inline__ int q5_qminus16(int q) {
    return static_cast<int>(((static_cast<uint32_t>(q) | 0x80808080u) -
                             0x10101010u) ^ 0x80808080u);
}

__dpct_inline__ float
q5_q8_dot_qminus_impl(const int *__restrict__ q0s, const int *__restrict__ q1s,
                      const int *__restrict__ u, const uint8_t *__restrict__ sc,
                      const uint8_t *__restrict__ m, const sycl::half2 &dm5,
                      const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int dot2 =
            dpct::dp4a(0x01010101, u[2 * i],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0));
        const int dot1_minus =
            dpct::dp4a(q0s[i], u[2 * i],
                       dpct::dp4a(q1s[i], u[2 * i + 1], 0));
        const int dot1 = dot1_minus + dot2 * 16;
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        sycl::float2(sycl::ext::intel::math::half2float(dm5.x()),
                     sycl::ext::intel::math::half2float(dm5.y()));
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

/*
DPCT1110:41: The total declared local variable size in device function q5_q8_dot
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ float q5_q8_dot(const Q5KBlock *__restrict__ bq5,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>

void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out,
                                        float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:293: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The rotated one-column Q5_K path assigns row i's original warp role to
// (warp - i) & 3. Each role therefore uses the original warp's tid, block
// order, and q5_q8_dot expression; only the row-to-warp assignment changes.
// The reduction remains role 0 + role 1 + role 2 + role 3 followed by the
// same warp_sum, so no floating-point term is reordered.
void native_q5_k_mmvq_rot_kernel(
    const Q5KBlock* __restrict__ w, const Q81Block* __restrict__ x,
    float* __restrict__ y, int n_in, int n_out,
    float partial[3][WARPS][WARP]) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = int(it.get_local_id(1));
    const int lane = int(it.get_local_id(2));
    const int row0 = WARPS * int(it.get_group(2));
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int blocks_per_row = n_in / QK;
    float tmp[WARPS] = {};
#pragma unroll
    for (int i = 0; i < WARPS; ++i) {
        const int role = (warp - i) & (WARPS - 1);
        const int tid = WARP * role + lane;
        if (row0 + i >= n_out) continue;
        for (int kbx = tid / (QI / VDR); kbx < blocks_per_row;
             kbx += BLOCKS_PER_ITER) {
            const int kby = kbx * (QK / Q8K);
            const int kqs = VDR * (tid % (QI / VDR));
            const std::size_t block =
                std::size_t(row0 + i) * blocks_per_row + kbx;
            tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
        }
    }

    for (int i = 0; i < WARPS; ++i) {
        const int role = (warp - i) & (WARPS - 1);
        if (role > 0) partial[role - 1][i][lane] = tmp[i];
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (row0 + warp >= n_out) return;

    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < WARPS; ++i)
        if (i == warp) t = tmp[i];
#pragma unroll
    for (int l = 0; l < WARPS - 1; ++l)
        t += partial[l][warp][lane];
    t = warp_sum(t);
    if (lane == 0) y[row0 + warp] = t;
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__dpct_inline__ float q2_q8_dot(const Q20Block *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = dpct::dp4a(u, qx, sumi);
        sumi = dpct::dp4a(v, qy, sumi);
    }
    const float d8 = chunk->ds[0];
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>

void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out,
                                        float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:294: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__dpct_inline__ int load_int_b2(const void *ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__dpct_inline__ float q3_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const uint8_t *__restrict__ scales,
                                     int scale_offset, float d3,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = sycl::ext::intel::math::vsubss4<unsigned>(vil, vih);
        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__dpct_inline__ float q3_q8_dot(const Q3KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + i].ds[0];
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>

void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out,
                                        float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:295: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
inline dpct::global_memory<int8_t, 1>
    iq4nl_values(sycl::range<1>(16), {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1, 13, 25, 38, 53, 69, 89, 113});

__dpct_inline__ sycl::int2 iq4_table_lookup(int q4, int8_t *iq4nl_values) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__dpct_inline__ float iq4_xs_q8_dot(const IQ4XSBlock *__restrict__ w,
                                    const Q81Block *__restrict__ x, int iqs,
                                    int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const sycl::int2 v = iq4_table_lookup(aux_q4, iq4nl_values);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = dpct::dp4a(v.x(), u0, sumi);
        sumi = dpct::dp4a(v.y(), u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::ext::intel::math::half2float(w->d) * x[iqs / 4].ds[0];
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>

void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out,
                                         int8_t *iq4nl_values,
                                         float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs, iq4nl_values);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:296: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q4_q8_dot_impl(const int *__restrict__ v, const int *__restrict__ u,
               const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
               const sycl::half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 =
            dpct::dp4a(v1i, u[2 * i + 1], dpct::dp4a(v0i, u[2 * i], 0));
        const int dot2 = dpct::dp4a(0x01010101, u[2 * i + 1],
                                    dpct::dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm4f =
        sycl::float2(sycl::ext::intel::math::half2float(dm4.x()),
                     sycl::ext::intel::math::half2float(dm4.y()));
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

__dpct_inline__ float q4_q8_dot(const Q4KBlock *__restrict__ bq4,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>

void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out,
                                        float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:297: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__dpct_inline__ float q6_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const int8_t *__restrict__ scales, float d,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int q = vil | vih;
        const int dot =
            dpct::dp4a(q, u[i], 0) - dpct::dp4a(0x20202020, u[i], 0);
        sumf += d8[i] * (dot * sc);
    }
    return d * sumf;
}

__dpct_inline__ float q6_q8_dot(const Q6KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + 2 * i].ds[0];
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>

void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out,
                                        float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:298: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}
// Plane-major Q6_K view used by the measured reordered benchmark.  Its four
// planes together contain exactly 210 bytes per Q6_K block.
struct Q6KReorderedW {
    int q0s;
    int q1s;
    int sc0;
    int sc1;
    int bq8_offset;
    float d;
};

__dpct_inline__ int load_signed_scale32(const int8_t* scales, std::size_t index) {
    const auto* words = reinterpret_cast<const uint32_t*>(scales);
    const uint32_t word = words[index >> 2];
    const uint32_t byte = (word >> (8 * (index & 3))) & 0xffu;
    return static_cast<int>(static_cast<int8_t>(byte));
}

__dpct_inline__ Q6KReorderedW q6_k_reordered_load(
    const uint8_t* __restrict__ rq, const uint8_t* __restrict__ rh,
    const int8_t* __restrict__ rs, const sycl::half* __restrict__ rd,
    std::size_t block, int iqs) {
    Q6KReorderedW r;
    r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const auto* qlp = reinterpret_cast<const uint32_t*>(rq + block * 128);
    const auto* qhp = reinterpret_cast<const uint32_t*>(rh + block * 64);
    const int vl = static_cast<int>(qlp[iqs]);
    const int vh = static_cast<int>(qhp[8 * (iqs / 16) + iqs % 8]) >> vh_shift;
    const int q0 = (vl & 0x0f0f0f0f) | ((vh << 4) & 0x30303030);
    const int q1 = ((vl >> 4) & 0x0f0f0f0f) |
                   (((vh >> 4) << 4) & 0x30303030);
    r.q0s = static_cast<int>(((static_cast<uint32_t>(q0) | 0x80808080u) -
                               0x20202020u) ^ 0x80808080u);
    r.q1s = static_cast<int>(((static_cast<uint32_t>(q1) | 0x80808080u) -
                               0x20202020u) ^ 0x80808080u);
    r.sc0 = load_signed_scale32(rs + block * 16, scale_offset);
    r.sc1 = load_signed_scale32(rs + block * 16, scale_offset + 4);
    r.d = static_cast<float>(rd[block]);
    return r;
}

__dpct_inline__ float q6_k_reordered_dot(
    const Q6KReorderedW& w, const Q81Block* __restrict__ x, int iqs) {
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* chunk = x + w.bq8_offset + 2 * i;
        u[i] = reinterpret_cast<const int*>(chunk->qs)[iqs % 8];
        d8[i] = chunk->ds[0];
    }
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = (i == 0) ? w.sc0 : w.sc1;
        const int q = (i == 0) ? w.q0s : w.q1s;
        const int dot = dpct::dp4a(q, u[i], 0);
        sumf += d8[i] * static_cast<float>(dot * sc);
    }
    return w.d * sumf;
}

template<int NCOLS, int NW, int ROWS>
void native_q6_k_reordered_kernel(
    const uint8_t* __restrict__ rq, const uint8_t* __restrict__ rh,
    const int8_t* __restrict__ rs, const sycl::half* __restrict__ rd,
    const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
    float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BLOCKS_PER_ITER = NW;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    const int x_stride = n_in / Q8K;
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block =
                    std::size_t(row0 + i) * blocks_per_row + kbx;
                const Q6KReorderedW w =
                    q6_k_reordered_load(rq, rh, rs, rd, block, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const Q81Block* xj =
                        x + std::size_t(j) * x_stride + kby;
                    tmp[j][i] += q6_k_reordered_dot(w, xj, kqs);
                }
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i]
                       [item_ct1.get_local_id(2)] = tmp[j][i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l)
                tmp[j][i] +=
                    partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (item_ct1.get_local_id(2) == i && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// Q6_K reordered layout with the same four-row role rotation as the generic
// path. Every role uses the original kbx = tid / 32, kqs = tid % 32 mapping
// and q6_k_reordered_dot sequence. Sums are combined in role order before the
// unchanged warp_sum, preserving the original floating-point order.
template <int NCOLS>
void native_q6_k_reordered_rot_kernel(
    const uint8_t* __restrict__ rq, const uint8_t* __restrict__ rh,
    const int8_t* __restrict__ rs, const sycl::half* __restrict__ rd,
    const Q81Block* __restrict__ x, float* __restrict__ y, int n_in,
    int n_out, float partial[3][NCOLS][4][32]) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = int(it.get_local_id(1));
    const int lane = int(it.get_local_id(2));
    const int row0 = WARPS * int(it.get_group(2));
    const int blocks_per_row = n_in / QK;
    const int x_stride = n_in / Q8K;
    float tmp[NCOLS][WARPS] = {};
#pragma unroll
    for (int i = 0; i < WARPS; ++i) {
        const int role = (warp - i) & (WARPS - 1);
        const int tid = WARP * role + lane;
        if (row0 + i >= n_out) continue;
        for (int kbx = tid / 32; kbx < blocks_per_row; kbx += WARPS) {
            const int kby = kbx * 8;
            const int kqs = tid % 32;
            const std::size_t block =
                std::size_t(row0 + i) * blocks_per_row + kbx;
            const Q6KReorderedW wv =
                q6_k_reordered_load(rq, rh, rs, rd, block, kqs);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                tmp[j][i] += q6_k_reordered_dot(
                    wv, x + std::size_t(j) * x_stride + kby, kqs);
        }
    }

#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < WARPS; ++i) {
            const int role = (warp - i) & (WARPS - 1);
            if (role > 0) partial[role - 1][j][i][lane] = tmp[j][i];
        }
    it.barrier(sycl::access::fence_space::local_space);
    if (row0 + warp >= n_out) return;

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i)
            if (i == warp) t = tmp[j][i];
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            t += partial[l][j][warp][lane];
        t = warp_sum(t);
        if (lane == 0)
            y[std::size_t(j) * n_out + row0 + warp] = t;
    }
}

template<int NCOLS>
void launch_q6_k_reordered_n(
    const uint8_t* rq, const uint8_t* rh, const int8_t* rs,
    const sycl::half* rd, const void* x_q8_1, float* y, int n_in, int n_out,
    dpct::queue_ptr s) {
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const dpct::dim3 threads(WARP, WARPS);
    if (mmvq_rot_on()) {
        const unsigned blocks =
            unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[3][NCOLS][WARPS][32 /*WARP*/], 0>
                partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_reordered_rot_kernel<NCOLS>(
                            rq, rh, rs, rd, x, y, n_in, n_out, partial);
                    });
        });
        return;
    }
    if constexpr (NCOLS == 1) {
        const unsigned blocks =
            unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[WARPS - 1][1][WARPS][32 /*WARP*/], 0>
                partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_reordered_kernel<1, WARPS, WARPS>(
                            rq, rh, rs, rd, x, y, n_in, n_out, partial);
                    });
        });
    } else {
        const unsigned blocks = unsigned(n_out);
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[WARPS - 1][NCOLS][1][32 /*WARP*/], 0>
                partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_reordered_kernel<NCOLS, WARPS, 1>(
                            rq, rh, rs, rd, x, y, n_in, n_out, partial);
                    });
        });
    }
}

void launch_q6_k_reordered(
    const void* rq, const void* rh, const void* rs, const void* rd,
    const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
    void* stream) {
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const auto* qlp = static_cast<const uint8_t*>(rq);
    const auto* qhp = static_cast<const uint8_t*>(rh);
    const auto* scales = static_cast<const int8_t*>(rs);
    const auto* d = static_cast<const sycl::half*>(rd);
    switch (ncols) {
    case 1: launch_q6_k_reordered_n<1>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 2: launch_q6_k_reordered_n<2>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 3: launch_q6_k_reordered_n<3>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 4: launch_q6_k_reordered_n<4>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 5: launch_q6_k_reordered_n<5>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 6: launch_q6_k_reordered_n<6>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 7: launch_q6_k_reordered_n<7>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    case 8: launch_q6_k_reordered_n<8>(qlp, qhp, scales, d, x_q8_1, y, n_in, n_out, s); break;
    default: throw std::invalid_argument("native Q6_K reorder requires 1..8 columns");
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__dpct_inline__ float small_q8_dot(const Q40Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(vi0, reinterpret_cast<const int *>(x->qs)[iqs + i],
                          sumi);
        sumi = dpct::dp4a(
            vi1, reinterpret_cast<const int *>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        sycl::float2(sycl::ext::intel::math::half2float((x->ds).x()),
                     sycl::ext::intel::math::half2float((x->ds).y()));
    const float d = w->d;
    return d * (sumi * ds.x() - 4 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q50Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dpct::dp4a(vi0, reinterpret_cast<const int *>(x->qs)[iqs + i],
                          sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dpct::dp4a(
            vi1, reinterpret_cast<const int *>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        sycl::float2(sycl::ext::intel::math::half2float((x->ds).x()),
                     sycl::ext::intel::math::half2float((x->ds).y()));
    const float d = w->d;
    return d * (sumi * ds.x() - 8 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q80Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = dpct::dp4a(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = x->ds[0];
    return d0 * d1 * float(sumi);
}

__dpct_inline__ float small_q8_dot(const IQ4NLBlock *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const sycl::int2 v =
            iq4_table_lookup(load_int_b2(w->qs, iqs + i), iq4nl_values);
        sumi = dpct::dp4a(v.x(), q8[i], sumi);
        sumi = dpct::dp4a(v.y(), q8[i + 4], sumi);
    }
    const float d = sycl::ext::intel::math::half2float(w->d) * x->ds[0];
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>

void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out,
                                         int8_t *iq4nl_values,
                                         float partial[3/*WARPS - 1*/][SmallK ? WARPS : 1][32/*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs, iq4nl_values);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    /*
    DPCT1065:299: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int vl[2], vh[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset;
    };
    static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};

struct Q5KQMinusTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int q0s[2], q1s[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset;
    };
    static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl0 = (ql[0] >> (4 * i)) & 0x0f0f0f0f;
            const int vl1 = (ql[4] >> (4 * i)) & 0x0f0f0f0f;
            const int vh0 = ((qh[0] >> (r.bq8_offset + i)) << 4) & 0x10101010;
            const int vh1 = ((qh[4] >> (r.bq8_offset + i)) << 4) & 0x10101010;
            r.q0s[i] = q5_qminus16(vl0 | vh0);
            r.q1s[i] = q5_qminus16(vl1 | vh1);
        }
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_qminus_impl(r.q0s, r.q1s, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset; };
    static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
            const int qo =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = dpct::dp4a(q8[j * 2], r.qx[j], sumi);
            sumi = dpct::dp4a(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + i].ds[0];
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + 2 * i].ds[0];
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { sycl::int2 v[4]; int ls; float dw; };
    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(
            reinterpret_cast<const int *>(w->qs)[iqs + j], iq4nl_values);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::ext::intel::math::half2float(w->d);
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = dpct::dp4a(r.v[j].x(), u0, sumi);
            sumi = dpct::dp4a(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * x[iqs / 4].ds[0];
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    static W load(const Block* __restrict__ w, int) { return W{w}; }
    static float apply(const W &r, const Q81Block *__restrict__ x, int k,
                       int8_t *iq4nl_values) {
        return small_q8_dot(r.w, x, k, iq4nl_values);
    }
};

template <typename F>
auto trait_load(const typename F::Block* w, int iqs, int8_t *iq4nl_values) {
    if constexpr (requires { F::load(w, iqs, iq4nl_values); })
        return F::load(w, iqs, iq4nl_values);
    else
        return F::load(w, iqs);
}

template <typename F>
float trait_apply(const typename F::W& wv, const Q81Block* x, int iqs,
                  int8_t *iq4nl_values) {
    if constexpr (requires { F::apply(wv, x, iqs, iq4nl_values); })
        return F::apply(wv, x, iqs, iq4nl_values);
    else
        return F::apply(wv, x, iqs);
}

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

template <typename F, int NCOLS, int NW, int ROWS>
/*
DPCT1110:42: The total declared local variable size in device function
native_mmvq_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void native_mmvq_multi_kernel(
    const typename F::Block *__restrict__ w, const Q81Block *__restrict__ x,
    float *__restrict__ y, int n_in, int n_out,
    int8_t *iq4nl_values,
    float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BPI =
        F::BPI * NW / WARPS; // blocks per iteration scale with the warp count
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                /*
                DPCT1084:150: The function call "Q5KTraits::load" has multiple
                migration results in different template instantiations that
                could not be unified. You may need to adjust the code.
                */
                const typename F::W wv =
                    trait_load<F>(w + block, kqs, iq4nl_values); // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    /*
                    DPCT1084:151: The function call "Q5KTraits::apply" has
                    multiple migration results in different template
                    instantiations that could not be unified. You may need to
                    adjust the code.
                    */
                    tmp[j][i] +=
                        trait_apply<F>(wv, x + std::size_t(j) * x_stride + kby, kqs, iq4nl_values);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i]
                       [item_ct1.get_local_id(2)] = tmp[j][i];
    }
    /*
    DPCT1065:300: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] +=
                partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (item_ct1.get_local_id(2) == i && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// Four-row rotated exact layout. For row i, physical warp `warp` takes the
// original role `(warp - i) & 3`, hence the same tid, blocks, kqs values, and
// trait_apply calls in the same order as the unrotated kernel. The cross-warp
// sum is still role 0 + role 1 + role 2 + role 3, followed by the same
// warp_sum; rotating roles cannot reorder any floating-point addition.
template <typename F, int NCOLS>
void native_mmvq_rot_kernel(
    const typename F::Block *__restrict__ w, const Q81Block *__restrict__ x,
    float *__restrict__ y, int n_in, int n_out, int8_t *iq4nl_values,
    float partial[3][NCOLS][4][32]) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = int(it.get_local_id(1));
    const int lane = int(it.get_local_id(2));
    const int row0 = WARPS * int(it.get_group(2));
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    float tmp[NCOLS][WARPS] = {};
#pragma unroll
    for (int i = 0; i < WARPS; ++i) {
        const int role = (warp - i) & (WARPS - 1);
        const int tid = WARP * role + lane;
        if (row0 + i >= n_out) continue;
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
            const int kqs = F::kqs(tid);
            const std::size_t block =
                std::size_t(row0 + i) * blocks_per_row + kbx;
            const typename F::W wv =
                trait_load<F>(w + block, kqs, iq4nl_values);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                tmp[j][i] += trait_apply<F>(
                    wv, x + std::size_t(j) * x_stride + kby, kqs,
                    iq4nl_values);
        }
    }

#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < WARPS; ++i) {
            const int role = (warp - i) & (WARPS - 1);
            if (role > 0) partial[role - 1][j][i][lane] = tmp[j][i];
        }
    it.barrier(sycl::access::fence_space::local_space);
    if (row0 + warp >= n_out) return;

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i)
            if (i == warp) t = tmp[j][i];
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            t += partial[l][j][warp][lane];
        t = warp_sum(t);
        if (lane == 0)
            y[std::size_t(j) * n_out + row0 + warp] = t;
    }
}

template <typename F, int NCOLS>
void launch_multi_n(const void *weights, const void *x_q8_1, float *y, int n_in,
                    int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    iq4nl_values.init(*s);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        /*
        DPCT1049:43: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();
                /*
                DPCT1101:515: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1060:516: SYCL range can only be a 1D, 2D, or 3D vector.
                Adjust the code.
                */
                sycl::local_accessor<
                    float[NW - 1 > 0 ? NW - 1 : 1][NCOLS][2][32 /*WARP*/], 0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, NW, WARP),
                                      sycl::range<3>(1, NW, WARP)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_mmvq_multi_kernel<F, NCOLS, NW, 2>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1, partial_acc_ct1);
                        });
            });
        }
        return;
    }
    const dpct::dim3 threads(WARP, WARPS);
    if (mmvq_rot_on()) {
        const unsigned blocks =
            unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});
        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();
            sycl::local_accessor<float[3][NCOLS][WARPS][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_rot_kernel<F, NCOLS>(
                            w, x, y, n_in, n_out, iq4nl_values_ptr_ct1,
                            partial_acc_ct1);
                    });
        });
        return;
    }
    if (n_in / F::DIV < F::BPI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:44: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();
                /*
                DPCT1101:517: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[WARPS - 1 > 0 ? WARPS - 1 : 1][NCOLS]
                                          [WARPS][32 /*WARP*/],
                                     0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                      threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_mmvq_multi_kernel<F, NCOLS, WARPS, WARPS>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1, partial_acc_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049:45: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();
            /*
            DPCT1101:519: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<
                float[WARPS - 1 > 0 ? WARPS - 1 : 1][NCOLS][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1>(
                            w, x, y, n_in, n_out, iq4nl_values_ptr_ct1, partial_acc_ct1);
                    });
        });
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<dpct::queue_ptr>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    /*
    DPCT1010:303: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1000:302: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error != 0) {
        /*
        DPCT1009:304: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:301: The statement could not be removed.
        */
        throw std::runtime_error(std::string("native MMVQ launch: ") +
                                 dpct::get_error_string_dummy(error));
    }
}

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:46: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            iq4nl_values.init(*s);

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

                /*
                DPCT1101:520: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:521: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/],
                                     0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                      threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_small_mmvq_kernel<Weight, Qi, true>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1,
                                partial_acc_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049:47: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        iq4nl_values.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

            /*
            DPCT1101:522: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:523: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_small_mmvq_kernel<Weight, Qi, false>(
                            w, x, y, n_in, n_out, iq4nl_values_ptr_ct1,
                            partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

} // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    ((sycl::queue *)(static_cast<dpct::queue_ptr>(stream)))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                  sycl::range<3>(1, 1, QUANT_THREADS),
                              sycl::range<3>(1, 1, QUANT_THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                native_quantize_q8_1_kernel(x, static_cast<Q81Block *>(x_q8_1),
                                            n_total);
            });
    launch_check();
}

bool native_q5_k_qminus_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_Q5K_QMINUS");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        if (native_q5_k_qminus_enabled())
            launch_multi<Q5KQMinusTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        else
            launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:48: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:524: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:525: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/],
                                     0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                      threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_q5_k_mmvq_kernel<true>(w, x, y, n_in, n_out,
                                                          partial_acc_ct1);
                        });
            });
        }
    } else if (mmvq_rot_on()) {
        const unsigned blocks =
            unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});
        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float[3][WARPS][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q5_k_mmvq_rot_kernel(
                            w, x, y, n_in, n_out, partial_acc_ct1);
                    });
        });
    } else {
        /*
        DPCT1049:49: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:526: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:527: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q5_k_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                       partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:50: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:528: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:529: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q2_0_mmvq_kernel<true>(w, x, y, n_in, n_out,
                                                      partial_acc_ct1);
                    });
        });
    } else {
        /*
        DPCT1049:51: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:530: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:531: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q2_0_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                       partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:52: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:532: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:533: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q3_k_mmvq_kernel<true>(w, x, y, n_in, n_out,
                                                      partial_acc_ct1);
                    });
        });
    } else {
        /*
        DPCT1049:53: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:534: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:535: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q3_k_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                       partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:54: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            iq4nl_values.init(*s);

            s->submit([&](sycl::handler &cgh) {
                auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

                /*
                DPCT1101:536: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:537: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/],
                                     0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                      threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_iq4_xs_mmvq_kernel<true>(
                                w, x, y, n_in, n_out, iq4nl_values_ptr_ct1,
                                partial_acc_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049:55: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        iq4nl_values.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq4nl_values_ptr_ct1 = iq4nl_values.get_ptr();

            /*
            DPCT1101:538: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:539: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_iq4_xs_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                         iq4nl_values_ptr_ct1,
                                                         partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:56: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:540: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:541: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/],
                                     0>
                    partial_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                      threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            native_q4_k_mmvq_kernel<true>(w, x, y, n_in, n_out,
                                                          partial_acc_ct1);
                        });
            });
        }
    } else {
        /*
        DPCT1049:57: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:542: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:543: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q4_k_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                       partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<dpct::queue_ptr>(stream);
    const dpct::dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:58: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:544: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:545: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][WARPS][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_mmvq_kernel<true>(w, x, y, n_in, n_out,
                                                      partial_acc_ct1);
                    });
        });
    } else {
        /*
        DPCT1049:59: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:546: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:547: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0>
                partial_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_q6_k_mmvq_kernel<false>(w, x, y, n_in, n_out,
                                                       partial_acc_ct1);
                    });
        });
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

bool native_q6_k_reorder_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_Q6K_REORDER");
        // The measured path is the default after port-02cs; explicit 0 preserves the byte-for-byte fallback.
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool native_q6_k_reorder_shape(int n_in, int n_out, int ncols) noexcept {
    return n_out > 0 && (n_in == 2560 || n_in == 10240 || n_in == 12288) &&
           (ncols >= 1 && ncols <= MAX_NCOLS);
}

std::size_t native_q6_k_reordered_bytes(int n_in, int n_out) {
    if (!native_q6_k_reorder_shape(n_in, n_out, 1))
        throw std::invalid_argument("native Q6_K reorder does not support this shape");
    return native_mmvq_weight_bytes(14, n_in, n_out);
}

void native_q6_k_reorder_host(const void* src, void* dst, int n_in, int n_out) {
    if (!src || !dst) throw std::invalid_argument("native Q6_K reorder requires non-null host buffers");
    const std::size_t bytes = native_q6_k_reordered_bytes(n_in, n_out);
    const std::size_t blocks = std::size_t(n_in / QK) * std::size_t(n_out);
    const std::size_t ql_bytes = blocks * 128;
    const std::size_t qh_bytes = blocks * 64;
    const std::size_t scale_bytes = blocks * 16;
    const auto* raw = static_cast<const uint8_t*>(src);
    auto* out = static_cast<uint8_t*>(dst);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto* p = raw + block * sizeof(Q6KBlock);
        std::memcpy(out + block * 128, p, 128);
        std::memcpy(out + ql_bytes + block * 64, p + 128, 64);
        std::memcpy(out + ql_bytes + qh_bytes + block * 16, p + 192, 16);
        std::memcpy(out + ql_bytes + qh_bytes + scale_bytes + block * 2, p + 208, 2);
    }
    (void) bytes;
}

// port-02co generic reordered native MMVQ paths.
struct Q5KReorderedBlock {
    uint8_t qs[128];
    uint8_t qh[32];
    uint8_t scales[12];
    sycl::half2 dm;
};
struct Q4KReorderedBlock {
    uint8_t qs[128];
    uint8_t scales[12];
    sycl::half2 dm;
};
struct IQ4XSReorderedBlock {
    uint8_t qs[128];
    uint8_t scales_l[4];
    uint16_t scales_h;
    sycl::half d;
};
static_assert(sizeof(Q5KReorderedBlock) == sizeof(Q5KBlock));
static_assert(sizeof(Q4KReorderedBlock) == sizeof(Q4KBlock));
static_assert(sizeof(IQ4XSReorderedBlock) == sizeof(IQ4XSBlock));

struct Q5KReorderedTraits {
    using Block = Q5KReorderedBlock;
    using W = Q5KTraits::W;
    static constexpr int DIV = Q5KTraits::DIV, T = Q5KTraits::T,
                         KBY = Q5KTraits::KBY, BPI = Q5KTraits::BPI;
    static int kqs(int tid) { return Q5KTraits::kqs(tid); }
    static W load(const Block* b, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(b->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(b->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0]; r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset; r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(b->scales);
        const int j = r.bq8_offset / 2, jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        return Q5KTraits::apply(r, x, iqs);
    }
};
struct Q4KReorderedTraits {
    using Block = Q4KReorderedBlock;
    using W = Q4KTraits::W;
    static constexpr int DIV = Q4KTraits::DIV, T = Q4KTraits::T,
                         KBY = Q4KTraits::KBY, BPI = Q4KTraits::BPI;
    static int kqs(int tid) { return Q4KTraits::kqs(tid); }
    static W load(const Block* b, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(b->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0]; r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(b->scales);
        const int j = r.bq8_offset / 2, jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                            ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                            ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = b->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        return Q4KTraits::apply(r, x, iqs);
    }
};
struct IQ4XSReorderedTraits {
    using Block = IQ4XSReorderedBlock;
    using W = IQ4XSTraits::W;
    static constexpr int DIV = IQ4XSTraits::DIV, T = IQ4XSTraits::T,
                         KBY = IQ4XSTraits::KBY, BPI = IQ4XSTraits::BPI;
    static int kqs(int tid) { return IQ4XSTraits::kqs(tid); }
    static W load(const Block* b, int iqs, int8_t* table) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(b->qs)[iqs + j], table);
        r.ls = ((b->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
               (((b->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::ext::intel::math::half2float(b->d);
        return r;
    }
    static float apply(const W& r, const Q81Block* x, int iqs) {
        return IQ4XSTraits::apply(r, x, iqs);
    }
};

template <typename F, int NCOLS>
void launch_reordered_multi_n(const void* weights, const void* x_q8_1, float* y,
                              int n_in, int n_out, dpct::queue_ptr s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    iq4nl_values.init(*s);
    const dpct::dim3 threads(WARP, WARPS);
    if (mmvq_rot_on()) {
        const unsigned blocks =
            unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        s->submit([&](sycl::handler& cgh) {
            auto table = iq4nl_values.get_ptr();
            sycl::local_accessor<float[3][NCOLS][WARPS][32 /*WARP*/], 0>
                partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads,
                                  threads),
                [=](sycl::nd_item<3> item)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_rot_kernel<F, NCOLS>(
                            w, x, y, n_in, n_out, table, partial);
                    });
        });
        return;
    }
    if (n_in / F::DIV < F::BPI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        s->submit([&](sycl::handler& cgh) {
            auto table = iq4nl_values.get_ptr();
            sycl::local_accessor<float[WARPS - 1][NCOLS][WARPS][32 /*WARP*/], 0> partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                [=](sycl::nd_item<3> item)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_multi_kernel<F, NCOLS, WARPS, WARPS>(
                            w, x, y, n_in, n_out, table, partial);
                    });
        });
    } else {
        s->submit([&](sycl::handler& cgh) {
            auto table = iq4nl_values.get_ptr();
            sycl::local_accessor<float[WARPS - 1][NCOLS][1][32 /*WARP*/], 0> partial(cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                [=](sycl::nd_item<3> item)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1>(
                            w, x, y, n_in, n_out, table, partial);
                    });
        });
    }
}
template <typename F>
void launch_reordered_multi(const void* weights, const void* x_q8_1, float* y,
                            int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<dpct::queue_ptr>(stream);
    switch (ncols) {
    case 1: launch_reordered_multi_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 2: launch_reordered_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 3: launch_reordered_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 4: launch_reordered_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 5: launch_reordered_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 6: launch_reordered_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 7: launch_reordered_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 8: launch_reordered_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
    default: throw std::invalid_argument("native reordered MMVQ requires 1..8 columns");
    }
}

bool native_reorder_enabled(int ggml_type) noexcept {
    const char* name = nullptr;
    switch (ggml_type) {
    case 13: name = "STRATA_Q5K_REORDER"; break;
    case 23: name = "STRATA_IQ4XS_REORDER"; break;
    case 12: name = "STRATA_Q4K_REORDER"; break;
    default: return ggml_type == 14 && native_q6_k_reorder_enabled();
    }
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}
bool native_reorder_shape(int ggml_type, int n_in, int n_out, int ncols) noexcept {
    if (ggml_type == 14) return native_q6_k_reorder_shape(n_in, n_out, ncols);
    if (ggml_type != 12 && ggml_type != 13 && ggml_type != 23) return false;
    return n_out > 0 && n_in > 0 && n_in % 256 == 0 && ncols >= 1 && ncols <= MAX_NCOLS;
}
std::size_t native_reordered_bytes(int ggml_type, int n_in, int n_out) {
    if (!native_reorder_shape(ggml_type, n_in, n_out, 1))
        throw std::invalid_argument("native reorder does not support this shape");
    return native_mmvq_weight_bytes(ggml_type, n_in, n_out);
}
void native_reorder_host(int ggml_type, const void* src, void* dst, int n_in, int n_out) {
    if (!src || !dst) throw std::invalid_argument("native reorder requires non-null host buffers");
    const std::size_t blocks = std::size_t(n_in / QK) * std::size_t(n_out);
    const auto* raw = static_cast<const uint8_t*>(src);
    auto* out = static_cast<uint8_t*>(dst);
    if (ggml_type == 14) { native_q6_k_reorder_host(src, dst, n_in, n_out); return; }
    const std::size_t bytes = native_reordered_bytes(ggml_type, n_in, n_out);
    for (std::size_t block = 0; block < blocks; ++block) {
        if (ggml_type == 13) {
            const auto* p = raw + block * sizeof(Q5KBlock);
            auto* r = reinterpret_cast<Q5KReorderedBlock*>(out + block * sizeof(Q5KReorderedBlock));
            std::memcpy(r->qs, p + offsetof(Q5KBlock, qs), 128);
            std::memcpy(r->qh, p + offsetof(Q5KBlock, qh), 32);
            std::memcpy(r->scales, p + offsetof(Q5KBlock, scales), 12);
            std::memcpy(&r->dm, p + offsetof(Q5KBlock, dm), sizeof(r->dm));
        } else if (ggml_type == 12) {
            const auto* p = raw + block * sizeof(Q4KBlock);
            auto* r = reinterpret_cast<Q4KReorderedBlock*>(out + block * sizeof(Q4KReorderedBlock));
            std::memcpy(r->qs, p + offsetof(Q4KBlock, qs), 128);
            std::memcpy(r->scales, p + offsetof(Q4KBlock, scales), 12);
            std::memcpy(&r->dm, p + offsetof(Q4KBlock, dm), sizeof(r->dm));
        } else if (ggml_type == 23) {
            const auto* p = raw + block * sizeof(IQ4XSBlock);
            auto* r = reinterpret_cast<IQ4XSReorderedBlock*>(out + block * sizeof(IQ4XSReorderedBlock));
            std::memcpy(r->qs, p + offsetof(IQ4XSBlock, qs), 128);
            std::memcpy(r->scales_l, p + offsetof(IQ4XSBlock, scales_l), 4);
            std::memcpy(&r->scales_h, p + offsetof(IQ4XSBlock, scales_h), 2);
            std::memcpy(&r->d, p + offsetof(IQ4XSBlock, d), 2);
        }
    }
    (void) bytes;
}
void native_mmvq_with_reorder(int ggml_type, const void* weights,
                              const void* reordered, const void* x_q8_1,
                              float* y, int n_in, int n_out, int ncols,
                              void* stream) {
    if (ggml_type == 14 && reordered && native_q6_k_reorder_enabled() &&
        native_q6_k_reorder_shape(n_in, n_out, ncols) &&
        native_mmvq_multi_exact()) {
        validate_shape(n_in, ncols, QK);
        if (n_out <= 0)
            throw std::invalid_argument("native MMVQ requires n_out > 0");
        validate_pointer(weights);
        validate_pointer(reordered);
        validate_pointer(x_q8_1);
        validate_pointer(y);
        validate_stream(stream);
        const std::size_t blocks = std::size_t(n_in / QK) * std::size_t(n_out);
        const std::size_t ql_bytes = blocks * 128;
        const std::size_t qh_bytes = blocks * 64;
        const std::size_t scale_bytes = blocks * 16;
        const auto* base = static_cast<const uint8_t*>(reordered);
        launch_q6_k_reordered(base, base + ql_bytes, base + ql_bytes + qh_bytes,
                              base + ql_bytes + qh_bytes + scale_bytes,
                              x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    if (reordered && native_reorder_enabled(ggml_type) &&
        native_reorder_shape(ggml_type, n_in, n_out, ncols) &&
        native_mmvq_multi_exact()) {
        validate_shape(n_in, ncols, QK);
        if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
        validate_pointer(reordered);
        validate_pointer(x_q8_1);
        validate_pointer(y);
        validate_stream(stream);
        switch (ggml_type) {
        case 13: launch_reordered_multi<Q5KReorderedTraits>(reordered, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 12: launch_reordered_multi<Q4KReorderedTraits>(reordered, x_q8_1, y, n_in, n_out, ncols, stream); break;
        case 23: launch_reordered_multi<IQ4XSReorderedTraits>(reordered, x_q8_1, y, n_in, n_out, ncols, stream); break;
        default: break;
        }
        if (ggml_type == 12 || ggml_type == 13 || ggml_type == 23) {
            launch_check();
            return;
        }
    }
    native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

} // namespace strata::kernels
