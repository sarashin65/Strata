// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010:242: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009:243: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
__dpct_inline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__dpct_inline__ sycl::int2 get_int_from_table_16(const int &q4,
                                                 const int8_t *table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
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
#define ggml_cuda_dp4a(a, b, c) dpct::dp4a((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__dpct_inline__ float vec_dot_q2_0_q8_1(const void *__restrict__ vbq,
                                        const block_q8_1 *__restrict__ bq8_1,
                                        const int &kbx, const int &iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = bq8_1_chunk->ds[0];
    return d2 * d8 * sumi;
}

__dpct_inline__ float vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs,
                                           const uint64_t *iq2xxs_grid) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
        const int grid0 = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
        const int grid1 = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d =
        sycl::ext::intel::math::half2float(bq2->d) * bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs,
                                          const uint64_t *iq2xs_grid) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const sycl::int2 q2_packed =
        sycl::int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
        const int grid_l = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
        const int grid_h = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d =
        sycl::ext::intel::math::half2float(bq2->d) * bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs,
                                         const uint64_t *iq2s_grid) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = sycl::ext::intel::math::vcmpne4<unsigned>(
            ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21),
            0x00000000);
        const int signs1 = sycl::ext::intel::math::vcmpne4<unsigned>(
            ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17),
            0x00000000);
        const int grid_l = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos[0] ^ signs0, signs0);
        const int grid_h = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d =
        sycl::ext::intel::math::half2float(bq2->d) * bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs,
                                           const uint32_t *iq3xxs_grid) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const sycl::int2 q3_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
        const int grid_l = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 =
            sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
        const int grid_h = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d =
        sycl::ext::intel::math::half2float(bq3->d) * bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs,
                                         const uint32_t *iq3s_grid) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const sycl::int2 qs_packed =
        sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::int2 grid_pos =
            sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                       iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = sycl::ext::intel::math::vcmpne4<unsigned>(
            ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21),
            0x00000000);
        const int signs1 = sycl::ext::intel::math::vcmpne4<unsigned>(
            ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17),
            0x00000000);
        const int grid_l = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.x() ^ signs0, signs0);
        const int grid_h = sycl::ext::intel::math::vsub4<unsigned>(
            grid_pos.y() ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d =
        sycl::ext::intel::math::half2float(bq3->d) * bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

__dpct_inline__ float vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs,
                                         const uint32_t *iq1s_grid_gpu) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d =
        sycl::ext::intel::math::half2float(scale.f16) * bq8_1[iqs].ds[0];
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__dpct_inline__ float vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs,
                                          const int8_t *kvalues_iq4nl) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y(), q8[l + 4], sumi);
    }
    const float d = sycl::ext::intel::math::half2float(bq4->d) * bq8_1->ds[0];
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__dpct_inline__ float vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs,
                                          const int8_t *kvalues_iq4nl) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x(), u0, sumi);
        sumi = ggml_cuda_dp4a(v.y(), u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d =
        sycl::ext::intel::math::half2float(bq4->d) * bq8_1[iqs / 4].ds[0];
    return d * sumi;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs, (const uint64_t *)grid); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs, (const uint64_t *)grid); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs, (const uint32_t *)grid); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs, (const int8_t *)grid); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs, (const uint32_t *)grid); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs, (const int8_t *)grid); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs, (const uint64_t *)grid); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs,
                     const void *grid) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs, (const uint32_t *)grid); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const void *) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1096:548: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int o = 16; o > 0; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
// V (power of two, 2..32) values reduced at once.  Each transposed step
// keeps the same value index and adds its XOR partner, so every value sees
// the same pairs in the same order as warp_sum.  After the transposed steps,
// lane L holds warp_sum_multi_index<V>(L) in v[0].
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
    for (int st = 0; (V >> st) > 1; ++st) idx = 2 * idx + ((lane >> (4 - st)) & 1);
    return idx;
}


// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template <int TY>
__dpct_inline__ float row_dot(const uint8_t *row, const block_q8_1 *x, int nb,
                              int lane, const void *grid) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084:152: The function call "Fmt::dot" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs, grid);
    }
    return warp_sum(s);
}

template<int TY>
void mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols,
                                                   const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s =
            row_dot<TY>(wr, x + (size_t)c * (n_in / 32), nb, lane, grid);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

bool native_expert_grouped_split_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("STRATA_EXPERT_GROUPED");
        return v != nullptr && v[0] != '0';
    }();
    return on;
}

template<int TG>
void native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up,
                                                        const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * GU_ROWS + warp; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TG>(wr, xq + (size_t)ent_tok[e] * xb, nb, lane,
                                    grid);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
    }
}

void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + sycl::native::exp(-g))) * up[i];
}

template<int TD>
void native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out,
                                                          const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 8 + warp;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s =
            row_dot<TD>(wr, hq + (size_t)e * hb, nb, lane, grid);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
    }
}

template<int TD>
void native_down_kernel_split(const unsigned long long* __restrict__ grp_ptr,
                              const int32_t* __restrict__ grp_start,
                              const int32_t* __restrict__ n_groups,
                              const int32_t* __restrict__ ent_dst,
                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                              float* __restrict__ out,
                              const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2);
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s =
            row_dot<TD>(wr, hq + (size_t)e * hb, nb, lane, grid);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
    }
}

template<int TD>
void launch_native_down(const unsigned long long* __restrict__ grp_ptr,
                        const int32_t* __restrict__ grp_start,
                        const int32_t* __restrict__ n_groups,
                        const int32_t* __restrict__ ent_dst,
                        const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                        float* __restrict__ out, const void *grid,
                        int64_t cap_groups, bool split, dpct::queue_ptr s) {
    const dpct::dim3 gd((unsigned)((L.n_embd + 7) / 8), (unsigned)cap_groups);
    if (split) {
        const dpct::dim3 gds((unsigned)L.n_embd, (unsigned)cap_groups);
        s->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(gds * sycl::range<3>(1, 1, 32),
                                  sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_kernel_split<TD>(
                            grp_ptr, grp_start, n_groups, ent_dst,
                            hq, L, out, grid);
                    });
        });
    } else {
        s->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(gd * sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        native_down_kernel<TD>(
                            grp_ptr, grp_start, n_groups, ent_dst,
                            hq, L, out, grid);
                    });
        });
    }
}

// ---------------------------------------------------------------- decode once, apply to every column (upstream 8ec94aa)
// The kernels above call Fmt<TY>::dot once per (call, column / entry): each column re-reads the weight words and
// redoes the grid lookups and sign unpacking.  Here each dot is split, as upstream 8ec94aa does, into `load`
// (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block scale as a
// float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale step and the
// same float expression).  apply(load(...)) does the dot's operations in the same order on the same values, so a
// column of the kernels below is BITWISE equal to the same column of mmvq_kernel / native_gu_kernel /
// native_down_kernel (mb-exp/iq-once/iq_once_parity checks it).  STRATA_IQ_ONCE=1 selects them; unset, "0", or
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels.  IQ1_M (29) and Q3_K (11) always take the old kernels.
template<int TY> struct Split;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const uint64_t *iq2xxs_grid = (const uint64_t *)grid;
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const sycl::uint2 grid_pos =
                ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
            r.g[k0 + 0] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.x() ^ signs0, signs0);
            const int signs1 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
            r.g[k0 + 1] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.y() ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = sycl::ext::intel::math::half2float(bq2->d);
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const uint64_t *iq2xs_grid = (const uint64_t *)grid;
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        // vec_dot_iq2_xs_q8_1 reads these four halves through a uint16_t* to a sycl::int2 (a strict-aliasing pun that
        // g++ -O2 miscompiles on the host); the shifts give the same four values (little-endian) without it.
        const int q2_lo = get_int_b2(bq2->qs, iqs + 0), q2_hi = get_int_b2(bq2->qs, iqs + 1);
        const uint16_t q2[4] = {(uint16_t) ((uint32_t) q2_lo & 0xFFFF), (uint16_t) ((uint32_t) q2_lo >> 16),
                                (uint16_t) ((uint32_t) q2_hi & 0xFFFF), (uint16_t) ((uint32_t) q2_hi >> 16)};
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::uint2 grid_pos =
                ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
            r.g[l0 + 0] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.x() ^ signs0, signs0);
            const int signs1 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
            r.g[l0 + 1] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.y() ^ signs1, signs1);
        }
        r.dw = sycl::ext::intel::math::half2float(bq2->d);
        return r;
    }
};
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const uint64_t *iq2s_grid = (const uint64_t *)grid;
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = sycl::ext::intel::math::vcmpne4<unsigned>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000);
            const int signs1 = sycl::ext::intel::math::vcmpne4<unsigned>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000);
            r.g[l0 + 0] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos[1] ^ signs1, signs1);
        }
        r.dw = sycl::ext::intel::math::half2float(bq2->d);
        return r;
    }
};
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const uint32_t *iq3xxs_grid = (const uint32_t *)grid;
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const sycl::int2 q3_packed =
            sycl::int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x08040201, 0);
            r.g[l0 + 0] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.x() ^ signs0, signs0);
            const int signs1 =
                sycl::ext::intel::math::vcmpne4<unsigned>(signs & 0x80402010, 0);
            r.g[l0 + 1] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.y() ^ signs1, signs1);
        }
        r.ls = aux32 >> 28;
        r.dw = sycl::ext::intel::math::half2float(bq3->d);
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const uint32_t *iq3s_grid = (const uint32_t *)grid;
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed =
            sycl::int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                           iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = sycl::ext::intel::math::vcmpne4<unsigned>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000);
            const int signs1 = sycl::ext::intel::math::vcmpne4<unsigned>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000);
            r.g[l0 + 0] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.x() ^ signs0, signs0);
            r.g[l0 + 1] = sycl::ext::intel::math::vsub4<unsigned>(
                grid_pos.y() ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::ext::intel::math::half2float(bq3->d);
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
template<> struct Split<20> {   // IQ4_NL
    struct W { sycl::int2 v[2]; float dw; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const int8_t *kvalues_iq4nl = (const int8_t *)grid;
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = sycl::ext::intel::math::half2float(bq4->d);
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x(), q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y(), q8[l + 4], sumi);
        }
        const float d = r.dw * bq8_1->ds[0];
        return d * sumi;
    }
};
template<> struct Split<23> {   // IQ4_XS
    struct W { sycl::int2 v[4]; int ls; float dw; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *grid) {
        const int8_t *kvalues_iq4nl = (const int8_t *)grid;
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::ext::intel::math::half2float(bq4->d);
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x(), u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * bq8_1[iqs / 4].ds[0];
        return d * sumi;
    }
};
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    static __dpct_inline__ W load(const void *__restrict__ vbq, int kbx, int iqs, const void *) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 0);
            const int qo = dpct::byte_level_permute(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static __dpct_inline__ float apply(const W &r, const block_q8_1 *__restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = bq8_1_chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};

// One row against the n <= NC activations x + off[0..n) (n >= 1, uniform over the sub-group; offsets in q8_1 blocks),
// the whole sub-group.  Per activation this is row_dot: the same calls k, lane-strided the same way, summed in the
// same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.  warp_sum runs for
// every c (no sub-group operation under a branch); the columns c >= n hold 0 and are not stored.
template <int TY, int NC>
__dpct_inline__ void row_dot_multi(const uint8_t *row, const block_q8_1 *x, const int (&off)[NC], int n, int nb,
                                   int lane, float (&s)[NC], const void *grid) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = warp_sum(s[c]);
}
// Two rows (gate and up) share one sub-group.  The first passes use lane and
// lane^16 for the two rows; the tail is split between the two half-warps.
// Thus each row receives the same terms in the same order as row_dot_multi,
// while the final tail pass keeps the original `acc + apply` expression.
template <int TY, int NC>
__dpct_inline__ void row_dot_pair(const uint8_t *rg, const uint8_t *ru, const block_q8_1 *x,
                                  const int (&off)[NC], int n, int nb, int lane, float (&s)[2 * NC],
                                  const void *grid) {
    using F = Fmt<TY>;
    using S = Split<TY>;
    const int P = nb * F::ipb / 32, lu = lane ^ 16;
#pragma unroll
    for (int c = 0; c < 2 * NC; ++c) s[c] = 0.0f;
    for (int p = 0; p < P; ++p) {
        {
            const int k = lane + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(rg, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
        {
            const int k = lu + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(ru, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) s[NC + c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    {
        const bool up = lane >= 16;
        const int k = 32 * P + (lane & 15), kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(up ? ru : rg, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < n) {
            const float a = up ? s[NC + c] : s[c];
            const float r = a + S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
            if (up) s[NC + c] = r; else s[c] = r;
        }
    }
    // Every lane computes the same per-row/per-column sum as its old lane
    // (lane^16 only permutes ownership); only the reduction layout changes.
    warp_sum_multi<2 * NC>(s, lane);
}


// mmvq_kernel with the columns taken NC at a time; lane 0 stores, as there.
template<int TY, int NC>
void mmvq_multi_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                       const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                       int n_out, int ncols,
                       const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int n = sycl::min(NC, ncols - c0);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (c0 + sycl::min(c, n - 1)) * xb;
        float s[NC];
        row_dot_multi<TY, NC>(wr, x, off, n, nb, lane, s, grid);
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) y[(size_t) (c0 + c) * n_out + row] = s[c];
        }
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.  A group
// has at most one entry per token of the window (--spec 4: <= 4), so 4 takes the window in one pass.
constexpr int GRP_NC = 4;
constexpr int GRP_NC_WIDE = 8;

template<int TG, int NC = GRP_NC>
void native_gu_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                            const int32_t* __restrict__ grp_start,
                            const int32_t* __restrict__ n_groups,
                            const int32_t* __restrict__ ent_tok,
                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                            float* __restrict__ gate, float* __restrict__ up,
                            const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * GU_ROWS + warp; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    float* dst = is_up ? up : gate;
    for (int e = e0; e < e1; e += NC) {
        const int n = sycl::min(NC, e1 - e);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
        float s[NC];
        row_dot_multi<TG, NC>(wr, xq, off, n, nb, lane, s, grid);
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
        }
    }
}
// One sub-group handles the matching gate/up rows of one expert.  Packing
// rows does not change any dot-product term or its per-row accumulation order;
// only the ownership of lane 16..31 is rotated to the paired row.
template<int TG, int NC = GRP_NC>
void native_gu_pair_kernel(const unsigned long long* __restrict__ grp_ptr,
                           const int32_t* __restrict__ grp_start,
                           const int32_t* __restrict__ n_groups,
                           const int32_t* __restrict__ ent_tok,
                           const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                           float* __restrict__ gate, float* __restrict__ up,
                           const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 8 + warp;
    if (r >= L.n_ff) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* rg = blob + (size_t) r * L.gu_row;
    const uint8_t* ru = blob + L.up_off + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += NC) {
        const int n = sycl::min(NC, e1 - e);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
        float s[2 * NC];
        row_dot_pair<TG, NC>(rg, ru, xq, off, n, nb, lane, s, grid);
        const int idx = warp_sum_multi_index<2 * NC>(lane);
        const int c = idx < NC ? idx : idx - NC;
        if ((lane & 3) == 0 && c < n) {
            float *dst = idx < NC ? gate : up;
            dst[(size_t) (e + c) * L.n_ff + r] = s[0];
        }
    }
}
// Four IQ4_NL rows share a sub-group.  The four full passes are lane^0,
// lane^8, lane^16, and lane^24; the final eight items are assigned by
// lane>>3.  The four explicit accumulators avoid a runtime row subscript in
// private memory and retain the old `acc + apply` expression for the tail.
template <int TY, int NC>
__dpct_inline__ void row_dot_quad(const uint8_t *r0, const uint8_t *r1, const uint8_t *r2,
                                  const uint8_t *r3, const block_q8_1 *x, const int (&off)[NC], int n,
                                  int nb, int lane, float (&v)[4 * NC], const void *grid) {
    using F = Fmt<TY>;
    using S = Split<TY>;
    const int P = nb * F::ipb / 32;
#pragma unroll
    for (int c = 0; c < 4 * NC; ++c) v[c] = 0.0f;
    for (int p = 0; p < P; ++p) {
        {
            const int k = lane + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(r0, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) v[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
        {
            const int k = (lane ^ 8) + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(r1, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) v[NC + c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
        {
            const int k = (lane ^ 16) + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(r2, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) v[2 * NC + c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
        {
            const int k = (lane ^ 24) + 32 * p, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(r3, kbx, iqs, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) v[3 * NC + c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    const int k = 32 * P + (lane & 7), kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
    if ((lane >> 3) == 0) {
        const typename S::W w = S::load(r0, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < n) {
            const float a = v[c];
            v[c] = a + S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    if ((lane >> 3) == 1) {
        const typename S::W w = S::load(r1, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < n) {
            const float a = v[NC + c];
            v[NC + c] = a + S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    if ((lane >> 3) == 2) {
        const typename S::W w = S::load(r2, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < n) {
            const float a = v[2 * NC + c];
            v[2 * NC + c] = a + S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    if ((lane >> 3) == 3) {
        const typename S::W w = S::load(r3, kbx, iqs, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < n) {
            const float a = v[3 * NC + c];
            v[3 * NC + c] = a + S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
    warp_sum_multi<4 * NC>(v, lane);
}

template<int TD, int NC = GRP_NC>
void native_down_quad_kernel(const unsigned long long* __restrict__ grp_ptr,
                             const int32_t* __restrict__ grp_start,
                             const int32_t* __restrict__ n_groups,
                             const int32_t* __restrict__ ent_dst,
                             const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                             float* __restrict__ out, const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 32 + 4 * warp;
    if (r + 3 >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* r0 = blob + L.down_off + (size_t) r * L.d_row;
    const uint8_t* r1 = r0 + L.d_row;
    const uint8_t* r2 = r1 + L.d_row;
    const uint8_t* r3 = r2 + L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += NC) {
        const int n = sycl::min(NC, e1 - e);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (e + sycl::min(c, n - 1)) * hb;
        float v[4 * NC];
        row_dot_quad<TD, NC>(r0, r1, r2, r3, hq, off, n, nb, lane, v, grid);
        const int idx = warp_sum_multi_index<4 * NC>(lane);
        const int row = idx >> 2, c = idx & 3;
        if ((lane & 1) == 0 && c < n)
            out[(size_t) ent_dst[e + c] * L.n_embd + r + row] = v[0];
    }
}



// native_down_kernel with the entries taken GRP_NC at a time
template<int TD, int NC = GRP_NC>
void native_down_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                              const int32_t* __restrict__ grp_start,
                              const int32_t* __restrict__ n_groups,
                              const int32_t* __restrict__ ent_dst,
                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                              float* __restrict__ out,
                              const void *grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int r = item_ct1.get_group(2) * 8 + warp;
    if (r >= L.n_embd) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += NC) {
        const int n = sycl::min(NC, e1 - e);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (e + sycl::min(c, n - 1)) * hb;
        float s[NC];
        row_dot_multi<TD, NC>(wr, hq, off, n, nb, lane, s, grid);
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
        }
    }
}

// STRATA_IQ_ONCE=1: the decode-once kernels; unset, "0", or STRATA_OLD_IQ_MMVQ=1 (upstream's name): the per-column
// ones.  Read once.  A captured SYCL graph keeps the kernels it captured, so iq_set_old_kernels() (below the
// anonymous namespace) is for the parity harness, not for a running session.
bool iq_once_env() {
    static const bool on = [] {
        const char *old = std::getenv("STRATA_OLD_IQ_MMVQ");
        if (old != nullptr && old[0] != '\0' && old[0] != '0') return false;
        const char *v = std::getenv("STRATA_IQ_ONCE");
        const bool r = v == nullptr || (v[0] != '\0' && v[0] != '0');
        if (r) std::fprintf(stderr, "strata iq: decode-once IQ kernels ON (STRATA_IQ_ONCE)\n");
        return r;
    }();
    return on;
}
bool iq_nc8_env() {
    static const bool on = [] {
        const char *v = std::getenv("STRATA_IQ_NC8");
        const bool r = v != nullptr && v[0] != '0';
        if (r) std::fprintf(stderr, "strata iq: 8-wide decode-once passes for windows > 4 (STRATA_IQ_NC8)\n");
        return r;
    }();
    return on;
}
std::atomic<int> g_iq_window{0};
}  // namespace

void iq_once_set_window(int n) {
    g_iq_window.store(n, std::memory_order_relaxed);
}

namespace {

static bool iq_wide() {
    return iq_nc8_env() && g_iq_window.load(std::memory_order_relaxed) > 4;
}
std::atomic<int> g_iq_pack_override{-1};   // -1: environment; 0: off; 1: packed path
bool iq_pack_env() {
    static const bool on = [] {
        const char *v = std::getenv("STRATA_IQ_PACK");
        const bool r = v != nullptr && v[0] == '1' && v[1] == '\0';
        if (r) std::fprintf(stderr, "strata iq: packed tail passes ON (STRATA_IQ_PACK)\n");
        return r;
    }();
    return on;
}
bool iq_pack_on() {
    const int o = g_iq_pack_override.load(std::memory_order_relaxed);
    return o < 0 ? iq_pack_env() : o == 1;
}

std::atomic<int> g_iq_old_override{-1};   // -1: the environment decides; 0: decode-once; 1: per-column
bool iq_once() {
    const int o = g_iq_old_override.load(std::memory_order_relaxed);
    return o < 0 ? iq_once_env() : o == 0;
}

// The format's table on this queue's device (get_ptr(q), not get_ptr(): no default queue, patch #6)
const void *iq_once_grid(int t, sycl::queue &q) {
    switch (t) {
    case 16: return iq2xxs_grid.get_ptr(q);
    case 17: return iq2xs_grid.get_ptr(q);
    case 18: return iq3xxs_grid.get_ptr(q);
    case 20: case 23: return kvalues_iq4nl.get_ptr(q);
    case 21: return iq3s_grid.get_ptr(q);
    case 22: return iq2s_grid.get_ptr(q);
    default: return nullptr;   // 42 (Q2_0) reads no table
    }
}

template<int TY, int NC>
void iq_once_mmvq_submit(const uint8_t *W, size_t rb, const block_q8_1 *X, float *y, int n_in, int n_out, int ncols,
                         const void *grid, dpct::queue_ptr s) {
    const dpct::dim3 grid3((unsigned)((n_out + 3) / 4)), block(32, 4);
    s->parallel_for(sycl::nd_range<3>(grid3 * block, block),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        mmvq_multi_kernel<TY, NC>(W, rb, X, y, n_in, n_out, ncols, grid);
                    });
}

// 1, 2 or 4 columns per pass (3 takes the 4-wide pass); more than 4 go 4 at a time unless STRATA_IQ_NC8
// enables the 8-wide pass.  Upstream also has an 8-wide pass; it remains opt-in because 8 columns of state is
// where Xe would spill first.
template<int TY>
void iq_once_mmvq_t(const uint8_t *W, size_t rb, const block_q8_1 *X, float *y, int n_in, int n_out, int ncols,
                    dpct::queue_ptr s) {
    const void *grid = iq_once_grid(TY, *s);
    if (ncols <= 1) iq_once_mmvq_submit<TY, 1>(W, rb, X, y, n_in, n_out, ncols, grid, s);
    else if (ncols == 2) iq_once_mmvq_submit<TY, 2>(W, rb, X, y, n_in, n_out, ncols, grid, s);
    else if (ncols > 4 && iq_nc8_env()) iq_once_mmvq_submit<TY, 8>(W, rb, X, y, n_in, n_out, ncols, grid, s);
    else iq_once_mmvq_submit<TY, 4>(W, rb, X, y, n_in, n_out, ncols, grid, s);
}

// true: launched; false: the caller's per-column switch takes t
bool iq_once_mmvq(int t, const uint8_t *W, size_t rb, const block_q8_1 *X, float *y, int n_in, int n_out, int ncols,
                  dpct::queue_ptr s) {
    switch (t) {
    case 16: iq_once_mmvq_t<16>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 17: iq_once_mmvq_t<17>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 18: iq_once_mmvq_t<18>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 20: iq_once_mmvq_t<20>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 21: iq_once_mmvq_t<21>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 22: iq_once_mmvq_t<22>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 23: iq_once_mmvq_t<23>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    case 42: iq_once_mmvq_t<42>(W, rb, X, y, n_in, n_out, ncols, s); return true;
    default: return false;   // 29 (IQ1_M)
    }
}

template<int TG>
void iq_once_gu_submit(const dpct::dim3 &ggu, int64_t cap_groups,
                       const unsigned long long *grp_ptr, const int32_t *grp_start,
                       const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X, NativeExpertLayout L,
                       float *gate, float *up, dpct::queue_ptr s) {
    const void *grid = iq_once_grid(TG, *s);
    if (iq_pack_on() && !iq_wide() &&
        (L.n_embd / Fmt<TG>::qk) * Fmt<TG>::ipb % 32 == 16) {
        const dpct::dim3 gd((unsigned)((L.n_ff + 7) / 8), (unsigned)cap_groups);
        s->parallel_for(sycl::nd_range<3>(gd * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_gu_pair_kernel<TG>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, grid);
                        });
        return;
    }
    if (iq_wide()) {
        s->parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_gu_multi_kernel<TG, GRP_NC_WIDE>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up,
                                                                    grid);
                        });
    } else {
        s->parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_gu_multi_kernel<TG>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, grid);
                        });
    }
}

bool iq_once_gu(const NativeExpertLayout &L, const dpct::dim3 &ggu, int64_t cap_groups,
                const unsigned long long *grp_ptr, const int32_t *grp_start,
                const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                float *gate, float *up, dpct::queue_ptr s) {
    switch (L.gu_type) {
    case 16: iq_once_gu_submit<16>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 17: iq_once_gu_submit<17>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 18: iq_once_gu_submit<18>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 21: iq_once_gu_submit<21>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 22: iq_once_gu_submit<22>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 23: iq_once_gu_submit<23>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    case 42: iq_once_gu_submit<42>(ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, s); return true;
    default: return false;   // 29 (IQ1_M)
    }
}


template<int TD>
void iq_once_down_submit(const unsigned long long *grp_ptr, const int32_t *grp_start, const int32_t *n_groups,
                         const int32_t *ent_dst, const block_q8_1 *hq, NativeExpertLayout L, float *out,
                         int64_t cap_groups, dpct::queue_ptr s) {
    const void *grid = iq_once_grid(TD, *s);
    const dpct::dim3 gd((unsigned)((L.n_embd + 7) / 8), (unsigned)cap_groups);
    if (iq_pack_on() && !iq_wide() && TD == 20 && (L.n_ff / 32) * 2 % 32 == 8) {
        const dpct::dim3 gdq((unsigned)(L.n_embd / 32), (unsigned)cap_groups);
        s->parallel_for(sycl::nd_range<3>(gdq * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_down_quad_kernel<20>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, grid);
                        });
        return;
    }

    if (iq_wide()) {
        s->parallel_for(sycl::nd_range<3>(gd * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_down_multi_kernel<TD, GRP_NC_WIDE>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out,
                                                                      grid);
                        });
    } else {
        s->parallel_for(sycl::nd_range<3>(gd * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                            native_down_multi_kernel<TD>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, grid);
                        });
    }
}

bool iq_once_down(const NativeExpertLayout &L, const unsigned long long *grp_ptr, const int32_t *grp_start,
                  const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq, float *out,
                  int64_t cap_groups, dpct::queue_ptr s) {
    switch (L.d_type) {
    case 20: iq_once_down_submit<20>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, cap_groups, s); return true;
    case 23: iq_once_down_submit<23>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, cap_groups, s); return true;
    case 42: iq_once_down_submit<42>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, cap_groups, s); return true;
    default: return false;
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float xi = x[i];
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(
            amax,
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
        sum += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = sycl::half2(d, sum);
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template <typename dst_t> __dpct_inline__ dst_t cvt(float v);
template <> __dpct_inline__ float cvt<float>(float v) { return v; }
template <>
__dpct_inline__ sycl::half cvt<sycl::half>(float v) {
    return sycl::ext::intel::math::float2half_rn(v);
}
template <typename dst_t, bool VEC>
__dpct_inline__ void dq_store8(dst_t *y, const dst_t (&o)[8]) {
    if constexpr (VEC) {
        sycl::vec<sycl::half, 8> v;
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = o[j];
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(y) = v;
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) y[j] = o[j];
    }
}

template <typename dst_t, bool VEC>
__dpct_inline__ void dq_store_iq4(dst_t *y, const dst_t (&o)[8]) {
    if constexpr (VEC) {
        sycl::vec<sycl::half, 4> lo, hi;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo[j] = o[j];
            hi[j] = o[j + 4];
        }
        *reinterpret_cast<sycl::vec<sycl::half, 4> *>(y) = lo;
        *reinterpret_cast<sycl::vec<sycl::half, 4> *>(y + 16) = hi;
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = o[j + 0];
            y[j + 16] = o[j + 4];
        }
    }
}


template<typename dst_t, bool VEC = false>
void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid,
                const uint8_t *kmask_iq2xs, const uint8_t *ksigns_iq2xs,
                const uint64_t *iq2xxs_grid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    dst_t o[8];
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) o[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    dq_store8<dst_t, VEC>(y, o);
}
template<typename dst_t, bool VEC = false>
void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid,
               const uint8_t *kmask_iq2xs, const uint8_t *ksigns_iq2xs,
               const uint64_t *iq2xs_grid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    dst_t o[8];
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) o[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    dq_store8<dst_t, VEC>(y, o);
}

template<typename dst_t, bool VEC = false>
void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid,
              const uint8_t *kmask_iq2xs, const uint64_t *iq2s_grid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    dst_t o[8];
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) o[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    dq_store8<dst_t, VEC>(y, o);
}

template<typename dst_t, bool VEC = false>
void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid,
                const uint8_t *kmask_iq2xs, const uint8_t *ksigns_iq2xs,
                const uint32_t *iq3xxs_grid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    dst_t o[8];
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        o[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        o[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
    dq_store8<dst_t, VEC>(y, o);
}

template<typename dst_t, bool VEC = false>
void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid,
              const uint8_t *kmask_iq2xs, const uint32_t *iq3s_grid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    dst_t o[8];
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        o[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        o[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
    dq_store8<dst_t, VEC>(y, o);
}
template<typename dst_t>
void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid,
              const uint32_t *iq1s_grid_gpu) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t, bool VEC = false>
void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid,
               const int8_t *kvalues_iq4nl) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    dst_t o[8];
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        o[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        o[j + 4] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
    dq_store_iq4<dst_t, VEC>(y, o);
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid,
               const int8_t *kvalues_iq4nl) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t, bool VEC = false>
void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    dst_t o[8];
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        o[j] = cvt<dst_t>(d * (float) (code - 1));
    }
    dq_store8<dst_t, VEC>(yy + b * 64 + part * 8, o);
}

template <typename dst_t>
__dpct_inline__ void
dq_dispatch(int ty, const void *vx, int64_t ibs, dst_t *y, int tid,
            uint8_t *kmask_iq2xs, uint8_t *ksigns_iq2xs, uint64_t *iq2xxs_grid,
            uint64_t *iq2xs_grid, uint64_t *iq2s_grid, uint32_t *iq3xxs_grid,
            uint32_t *iq3s_grid, int8_t *kvalues_iq4nl,
            uint32_t *iq1s_grid_gpu) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs,
                            iq2xxs_grid);
            break;
        case 17: dq_iq2_xs(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs,
                           iq2xs_grid);
            break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs,
                            iq3xxs_grid);
            break;
        case 20: dq_iq4_nl(vx, ibs, y, tid, kvalues_iq4nl); break;
        case 21: dq_iq3_s(vx, ibs, y, tid, kmask_iq2xs, iq3s_grid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid, kmask_iq2xs, iq2s_grid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid, iq1s_grid_gpu); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid, kvalues_iq4nl); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}
template <typename dst_t>
__dpct_inline__ void
dq_dispatch_vec(int ty, const void *vx, int64_t ibs, dst_t *y, int tid,
                uint8_t *kmask_iq2xs, uint8_t *ksigns_iq2xs, uint64_t *iq2xxs_grid,
                uint64_t *iq2xs_grid, uint64_t *iq2s_grid, uint32_t *iq3xxs_grid,
                uint32_t *iq3s_grid, int8_t *kvalues_iq4nl,
                uint32_t *iq1s_grid_gpu) {
    switch (ty) {
    case 16: dq_iq2_xxs<dst_t, true>(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs, iq2xxs_grid); break;
    case 17: dq_iq2_xs<dst_t, true>(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs, iq2xs_grid); break;
    case 18: dq_iq3_xxs<dst_t, true>(vx, ibs, y, tid, kmask_iq2xs, ksigns_iq2xs, iq3xxs_grid); break;
    case 20: dq_iq4_nl<dst_t, true>(vx, ibs, y, tid, kvalues_iq4nl); break;
    case 21: dq_iq3_s<dst_t, true>(vx, ibs, y, tid, kmask_iq2xs, iq3s_grid); break;
    case 22: dq_iq2_s<dst_t, true>(vx, ibs, y, tid, kmask_iq2xs, iq2s_grid); break;
    case 42: dq_q2_0<dst_t, true>(vx, ibs, y, tid); break;
    default: break;
    }
}

bool dq_vec_type(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 42;
}
bool dq_vec_on() {
    static const bool on = [] {
        const char *v = std::getenv("STRATA_DQ_VEC");
        const bool r = v != nullptr && v[0] == '1' && v[1] == '\0';
        if (r) std::fprintf(stderr, "strata iq: vector dequant stores ON (STRATA_DQ_VEC)\n");
        return r;
    }();
    return on;
}


// flat: superblock i -> y + 256 i
template<typename dst_t>
void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y,
                         uint8_t *kmask_iq2xs, uint8_t *ksigns_iq2xs,
                         uint64_t *iq2xxs_grid, uint64_t *iq2xs_grid,
                         uint64_t *iq2s_grid, uint32_t *iq3xxs_grid,
                         uint32_t *iq3s_grid, int8_t *kvalues_iq4nl,
                         uint32_t *iq1s_grid_gpu) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, item_ct1.get_local_id(2),
                       kmask_iq2xs, ksigns_iq2xs, iq2xxs_grid, iq2xs_grid,
                       iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl,
                       iq1s_grid_gpu);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
void dequant_gu_kernel(int ty, const void *__restrict__ gate,
                       const void *__restrict__ up, int64_t per_row,
                       sycl::half *__restrict__ y, uint8_t *kmask_iq2xs,
                       uint8_t *ksigns_iq2xs, uint64_t *iq2xxs_grid,
                       uint64_t *iq2xs_grid, uint64_t *iq2s_grid,
                       uint32_t *iq3xxs_grid, uint32_t *iq3s_grid,
                       int8_t *kvalues_iq4nl, uint32_t *iq1s_grid_gpu) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = item_ct1.get_group(2);
    const int parity = item_ct1.get_group(1);
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<sycl::half>(ty, parity ? up : gate, i,
                            y + ((2 * r + parity) * per_row + c) * QK_K,
                            item_ct1.get_local_id(2), kmask_iq2xs, ksigns_iq2xs,
                            iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid,
                            iq3s_grid, kvalues_iq4nl, iq1s_grid_gpu);
}
// Vector path: one 256-thread work-group covers eight superblocks.  The
// tid-to-element mapping remains the old 32-thread mapping; only the store
// width and the work-group packing change.
// Every element is evaluated by the unchanged cvt expression before the
// vector store, so no arithmetic, accumulation, or half-rounding order changes.

template<typename dst_t>
void dequant_flat_vec_kernel(int ty, const void* __restrict__ vx, int64_t nsb,
                             dst_t* __restrict__ y, uint8_t *kmask_iq2xs,
                             uint8_t *ksigns_iq2xs, uint64_t *iq2xxs_grid,
                             uint64_t *iq2xs_grid, uint64_t *iq2s_grid,
                             uint32_t *iq3xxs_grid, uint32_t *iq3s_grid,
                             int8_t *kvalues_iq4nl, uint32_t *iq1s_grid_gpu) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lid = item_ct1.get_local_id(2);
    const int64_t sb = (int64_t) item_ct1.get_group(2) * 8 + lid / 32;
    if (sb >= nsb) return;
    dq_dispatch_vec<dst_t>(ty, vx, sb, y + sb * QK_K, lid % 32,
                           kmask_iq2xs, ksigns_iq2xs, iq2xxs_grid, iq2xs_grid,
                           iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl,
                           iq1s_grid_gpu);
}

template<typename dst_t>
void dequant_gu_vec_kernel(int ty, const void *__restrict__ gate,
                           const void *__restrict__ up, int64_t per_row, int64_t nsb,
                           dst_t *__restrict__ y, uint8_t *kmask_iq2xs,
                           uint8_t *ksigns_iq2xs, uint64_t *iq2xxs_grid,
                           uint64_t *iq2xs_grid, uint64_t *iq2s_grid,
                           uint32_t *iq3xxs_grid, uint32_t *iq3s_grid,
                           int8_t *kvalues_iq4nl, uint32_t *iq1s_grid_gpu) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lid = item_ct1.get_local_id(2);
    const int64_t sb = (int64_t) item_ct1.get_group(2) * 8 + lid / 32;
    if (sb >= nsb) return;
    const int parity = item_ct1.get_group(1);
    const int64_t r = sb / per_row, c = sb % per_row;
    dq_dispatch_vec<dst_t>(ty, parity ? up : gate, sb,
                           y + ((2 * r + parity) * per_row + c) * QK_K, lid % 32,
                           kmask_iq2xs, ksigns_iq2xs, iq2xxs_grid, iq2xs_grid,
                           iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl,
                           iq1s_grid_gpu);
}


bool is_iq(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11; }

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

// The parity harness's switch (mb-exp/iq-once/iq_once_parity.cpp; upstream's names): true = the per-column kernels.
// Not for a running session: a captured SYCL graph keeps the kernels it captured.
void iq_set_old_kernels(bool old) { g_iq_old_override.store(old ? 1 : 0, std::memory_order_relaxed); }
bool iq_old_kernels() { return !iq_once(); }


size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) *
                               sycl::range<3>(1, 1, 256),
                           sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1)
                           [[sycl::reqd_sub_group_size(32)]] {
                               quantize_q8_1_kernel(x, (block_q8_1 *)y, n);
                           });
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dpct::dim3 grid((unsigned)((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    dpct::queue_ptr s = (dpct::queue_ptr)stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    const bool once = iq_once() && iq_once_mmvq(t, W, rb, X, y, n_in, n_out, ncols, s);
    if (!once) switch (t) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 16: {
        iq2xxs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<16>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 17: {
        iq2xs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2xs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<17>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 18: {
        iq3xxs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<18>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 20: {
        kvalues_iq4nl.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = kvalues_iq4nl.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<20>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 21: {
        iq3s_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq3s_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<21>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 22: {
        iq2s_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2s_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<22>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 23: {
        kvalues_iq4nl.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = kvalues_iq4nl.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<23>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 29: {
        iq1s_grid_gpu.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq1s_grid_gpu.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<29>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 42: {
        const void *iq2xxs_grid_ptr_ct1 = nullptr;

        s->submit([&](sycl::handler &cgh) {
            const void *iq2xxs_grid_ptr_ct1 = nullptr;

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     mmvq_kernel<42>(W, rb, X, y, n_in, n_out,
                                                     ncols,
                                                     iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    {
        kmask_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        ksigns_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        kvalues_iq4nl.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq1s_grid_gpu.init(*((sycl::queue *)((dpct::queue_ptr)stream)));

        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp16});
        // Every vector destination is 16B aligned in the normal ring/borrow
        // layout.  If that invariant is not true, retain the scalar kernel.
        const bool vec = dq_vec_on() && dq_vec_type(t) && (((uintptr_t) dst & 15) == 0);


        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&, vec](sycl::handler &cgh) {
                auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();
                auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
                auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();
                auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

                if (vec) {
                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, (unsigned)((n / 256 + 7) / 8)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            dequant_flat_vec_kernel<sycl::half>(
                                t, src, n / 256, (sycl::half *)dst,
                                kmask_iq2xs_ptr_ct1, ksigns_iq2xs_ptr_ct1,
                                iq2xxs_grid_ptr_ct1, iq2xs_grid_ptr_ct1,
                                iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1,
                                iq1s_grid_gpu_ptr_ct1);
                        });
                } else {
                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, (unsigned)(n / 256)) *
                                sycl::range<3>(1, 1, 32),
                            sycl::range<3>(1, 1, 32)),
                        [=](sycl::nd_item<3> item_ct1) {
                            dequant_flat_kernel<sycl::half>(
                                t, src, (sycl::half *)dst, kmask_iq2xs_ptr_ct1,
                                ksigns_iq2xs_ptr_ct1, iq2xxs_grid_ptr_ct1,
                                iq2xs_grid_ptr_ct1, iq2s_grid_ptr_ct1,
                                iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                                kvalues_iq4nl_ptr_ct1, iq1s_grid_gpu_ptr_ct1);
                        });
                }
            });
    }
    check("iq_dequant_f16");
}

namespace {
void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y,
                                  uint8_t *kmask_iq2xs, uint8_t *ksigns_iq2xs,
                                  uint64_t *iq2xxs_grid, uint64_t *iq2xs_grid,
                                  uint64_t *iq2s_grid, uint32_t *iq3xxs_grid,
                                  uint32_t *iq3s_grid, int8_t *kvalues_iq4nl,
                                  uint32_t *iq1s_grid_gpu) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t b = item_ct1.get_group(2);
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t)t * n_embd + b * QK_K,
                       item_ct1.get_local_id(2), kmask_iq2xs, ksigns_iq2xs,
                       iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid,
                       iq3s_grid, kvalues_iq4nl, iq1s_grid_gpu);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    {
        kmask_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        ksigns_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        kvalues_iq4nl.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq1s_grid_gpu.init(*((sycl::queue *)((dpct::queue_ptr)stream)));

        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp16});

        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&](sycl::handler &cgh) {
                auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();
                auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
                auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();
                auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, (unsigned)n_tok,
                                                     (unsigned)(n_embd / 256)) *
                                          sycl::range<3>(1, 1, 32),
                                      sycl::range<3>(1, 1, 32)),
                    [=](sycl::nd_item<3> item_ct1) {
                        embed_rows_kernel(
                            t, (const uint8_t *)table, row_bytes, tokens,
                            n_embd, out, kmask_iq2xs_ptr_ct1,
                            ksigns_iq2xs_ptr_ct1, iq2xxs_grid_ptr_ct1,
                            iq2xs_grid_ptr_ct1, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, iq1s_grid_gpu_ptr_ct1);
                    });
            });
    }
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    {
        kmask_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        ksigns_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        kvalues_iq4nl.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq1s_grid_gpu.init(*((sycl::queue *)((dpct::queue_ptr)stream)));

        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp16});

        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&](sycl::handler &cgh) {
                auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();
                auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
                auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();
                auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, (unsigned)(n / 256)) *
                            sycl::range<3>(1, 1, 32),
                        sycl::range<3>(1, 1, 32)),
                    [=](sycl::nd_item<3> item_ct1) {
                        dequant_flat_kernel<float>(
                            t, src, dst, kmask_iq2xs_ptr_ct1,
                            ksigns_iq2xs_ptr_ct1, iq2xxs_grid_ptr_ct1,
                            iq2xs_grid_ptr_ct1, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, iq1s_grid_gpu_ptr_ct1);
                    });
            });
    }
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int64_t per_row = n_embd / 256;
    {
        kmask_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        ksigns_iq2xs.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2xs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq2s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3xxs_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq3s_grid.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        kvalues_iq4nl.init(*((sycl::queue *)((dpct::queue_ptr)stream)));
        iq1s_grid_gpu.init(*((sycl::queue *)((dpct::queue_ptr)stream)));

        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp16});
        // The vector path is safe only for the 16B-aligned destination ring;
        // otherwise the unchanged scalar launch remains the fallback.
        const bool vec = dq_vec_on() && dq_vec_type(t) && (((uintptr_t) dst & 15) == 0);


        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&, vec](sycl::handler &cgh) {
                auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();
                auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
                auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
                auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();
                auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

                if (vec) {
                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 2, (unsigned)((n_ff * per_row + 7) / 8)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            dequant_gu_vec_kernel<sycl::half>(
                                t, gate, up, per_row, n_ff * per_row, (sycl::half *)dst,
                                kmask_iq2xs_ptr_ct1, ksigns_iq2xs_ptr_ct1,
                                iq2xxs_grid_ptr_ct1, iq2xs_grid_ptr_ct1,
                                iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1,
                                iq1s_grid_gpu_ptr_ct1);
                        });
                } else {
                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 2, (unsigned)(n_ff * per_row)) *
                                sycl::range<3>(1, 1, 32),
                            sycl::range<3>(1, 1, 32)),
                        [=](sycl::nd_item<3> item_ct1) {
                            dequant_gu_kernel(
                                t, gate, up, per_row, (sycl::half *)dst,
                                kmask_iq2xs_ptr_ct1, ksigns_iq2xs_ptr_ct1,
                                iq2xxs_grid_ptr_ct1, iq2xs_grid_ptr_ct1,
                                iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1,
                                iq1s_grid_gpu_ptr_ct1);
                        });
                }
            });
    }
    check("iq_dequant_gu_f16");
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

// ---------------------------------------------------------------- Q2_0 down, grouped, staged (upstream 1d0c6e1's idea)
// STRATA_Q20_GROUPED=1: the expert down projection in GGUF Q2_0 (d_type 42) takes native_down_q20_kernel instead of
// native_down_multi_kernel<42>.  Unset or "0": unchanged.  It is upstream 1d0c6e1's down_grouped_t_kernel carried
// over from the S2 blob to the GGUF block (18 B: fp16 d, 16 B of 2-bit codes; q8_1 activations of 36 B):
//
//  - a work-group stages each entry's q8_1 chunk ONCE in local memory, regrouped so the codes expand with
//    shift + mask, with (dx, hx = sum of the chunk's 32 int8) beside it; the 8 sub-groups all read it from there;
//  - each sub-group takes TWO adjacent rows, so every staged read serves both;
//  - per (row, chunk) the 2-bit codes expand with 8 shift+and instead of 16 byte_level_permute + 8 permute.
//
// BITWISE THE SAME AS native_down_multi_kernel<42> / native_down_kernel<42>:  vec_dot_q2_0_q8_1 computes
// sumi = sum_e (code_e - 1) * x_e exactly in int; here s - hx = sum_e code_e * x_e - sum_e x_e is that integer.
// The float work is the same expression in the same order (0.0f, += d2 * d8 * (float) sumi), lane c holds chunk c as
// there (n_ff / 32 <= 32: one chunk per lane), and the reduction is the same warp_sum.  mb-exp/q20/q20_parity checks it.
namespace {

constexpr int Q20_GMAX = 8;                    // entries staged per pass (a group of more takes more passes)
constexpr int Q20_STRIDE = Q20_GMAX * 32;      // local words per regrouped word index j: [j][k * 32 + c]
constexpr int Q20_ROWS = 16;                   // rows per work-group: 8 sub-groups x 2 rows

// A chunk's two code words as the eight dp4a operands that pair with X[0..7]: byte b of m[4h + f] is the code of
// element 16h + 4b + f (codes 0..3, unsigned).  GGUF Q2_0 stores element e at byte e / 4, bits 2 * (e % 4).
__dpct_inline__ void q20_expand(uint32_t lo, uint32_t hi, int (&m)[8]) {
    const uint32_t M = 0x03030303u;
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        m[f] = (int) ((lo >> (2 * f)) & M);
        m[4 + f] = (int) ((hi >> (2 * f)) & M);
    }
}

// The 32 int8 of a q8_1 chunk regrouped: X[4h + f] = { x[16h + f], x[16h + 4 + f], x[16h + 8 + f], x[16h + 12 + f] }.
// Shifts and masks, not byte_level_permute (emulated per byte by dpct).  Returns hx = sum of the 32 int8.
__dpct_inline__ int q20_load_x(const block_q8_1 *__restrict__ blk, int (&X)[8]) {
    uint32_t n[8];
    int hx = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        n[j] = (uint32_t) get_int_b4(blk->qs, j);
        hx = ggml_cuda_dp4a((int) n[j], 0x01010101, hx);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            uint32_t w = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) w |= ((n[4 * h + b] >> (8 * f)) & 0xFFu) << (8 * b);
            X[4 * h + f] = (int) w;
        }
    return hx;
}

__dpct_inline__ int q20_chunk_s(const int (&m)[8], const int (&X)[8]) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = ggml_cuda_dp4a(X[j], m[j], s);
    return s;
}

// One work-group = Q20_ROWS down rows of ONE group; sub-group w takes rows r, r + 1 (r = row0 + 2w).  Every work-item
// reaches every barrier and every warp_sum (rows past n_embd compute nothing and store nothing).
template<bool PACK>
void native_down_q20_kernel(const unsigned long long *__restrict__ grp_ptr, const int32_t *__restrict__ grp_start,
                            const int32_t *__restrict__ n_groups, const int32_t *__restrict__ ent_dst,
                            const block_q8_1 *__restrict__ hq, NativeExpertLayout L, float *__restrict__ out,
                            int *__restrict__ xs_w, sycl::int2 *__restrict__ xs_dh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(1);
    if (g >= *n_groups) return;                                  // the whole work-group: before any barrier
    const int t = item_ct1.get_local_id(2), warp = t >> 5, lane = t & 31;
    const int NC = (int) (L.n_ff / 32);                          // chunks per entry row (host: <= 32, even)
    const int r = item_ct1.get_group(2) * Q20_ROWS + 2 * warp;
    const bool live = r < (int) L.n_embd && lane < NC;           // host: n_embd even, so r + 1 < n_embd too
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    int m0[8], m1[8];
    float dw0 = 0.0f, dw1 = 0.0f;
    if (live) {
        const int c = lane;
        const block_q2_0* b0 = (const block_q2_0*) (blob + L.down_off + (size_t) r * L.d_row) + (c >> 1);
        const block_q2_0* b1 = (const block_q2_0*) (blob + L.down_off + (size_t) (r + 1) * L.d_row) + (c >> 1);
        const uint16_t* q0 = (const uint16_t*) b0->qs + (c & 1) * 4;   // as vec_dot_q2_0_q8_1: qs + iqs * 4
        const uint16_t* q1 = (const uint16_t*) b1->qs + (c & 1) * 4;
        q20_expand((uint32_t) q0[0] | (uint32_t) q0[1] << 16, (uint32_t) q0[2] | (uint32_t) q0[3] << 16, m0);
        q20_expand((uint32_t) q1[0] | (uint32_t) q1[1] << 16, (uint32_t) q1[2] | (uint32_t) q1[3] << 16, m1);
        dw0 = b0->d;
        dw1 = b1->d;
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) m0[j] = m1[j] = 0;
    }
    const int hb = NC;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int eb = e0; eb < e1; eb += Q20_GMAX) {
        const int ne = sycl::min(Q20_GMAX, e1 - eb);
        sycl::group_barrier(item_ct1.get_group());               // the previous pass is done reading
        for (int i = t; i < ne * NC; i += 256) {
            const int k = i / NC, c = i - k * NC, at = k * 32 + c;
            const block_q8_1* blk = hq + (size_t) (eb + k) * hb + c;
            int X[8];
            const int hx = q20_load_x(blk, X);
#pragma unroll
            for (int j = 0; j < 8; ++j) xs_w[j * Q20_STRIDE + at] = X[j];
            const float d8 = blk->ds[0];
            xs_dh[at] = sycl::int2(sycl::bit_cast<int>(d8), hx);
        }
        sycl::group_barrier(item_ct1.get_group());
        if constexpr (PACK) {
            // PACK interleaves independent entries only.  Each acc keeps the
            // existing expression and entry order; warp_sum_multi performs
            // the same XOR pairs, so no floating-point rounding changes.
            float v[2 * Q20_GMAX];
#pragma unroll
            for (int k = 0; k < Q20_GMAX; ++k) {
                float acc0 = 0.0f, acc1 = 0.0f;
                if (k < ne && live) {
                    const int at = k * 32 + lane;
                    int X[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) X[j] = xs_w[j * Q20_STRIDE + at];
                    const sycl::int2 dh = xs_dh[at];
                    const float d8 = sycl::bit_cast<float>(dh.x());
                    acc0 += dw0 * d8 * (float) (q20_chunk_s(m0, X) - dh.y());
                    acc1 += dw1 * d8 * (float) (q20_chunk_s(m1, X) - dh.y());
                }
                v[k] = acc0;
                v[Q20_GMAX + k] = acc1;
            }
            warp_sum_multi<2 * Q20_GMAX>(v, lane);
            const int idx = warp_sum_multi_index<2 * Q20_GMAX>(lane);
            const int row = idx >> 3, k = idx & 7;
            if ((lane & 1) == 0 && k < ne && r < (int) L.n_embd) {
                const size_t o = (size_t) ent_dst[eb + k] * L.n_embd + (size_t) r + row;
                out[o] = v[0];
            }
        } else {
            for (int k = 0; k < ne; ++k) {
                float acc0 = 0.0f, acc1 = 0.0f;
                if (live) {
                    const int at = k * 32 + lane;
                    int X[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) X[j] = xs_w[j * Q20_STRIDE + at];
                    const sycl::int2 dh = xs_dh[at];
                    const float d8 = sycl::bit_cast<float>(dh.x());
                    acc0 += dw0 * d8 * (float) (q20_chunk_s(m0, X) - dh.y());
                    acc1 += dw1 * d8 * (float) (q20_chunk_s(m1, X) - dh.y());
                }
                const float s0 = warp_sum(acc0);
                const float s1 = warp_sum(acc1);
                if (lane == 0 && r < (int) L.n_embd) {
                    const size_t o = (size_t) ent_dst[eb + k] * L.n_embd + (size_t) r;
                    out[o] = s0;
                    out[o + 1] = s1;
                }
            }
        }
    }
}

// STRATA_Q20_GROUPED=0 selects the old path; unset or nonzero enables grouped Q2_0.  Read once.
std::atomic<int> g_q20_override{-1};   // -1: the environment; 0: off; 1: on (q20_set_mode, for the parity harness)
bool q20_on() {
    const int o = g_q20_override.load(std::memory_order_relaxed);
    if (o >= 0) return o == 1;
    static const bool on = [] {
        const char *v = std::getenv("STRATA_Q20_GROUPED");
        const bool r = v == nullptr || (v[0] != '\0' && v[0] != '0');
        if (r) std::fprintf(stderr, "strata q20: staged Q2_0 grouped down ON (STRATA_Q20_GROUPED)\n");
        return r;
    }();
    return on;
}
template<bool PACK>
void q20_submit(const unsigned long long *grp_ptr, const int32_t *grp_start, const int32_t *n_groups,
                const int32_t *ent_dst, const block_q8_1 *hq, NativeExpertLayout L, float *out,
                int64_t cap_groups, dpct::queue_ptr s) {
    const dpct::dim3 gd((unsigned) ((L.n_embd + Q20_ROWS - 1) / Q20_ROWS), (unsigned) cap_groups);
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> xs_w(sycl::range<1>(8 * Q20_STRIDE), cgh);     // 8 KB
        sycl::local_accessor<sycl::int2, 1> xs_dh(sycl::range<1>(Q20_STRIDE), cgh);  // 2 KB
        cgh.parallel_for(sycl::nd_range<3>(gd * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                             native_down_q20_kernel<PACK>(
                                 grp_ptr, grp_start, n_groups, ent_dst, hq, L, out,
                                 xs_w.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 xs_dh.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}


// true: launched; false: the caller's path takes it (switch off, not Q2_0, or a shape the kernel does not cover)
bool q20_down(const NativeExpertLayout &L, const unsigned long long *grp_ptr, const int32_t *grp_start,
              const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq, float *out, int64_t cap_groups,
              dpct::queue_ptr s) {
    if (!q20_on() || L.d_type != 42) return false;
    if (L.n_ff % 64 != 0 || L.n_ff / 32 > 32 || L.n_embd % 2 != 0) return false;
    if (((uintptr_t) hq & 3) != 0) return false;                 // get_int_b4 on qs needs a 4-byte aligned row
    if (iq_pack_on())
        q20_submit<true>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, cap_groups, s);
    else
        q20_submit<false>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, cap_groups, s);
    return true;
}

}  // namespace

// The parity harness's switch (mb-exp/q20/q20_parity.cpp): -1 = the environment, 0 = off, 1 = on.
void q20_set_mode(int mode) { g_q20_override.store(mode < 0 ? -1 : (mode != 0 ? 1 : 0), std::memory_order_relaxed); }
// The IQ-pack parity harness's switch: -1 = environment, 0 = off, 1 = packed path.
void iq_pack_set_mode(int mode) { g_iq_pack_override.store(mode < 0 ? -1 : (mode != 0 ? 1 : 0), std::memory_order_relaxed); }


void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    dpct::queue_ptr s = (dpct::queue_ptr)stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dpct::dim3 ggu((unsigned)((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS),
                         (unsigned)cap_groups);
    const bool gu_once =
        iq_once() && iq_once_gu(L, ggu, cap_groups, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, s);
    if (!gu_once) switch (L.gu_type) {
    case 16: {
        iq2xxs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<16>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 17: {
        iq2xs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2xs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<17>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 18: {
        iq3xxs_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<18>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 21: {
        iq3s_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq3s_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<21>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 22: {
        iq2s_grid.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2s_grid.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<22>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 23: {
        kvalues_iq4nl.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = kvalues_iq4nl.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<23>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 29: {
        iq1s_grid_gpu.init(*s);

        s->submit([&](sycl::handler &cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq1s_grid_gpu.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<29>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
    case 42: {
        const void *iq2xxs_grid_ptr_ct1 = nullptr;

        s->submit([&](sycl::handler &cgh) {
            const void *iq2xxs_grid_ptr_ct1 = nullptr;

            cgh.parallel_for(sycl::nd_range<3>(ggu * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     native_gu_kernel<42>(
                                         grp_ptr, grp_start, n_groups, ent_tok,
                                         X, L, gate, up, iq2xxs_grid_ptr_ct1);
                                 });
        });
    } break;
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    s->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((nh + 255) / 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            swiglu_entries_kernel(gate, up, h, nh);
        });
    s->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((nh + 255) / 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            quantize_q8_1_kernel(h, hq, nh);
        });
    const bool split = native_expert_grouped_split_enabled();
    const bool d_once = !split && iq_once() &&
                        (q20_down(L, grp_ptr, grp_start, n_groups, ent_dst, hq, out, cap_groups, s) ||
                         iq_once_down(L, grp_ptr, grp_start, n_groups, ent_dst, hq, out, cap_groups, s));
    if (!d_once) switch (L.d_type) {
    case 20: {
        kvalues_iq4nl.init(*s);
        launch_native_down<20>(grp_ptr, grp_start, n_groups, ent_dst, hq,
                               L, out, kvalues_iq4nl.get_ptr(), cap_groups,
                               split, s);
    } break;
    case 23: {
        kvalues_iq4nl.init(*s);
        launch_native_down<23>(grp_ptr, grp_start, n_groups, ent_dst, hq,
                               L, out, kvalues_iq4nl.get_ptr(), cap_groups,
                               split, s);
    } break;
    case 42: {
        launch_native_down<42>(grp_ptr, grp_start, n_groups, ent_dst, hq,
                               L, out, nullptr, cap_groups, split, s);
    } break;
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace strata::kernels
