// src/kernels/cuda/dequant_bf16.cu - see include/strata/kernels/dequant_bf16.hpp.
//
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {

__dpct_inline__ float h2f(const uint8_t *p) {
    return sycl::ext::intel::math::half2float(
        sycl::bit_cast<sycl::half, unsigned short>(
            (uint16_t)(p[0] | (p[1] << 8))));
}
__dpct_inline__ uint16_t f2bf(float f) {
    uint32_t u = sycl::bit_cast<unsigned int>(f);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
__dpct_inline__ void put(uint16_t *o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__dpct_inline__ void put(H16 *o, int i, float v) {
    o[i].v = sycl::bit_cast<unsigned short, sycl::half>(
        sycl::ext::intel::math::float2half_rn(v));
}
__dpct_inline__ void put(float *o, int i, float v) { o[i] = v; }

inline dpct::constant_memory<int8_t, 1>
    kv_iq4nl(sycl::range<1>(16), {-127, -104, -83, -65, -49, -35, -22, -10, 1,
                                  13, 25, 38, 53, 69, 89, 113});

__dpct_inline__ void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
/*
DPCT1110:5: The total declared local variable size in device function group32
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void group32(const uint8_t *row_blocks, int gi_in_row, T *out,
                             int8_t const *kv_iq4nl) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

template <int TYPE, typename T>
void dequant_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                               int64_t groups_per_row, T* __restrict__ out,
                               int8_t const *kv_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t g =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int)gi,
                  out + r * groups_per_row * 32 + gi * 32, kv_iq4nl);
}

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    dpct::queue_ptr st = (dpct::queue_ptr)stream;
#define STRATA_DQ(TY)                                                          \
    {                                                                          \
        kv_iq4nl.init(*st);                                                    \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto kv_iq4nl_ptr_ct1 = kv_iq4nl.get_ptr();                        \
                                                                               \
            auto p_ct0 = p;                                                    \
            auto row_bytes_ct1 = row_bytes;                                    \
            auto row0_ct2 = row0;                                              \
            auto rows_ct3 = rows;                                              \
            auto gpr_ct4 = gpr;                                                \
            auto out_ct5 = out;                                                \
                                                                               \
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *    \
                                                   sycl::range<3>(1, 1, 256),  \
                                               sycl::range<3>(1, 1, 256)),     \
                             [=](sycl::nd_item<3> item_ct1) {                  \
                                 dequant_kernel<TY, T>(                        \
                                     p_ct0, row_bytes_ct1, row0_ct2, rows_ct3, \
                                     gpr_ct4, out_ct5, kv_iq4nl_ptr_ct1);      \
                             });                                               \
        });                                                                    \
    } break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    /*
    DPCT1010:181: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009:182: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

// Q6_K's reordered copy is plane-major across the whole matrix: ql, qh, scales, then d.
template <int TYPE, typename T>
__dpct_inline__ void reordered_group32(const uint8_t* row_blocks, int gi_in_row, T* out,
                                       int8_t const* kv_iq4nl) {
    if constexpr (TYPE == 12) {
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b + 140), dmin = h2f(b + 142);
        int sc, m;
        scale_min_k4(gi, b + 128, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t* q = b + 32 * j64;
        for (int l = 0; l < 32; ++l)
            put(out, l, d1 * (float)(hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b + 172), dmin = h2f(b + 174);
        int sc, m;
        scale_min_k4(gi, b + 160, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t* qh = b + 128;
        const uint8_t* ql = b + 32 * j64;
        const uint8_t u = (uint8_t)(1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float)(nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 23) {
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b + 134);
        const uint16_t scales_h = (uint16_t)(b[132] | (b[133] << 8));
        const int ls = ((b[128 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t* qs = b + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float)kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float)kv_iq4nl[qs[j] >> 4]);
        }
    }
}
template <int TYPE, typename T>
void reordered_dequant_kernel(const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                              int64_t groups_per_row, T* out, int8_t const* kv_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t g = (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    reordered_group32<TYPE>(blocks + (row0 + r) * row_bytes, (int)gi,
                            out + r * groups_per_row * 32 + gi * 32, kv_iq4nl);
}
template <typename T>
void launch_reordered(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream) {
    int be = 0, bb = 0;
    if ((type != 12 && type != 13 && type != 23) || !geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant reordered: unsupported type or shape\n");
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32;
    const unsigned grid = (unsigned)((rows * gpr + 255) / 256);
    auto* st = (dpct::queue_ptr)stream;
    const auto* p = static_cast<const uint8_t*>(blocks);
    kv_iq4nl.init(*st);
    st->submit([&](sycl::handler& cgh) {
        const auto p0 = p; const auto out0 = out; const auto tab = kv_iq4nl.get_ptr();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) * sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                          [=](sycl::nd_item<3> item) {
                              if (type == 12)
                                  reordered_dequant_kernel<12, T>(p0, row_bytes, row0, rows, gpr, out0, tab);
                              else if (type == 13)
                                  reordered_dequant_kernel<13, T>(p0, row_bytes, row0, rows, gpr, out0, tab);
                              else
                                  reordered_dequant_kernel<23, T>(p0, row_bytes, row0, rows, gpr, out0, tab);
                          });
    });
}
// Dequantize it directly so the prompt path does not need the original 210-byte blocks.
template <typename T>
void q6_reordered_kernel(const uint8_t* base, int64_t ql_bytes, int64_t qh_bytes, int64_t scale_bytes,
                         int64_t row0, int64_t rows, int64_t cols, T* out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t groups_per_row = cols / 32;
    const int64_t g = (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      item_ct1.get_local_id(2);
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row;
    const int gi = (int) (g % groups_per_row);
    const int64_t block = (row0 + r) * (cols / 256) + gi / 8;
    const uint8_t* ql = base + block * 128;
    const uint8_t* qh = base + ql_bytes + block * 64;
    const int8_t* sc = reinterpret_cast<const int8_t*>(base + ql_bytes + qh_bytes + block * 16);
    const float d = h2f(base + ql_bytes + qh_bytes + scale_bytes + block * 2);
    const int n = (gi % 8) / 4;
    const int qu = gi % 4;
    T* dst = out + r * groups_per_row * 32 + gi * 32;
    for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        int q;
        if (qu == 0) q = (ql[64 * n + l] & 0xF) | (((qh[32 * n + l] >> 0) & 3) << 4);
        else if (qu == 1) q = (ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4);
        else if (qu == 2) q = (ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4);
        else q = (ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4);
        put(dst, l, d * (float) sc[8 * n + is + 2 * qu] * (float) (q - 32));
    }
}

template <typename T>
void launch_q6_reordered(const void* blocks, int64_t row0, int64_t rows, int64_t total_rows, int64_t cols, T* out,
                         void* stream) {
    if (!blocks || !out || row0 < 0 || rows <= 0 || total_rows <= 0 || row0 > total_rows - rows || cols <= 0 ||
        cols % 256 != 0) {
        std::fprintf(stderr, "dequant reordered Q6_K: invalid shape\n");
        std::exit(1);
    }
    // The plane offsets cover the complete matrix; row0 is applied only to the output/block index.
    const int64_t total_blocks = total_rows * (cols / 256);
    const int64_t ql_bytes = total_blocks * 128;
    const int64_t qh_bytes = total_blocks * 64;
    const int64_t scale_bytes = total_blocks * 16;
    const int64_t total = rows * (cols / 32);
    const unsigned grid = (unsigned) ((total + 255) / 256);
    auto* st = (dpct::queue_ptr) stream;
    const auto* base = static_cast<const uint8_t*>(blocks);
    st->submit([&](sycl::handler& cgh) {
        auto base_ct0 = base;
        auto ql_bytes_ct1 = ql_bytes;
        auto qh_bytes_ct2 = qh_bytes;
        auto scale_bytes_ct3 = scale_bytes;
        auto row0_ct4 = row0;
        auto rows_ct5 = rows;
        auto cols_ct6 = cols;
        auto out_ct7 = out;
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) * sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                          [=](sycl::nd_item<3> item_ct1) {
                              q6_reordered_kernel<T>(base_ct0, ql_bytes_ct1, qh_bytes_ct2, scale_bytes_ct3,
                                                      row0_ct4, rows_ct5, cols_ct6, out_ct7);
                          });
    });
}
}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16_q6_reordered(const void* blocks, int64_t row0, int64_t rows, int64_t total_rows, int64_t cols,
                              uint16_t* out, void* stream) {
    launch_q6_reordered<H16>(blocks, row0, rows, total_rows, cols, reinterpret_cast<H16*>(out), stream);
}

void dequant_f16_reordered(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                          uint16_t* out, void* stream) {
    launch_reordered<H16>(ggml_type, blocks, row0, rows, cols,
                          reinterpret_cast<H16*>(out), stream);
}
void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
