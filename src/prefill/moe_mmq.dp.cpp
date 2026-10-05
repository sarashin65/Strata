// src/prefill/moe_mmq.dp.cpp
//
// The former TU included llama.cpp's CUDA common/mmq/quantize headers.  Those
// headers expose CUDA-only warp intrinsics even for a host SYCL compile.  This
// TU keeps the Strata Product/bounds contract and uses the pinned llama.cpp
// SYCL quantization and vector-dot implementations instead.
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include <sycl/sycl.hpp>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common-upstream.h"

#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/sycl_mmq_adapter.hpp"
#include "ggml-sycl/mmvq.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/quants.hpp"
#include "ggml-sycl/vecdotq.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::prefill::mmq {
namespace {

int64_t pad512(int64_t n) {
    return (n + 511) / 512 * 512;
}

void copy16_kernel(const sycl::uint4 *__restrict__ a, int64_t na,
                   const sycl::uint4 *__restrict__ b, int64_t nb,
                   sycl::uint4 *__restrict__ ab_dst,
                   const sycl::uint4 *__restrict__ c, int64_t nc,
                   sycl::uint4 *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) {
        ab_dst[i] = a[i];
    } else if (i < na + nb) {
        ab_dst[i] = b[i - na];
    } else if (i < na + nb + nc) {
        c_dst[i - na - nb] = c[i - na - nb];
    }
}

void copy1_kernel(const uint8_t *__restrict__ a, int64_t na,
                  const uint8_t *__restrict__ b, int64_t nb,
                  uint8_t *__restrict__ ab_dst,
                  const uint8_t *__restrict__ c, int64_t nc,
                  uint8_t *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) {
        ab_dst[i] = a[i];
    } else if (i < na + nb) {
        ab_dst[i] = b[i - na];
    } else if (i < na + nb + nc) {
        c_dst[i - na - nb] = c[i - na - nb];
    }
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up
// scales [1280][40] f16, down scales [2560][10] f16.  A GGUF Q2_0 block is
// {f16 d; 16 code bytes}; the code order is unchanged by this conversion.
void strata_q2_kernel(const uint8_t *__restrict__ blob,
                      uint16_t *__restrict__ gu,
                      uint16_t *__restrict__ dn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr size_t O_D_CODES = (size_t) 1280 * 640;
    constexpr size_t O_GU_SC = O_D_CODES + (size_t) 2560 * 160;
    constexpr size_t O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i =
        (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    const int64_t n_gu = 1280LL * 40;
    const int64_t n_d = 2560LL * 10;
    const uint8_t * codes;
    const uint16_t * scale;
    uint16_t * out;
    if (i < n_gu) {
        const int64_t row = i / 40;
        const int64_t b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t *) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu;
        const int64_t row = j / 10;
        const int64_t b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t *) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const sycl::uint4 q = *(const sycl::uint4 *) codes;
    const uint16_t * qh = (const uint16_t *) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        out[1 + k] = qh[k];
    }
}

void swiglu_kernel(const float *__restrict__ gu, float *__restrict__ h,
                   int64_t rows, int64_t n_ff, bool interleaved) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * n_ff) {
        return;
    }
    const int64_t r = i / n_ff;
    const int64_t k = i % n_ff;
    const float * row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k];
    const float u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

void iota_kernel(int32_t * dst, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) {
        dst[i] = (int32_t) i;
    }
}

unsigned blocks(int64_t n) {
    return (unsigned) ((n + 255) / 256);
}

// This is the q8_1 path from ggml-sycl/quantize.hpp with the Strata source-row
// indirection retained.  The output is the standard SYCL block_q8_1 layout:
// int8 quants followed by half(d), half(sum) per QK8_1 block.
template <int ElementsPerWI>
void quantize_q8_1_ids_kernel(const float *__restrict__ x,
                              const int32_t *__restrict__ ids,
                              void *__restrict__ vy, int kx, int kx_padded,
                              int64_t ld, int ky,
                              const sycl::nd_item<1> & item) {
    const int group = item.get_group(0);
    const int local = item.get_local_id(0);
    const int blocks_per_row = kx / QK8_1;
    const int row = group / blocks_per_row;
    const int block = group % blocks_per_row;
    if (row >= ky) {
        return;
    }

    const int source_row = ids == nullptr ? row : ids[row];
    const float * source = x + (int64_t) source_row * ld + block * QK8_1;
    const sycl::vec<float, ElementsPerWI> values =
        *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(source + local * ElementsPerWI);

    float amax = 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ElementsPerWI; ++i) {
        sum += values[i];
        amax = sycl::fmax(amax, sycl::fabs(values[i]));
    }
    amax = sycl::reduce_over_group(item.get_sub_group(), amax, sycl::maximum<float>());
    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());

    const float d = amax == 0.0f ? 1.0f : amax / 127.0f;
    sycl::vec<int8_t, ElementsPerWI> quantized;
#pragma unroll
    for (int i = 0; i < ElementsPerWI; ++i) {
        quantized[i] = sycl::round(values[i] / d);
    }

    block_q8_1 * output = (block_q8_1 *) vy;
    const int output_block = row * (kx_padded / QK8_1) + block;
    *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(
        output[output_block].qs + local * ElementsPerWI) = quantized;
    if (local == 0) {
        output[output_block].ds = sycl::half2(
            sycl::half(amax == 0.0f ? 0.0f : d), sycl::half(sum));
    }
}

void quantize_q8_1_ids(const float * x, const int32_t * ids, void * vy,
                       int64_t cols, int64_t ld, int64_t rows,
                       dpct::queue_ptr stream) {
    if (rows <= 0) {
        return;
    }
    if (cols <= 0 || cols % QK8_1 != 0 || ld < cols) {
        std::fprintf(stderr, "prefill mmq: invalid q8_1 shape\n");
        std::exit(1);
    }

    constexpr int elements_per_wi = QK8_1 / WARP_SIZE;
    static_assert(elements_per_wi > 0 && QK8_1 % WARP_SIZE == 0);
    const int blocks_per_row = (int) (cols / QK8_1);
    const sycl::range<1> local_range((size_t) WARP_SIZE);
    const sycl::range<1> global_range((size_t) rows * blocks_per_row * WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            quantize_q8_1_ids_kernel<elements_per_wi>(
                x, ids, vy, (int) cols, (int) pad512(cols), ld, (int) rows, item);
        });
}

// The SYCL MMVQ implementation uses block_q8_1 activations and llama.cpp's
// vec_dot_q_sycl_t functions.  Unlike the generic llama MoE API, this launcher
// keeps the Strata variable expert bounds and destination row map in-device.
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl>
void mul_mat_q_bounds_kernel(
    const char *__restrict__ weights, const void *__restrict__ activations,
    const int32_t *__restrict__ bounds, const int32_t *__restrict__ ids_dst,
    float *__restrict__ dst, int n_experts, int weight_rows, int ncols,
    int max_rows, size_t expert_weight_stride, size_t dst_row_stride,
    size_t activation_row_stride, const sycl::nd_item<3> & item) {
    const int expert = (int) item.get_group(1);
    const int output_tiles = (weight_rows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const int group_z = (int) item.get_group(2);
    const int activation_row = group_z / output_tiles;
    const int output_tile = group_z % output_tiles;
    const int output_row = output_tile * GGML_SYCL_MMV_Y + (int) item.get_local_id(1);

    if (expert >= n_experts || activation_row >= max_rows || output_row >= weight_rows) {
        return;
    }

    const int row_low = bounds[expert];
    const int row_high = bounds[expert + 1];
    const int row_count = row_high - row_low;
    if (activation_row >= row_count) {
        return;
    }

    const int input_row = row_low + activation_row;
    const int destination_row = ids_dst == nullptr ? input_row : ids_dst[input_row];
    const char * expert_weights = weights + (size_t) expert * expert_weight_stride;
    const block_q_t * x = (const block_q_t *) expert_weights;
    const block_q8_1 * y = (const block_q8_1 *)
        ((const char *) activations + (size_t) input_row * activation_row_stride);

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_subgroup = (vdr * WARP_SIZE + qi - 1) / qi;
    constexpr int block_elements_per_subgroup = qi / vdr;
    float tmp = 0.0f;

    for (int i = (int) item.get_local_id(2) / (qi / vdr);
         i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = output_row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + vdr * ((int) item.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item.get_sub_group(), tmp, mask);
    }

    if (item.get_local_id(2) == 0) {
        dst[(size_t) destination_row * dst_row_stride + output_row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl>
void launch_mul_mat_q_bounds(
    const void * weights, const void * activations, const int32_t * bounds,
    const int32_t * ids_dst, float * dst, int n_experts, int weight_rows,
    int ncols, int max_rows, size_t expert_weight_stride,
    size_t dst_row_stride, size_t activation_row_stride,
    dpct::queue_ptr stream) {
    const int output_tiles = (weight_rows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> group_count(
        1, (size_t) n_experts, (size_t) max_rows * output_tiles);
    const sycl::range<3> group_size(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(group_count * group_size, group_size),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_q_bounds_kernel<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    (const char *) weights, activations, bounds, ids_dst, dst,
                    n_experts, weight_rows, ncols, max_rows, expert_weight_stride,
                    dst_row_stride, activation_row_stride, item);
            });
    });
}

// IQ dot functions need their codebook tables bound as template arguments,
// exactly as in ggml-sycl/mmvq.cpp.
static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int & iqs) {
    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int & iqs) {
    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int & iqs) {
    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int & iqs) {
    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
}

}  // namespace

bool built() {
    return true;
}

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) *
           ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (pad512(cols) / QK8_1) * sizeof(block_q8_1);
}

void quantize(const float * x, const int32_t * ids, void * xq, int, int64_t cols,
              int64_t ld, int64_t rows, void * stream) {
    if (stream == nullptr) std::abort();
    quantize_q8_1_ids(x, ids, xq, cols, ld, rows, (dpct::queue_ptr) stream);
}

Context::Context(void* stream) : ctx_(detail::new_context(stream)) {}

Context::~Context() {
    detail::delete_context(ctx_);
    ctx_ = nullptr;
}

void Context::run(const Product & p, void * stream) {
    if (p.n <= 0 || p.max_rows <= 0 || p.w_rows <= 0 || p.w_cols <= 0) {
        return;
    }
    const ggml_type t = (ggml_type) p.type;
    const dpct::queue_ptr queue =
        stream != nullptr ? (dpct::queue_ptr) stream : detail::context_stream(ctx_);
    const size_t q8_row_bytes =
        (size_t) (pad512(p.w_cols) / QK8_1) * sizeof(block_q8_1);

    switch (t) {
        case GGML_TYPE_Q2_0:
            launch_mul_mat_q_bounds<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ,
                                    vec_dot_q2_0_q8_1>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ2_XXS:
            launch_mul_mat_q_bounds<QK_K, QI2_XXS / 2, block_iq2_xxs,
                                    VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ2_XS:
            launch_mul_mat_q_bounds<QK_K, QI2_XS / 2, block_iq2_xs,
                                    VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ2_S:
            launch_mul_mat_q_bounds<QK_K, QI2_S / 2, block_iq2_s,
                                    VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ3_XXS:
            launch_mul_mat_q_bounds<QK_K, QI3_XXS / 2, block_iq3_xxs,
                                    VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_q_bounds<QK_K, QI3_S / 2, block_iq3_s,
                                    VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_moe>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_q_bounds<QK4_NL, QI4_NL, block_iq4_nl,
                                    VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_q_bounds<QK_K, QI4_XS / 4, block_iq4_xs,
                                    VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                p.w, p.xq, p.bounds, p.ids, p.dst, p.n, (int) p.w_rows,
                (int) p.w_cols, (int) p.max_rows, p.expert_bytes,
                (size_t) p.ld_dst, q8_row_bytes, queue);
            break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
}

void gather_native(const void * gate, const void * up, size_t gu_half_bytes,
                   const void * down, size_t d_bytes, void * gu_dst, void * d_dst,
                   void * stream) {
    const dpct::queue_ptr queue = (dpct::queue_ptr) stream;
    const bool aligned = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down |
                          (uintptr_t) gu_dst | (uintptr_t) d_dst |
                          gu_half_bytes | d_bytes) % 16 == 0;
    if (aligned) {
        const int64_t na = (int64_t) gu_half_bytes / 16;
        const int64_t nc = (int64_t) d_bytes / 16;
        queue->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks(2 * na + nc)) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item) {
                copy16_kernel((const sycl::uint4 *) gate, na,
                              (const sycl::uint4 *) up, na,
                              (sycl::uint4 *) gu_dst, (const sycl::uint4 *) down,
                              nc, (sycl::uint4 *) d_dst);
            });
    } else {
        const int64_t na = (int64_t) gu_half_bytes;
        const int64_t nc = (int64_t) d_bytes;
        queue->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks(2 * na + nc)) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item) {
                copy1_kernel((const uint8_t *) gate, na, (const uint8_t *) up,
                             na, (uint8_t *) gu_dst, (const uint8_t *) down, nc,
                             (uint8_t *) d_dst);
            });
    }
}

void gather_strata_q2(const uint8_t * blob, void * gu_dst, void * d_dst,
                      void * stream) {
    const dpct::queue_ptr queue = (dpct::queue_ptr) stream;
    queue->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks(1280LL * 40 + 2560LL * 10)) *
                sycl::range<3>(1, 1, 256),
            sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item) {
            strata_q2_kernel(blob, (uint16_t *) gu_dst, (uint16_t *) d_dst);
        });
}

void swiglu(const float * gu, float * h, int64_t rows, int64_t n_ff,
            bool interleaved, void * stream) {
    if (rows <= 0) {
        return;
    }
    const dpct::queue_ptr queue = (dpct::queue_ptr) stream;
    queue->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks(rows * n_ff)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item) {
            swiglu_kernel(gu, h, rows, n_ff, interleaved);
        });
}

void iota(int32_t * dst, int64_t n, void * stream) {
    if (n <= 0) {
        return;
    }
    const dpct::queue_ptr queue = (dpct::queue_ptr) stream;
    queue->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks(n)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item) {
            iota_kernel(dst, n);
        });
}

}  // namespace strata::prefill::mmq
