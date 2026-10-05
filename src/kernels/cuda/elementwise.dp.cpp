// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include "strata/kernels/elementwise.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/ext/intel/math.hpp>

#include <time.h>

namespace sx = sycl::ext::oneapi::experimental;
namespace si = sycl::ext::intel::experimental;
template <si::cache_level... Ls>
using uncached = decltype(sx::properties(si::read_hint<si::cache_control<si::cache_mode::uncached, Ls...>>));
using uncached_l1l3 = uncached<si::cache_level::L1, si::cache_level::L3>;

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

bool plan_copy_wide_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PLAN_COPY_WIDE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

void embedding_gather_kernel(const uint8_t* __restrict__ codes,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ offsets, int64_t n,
                                       int code_bits, int code_bias, int group_elems,
                                       float* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = sycl::ext::intel::math::fmul_rn(
        (float)(code + code_bias), scales[group]);
    out[i] = sycl::ext::intel::math::fadd_rn(product,
                                             offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
__dpct_inline__ float softplus_dev(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
}

void gdn_gate_kernel(const float* __restrict__ alpha, const float* __restrict__ dt,
                                const float* __restrict__ ssm_a, float* __restrict__ gate, int64_t h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

void scale_kernel(float* __restrict__ x, int64_t n, float s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] *= s;
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
void add_kernel(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) dst[i] += src[i];
}

void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = f16_from_f32(x[i]);
}

void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

void silu_kernel(float* __restrict__ x, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float)(v / (1.0 + sycl::exp(-v)));
}

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
void rms_norm_weighted_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t rows,
                                         int64_t cols, float eps) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t row =
        ((int64_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5)) +
        (item_ct1.get_local_id(2) >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
    for (int off = 16; off > 0; off >>= 1) acc += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), acc, off);
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = sycl::rsqrt(acc / (float)cols + eps);
    inv = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), inv, 0);
    for (int64_t c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

void sync_if_needed(void *stream, const char *what) try {
    if (stream != nullptr) return;
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool check_launch(const char* what) {
    /*
    DPCT1010:185: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           embedding_gather_kernel(codes, scales, offsets, n,
                                                   code_bits, code_bias,
                                                   group_elems, out);
                       });
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           gdn_gate_kernel(alpha, dt, ssm_a, gate, h_v);
                       });
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           scale_kernel(x, n, s);
                       });
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           add_kernel(dst, src, n);
                       });
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           to_f16_kernel(x, y, n);
                       });
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                           to_bf16_kernel(x, y, n);
                       });
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp64});

        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n)) *
                                                 sycl::range<3>(1, 1, THREADS),
                                             sycl::range<3>(1, 1, THREADS)),
                           [=](sycl::nd_item<3> item_ct1) {
                               silu_kernel(x, n);
                           });
    }
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
void doorbell_ring_kernel(uint32_t* seq) {
    /*
    DPCT1078:6: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    *seq = *seq + 1u;
}

void doorbell_wait_kernel(const volatile uint32_t* flag, const volatile uint32_t* seq) {
    const uint32_t want = *seq;
    sx::annotated_ptr<uint32_t, uncached_l1l3> flag_read(const_cast<uint32_t *>(flag));
    while (*flag_read != want && *flag_read != 0xffffffffu) {
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                doorbell_wait_kernel(d_flag, d_seq);
            });
    check_launch("doorbell_wait");
}

// The CUDA payload was volatile. Scalar volatile loads preserve that host-visibility
// contract because SYCL does not support member access on volatile vector types.
void copy_from_mapped_kernel(float *__restrict__ dst,
                             const volatile float *src, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                    item_ct1.get_local_range(2)) {
        dst[i] = src[i];
    }
}

void gather_miss_rows_kernel(float* __restrict__ parts, const volatile float* ymiss,
                             const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                             int64_t rows, int64_t n_embd, int groups_per_row, int& gpu_row) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t group = item_ct1.get_group(2);
    const int64_t row = group / groups_per_row;
    const int block = (int) (group % groups_per_row);
    if (row >= rows) return;
    if (item_ct1.get_local_id(2) == 0) {
        const int entries = *count;
        int is_gpu = 0;
        for (int j = 0; j < entries; ++j) {
            if (dst[j] == row) {
                is_gpu = 1;
                break;
            }
        }
        gpu_row = is_gpu;
    }
    item_ct1.barrier();
    for (int64_t c = (int64_t) block * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
         c < n_embd; c += (int64_t) groups_per_row * item_ct1.get_local_range(2)) {
        const int64_t i = row * n_embd + c;
        parts[i] = gpu_row ? 0.0f : ymiss[i];
    }
}

void gather_miss_rows(float* parts, const float* ymiss_mapped, const int32_t* dst, const int32_t* count,
                      int64_t rows, int64_t n_embd, void* stream) {
    if (rows <= 0 || n_embd <= 0) return;
    const int groups_per_row = (int) ((n_embd + THREADS - 1) / THREADS);
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 0> gpu_row(cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (size_t) rows * (size_t) groups_per_row) *
                    sycl::range<3>(1, 1, THREADS),
                sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                gather_miss_rows_kernel(parts, (const volatile float*) ymiss_mapped, dst, count, rows,
                                        n_embd, groups_per_row, gpu_row);
            });
    });
    check_launch("gather_miss_rows");
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int64_t nf = n4 * 4;
    const int blocks = (int) ((nf + 255) / 256 < 64 ? (nf + 255) / 256 : 64);
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           copy_from_mapped_kernel(
                               dst, (const volatile float *)src, nf);
                       });
    check_launch("copy_from_mapped");
}

void doorbell_publish_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                        const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
                                        float* w_out, uint32_t* seq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) x_out[i] = x[i];
    if ((int)item_ct1.get_local_id(2) < k) {
        ids_out[item_ct1.get_local_id(2)] = ids[item_ct1.get_local_id(2)];
        w_out[item_ct1.get_local_id(2)] = w[item_ct1.get_local_id(2)];
    }
    /*
    DPCT1078:8: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    /*
    DPCT1065:189: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        /*
        DPCT1078:9: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
    }
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                                         sycl::range<3>(1, 1, 1024)),
                       [=](sycl::nd_item<3> item_ct1) {
                           doorbell_publish_kernel(x, ids, weights, (int)n,
                                                   (int)k, x_out, ids_out,
                                                   weights_out, d_seq);
                       });
    check_launch("doorbell_publish");
}

void expand_pos_kernel(const int32_t* __restrict__ src, int64_t n_tok, int64_t src_rows,
                       int64_t rows_per_tok, int32_t* __restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    const int64_t total = n_tok * rows_per_tok;
    if (i >= total) return;
    const int64_t token = i / rows_per_tok;
    const int64_t row = i % rows_per_tok;
    dst[i] = src[token * src_rows + row];
}

void copy_i32_from_mapped_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const size_t groups = plan_copy_wide_enabled() ? (size_t) ((n + 127) / 128) : 1;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, groups * 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           copy_i32_from_mapped_kernel(
                               dst, (const volatile int32_t *)src, (int)n);
                       });
    check_launch("copy_i32_from_mapped");
}
void expand_pos(const int32_t* src, int64_t n_tok, int64_t src_rows, int64_t rows_per_tok, int32_t* dst,
                void* stream) {
    if (n_tok <= 0 || src_rows <= 0 || rows_per_tok <= 0) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_for(n_tok * rows_per_tok)) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                expand_pos_kernel(src, n_tok, src_rows, rows_per_tok, dst);
            });
    check_launch("expand_pos");
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                doorbell_ring_kernel(d_seq);
            });
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, warps_per_block * 32),
                              sycl::range<3>(1, 1, warps_per_block * 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rms_norm_weighted_kernel(x, w, rows, cols, eps);
            });
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
