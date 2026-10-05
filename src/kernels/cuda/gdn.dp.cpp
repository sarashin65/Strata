// src/kernels/cuda/gdn.cu - P2.S2: the gated delta-net's non-projection parts.
//
// See include/strata/kernels/gdn.hpp for the state layout (S, h_v, S) and for why the recurrence needs no
// barrier at all: every line of it touches only one (j, h) column, so one thread owns a column end to end.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;   ///< threads along j, the state's fast axis
/// Heads staged per block.  **THIS IS A LATENCY-HIDING KNOB, NOT A CAPACITY ONE, AND IT WAS SET WRONG.**
///
/// The kernel is one thread per `(h, j)` COLUMN of the state, so the thread count is fixed at
/// `S * h_v` = 128 * 48 = **6,144** no matter how the work is packed - and `MAX_H` decides only how many of
/// those columns share a block:
///
///     MAX_H = 8   grid = (S/32, h_v/8) = (4, 6)  =  24 blocks x 32 threads  ->  1 warp on 24 of 48 SMs
///     MAX_H = 1   grid = (S/32, h_v/1) = (4, 48) = 192 blocks x 32 threads  ->  4 warps on every SM
///
/// Each thread walks `i` in 0..S-1 TWICE per head with a strided global load each step, so it is entirely
/// latency-bound: a warp that issues one 128-byte transaction and then waits ~600 cycles needs other warps to
/// run meanwhile, and at 24 blocks there were none.  Measured 0.2800 ms per GDN block - 10 ms of a 71.5 ms
/// token, 14%, in a kernel touching 12.6 MB that a 643 GB/s card should do in 0.020 ms.
///
/// The arithmetic does not change at all: the same columns are computed by the same code, in different blocks.
constexpr int MAX_H = 1;       ///< heads staged in shared memory per block

/// ggml_compute_softplus_f32: log1p(exp(x)), with the large-x branch that avoids overflow.
__dpct_inline__ float softplus_f(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
}
__dpct_inline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

/// One step of the recurrence.  One thread per (h, j) column of the state.
///
/// `ks`/`qs` stage `k[idx[h]]` and `q[idx[h]]` for the block's heads.  Within a warp h is constant and j
/// varies, so the staged row is read with a single broadcast - which is also why the GLOBAL loads of
/// `q`/`k` are done once per block into shared instead of once per thread.
void gdn_step_kernel(float* __restrict__ state, const float* __restrict__ q,
                                const float* __restrict__ k, const float* __restrict__ v,
                                const float* __restrict__ gate, const float* __restrict__ beta,
                                float* __restrict__ o, int S, int h_k, int h_v,
                                float ks[1/*MAX_H*/][128],
                                float qs[1/*MAX_H*/][128], float *dec_s,
                                float *beta_s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int h0 = item_ct1.get_group(1) * MAX_H;
    const int nh = sycl::min(MAX_H, h_v - h0);
    const int j = item_ct1.get_group(2) * JTHREADS + item_ct1.get_local_id(2);

    // stage this block's q/k rows and its per-head scalars
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const int src = h % h_k;                       // MODULO head pairing
        for (int i = item_ct1.get_local_id(2); i < S; i += JTHREADS) {
            ks[hi][i] = k[src * S + i];
            qs[hi][i] = q[src * S + i];
        }
        if (item_ct1.get_local_id(2) == 0) {
            dec_s[hi] = sycl::native::exp(gate[h]);
            beta_s[hi] = beta[h];
        }
    }
    /*
    DPCT1065:225: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (j >= S) return;

    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        float* col = state + (size_t) h * S + j;        // (S, h_v, S) with j fastest
        const size_t stride = (size_t) h_v * S;

        // pass 1: decay the state and contract it against k.  Decay BEFORE the update - PROPERTY 5.
        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] * dec;
            col[(size_t) i * stride] = s;
            sk += s * ks[hi][i];
        }
        const float d = (v[(size_t) h * S + j] - sk) * b;
        // pass 2: the rank-1 update, then read out against q using the UPDATED state
        float dot = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] + ks[hi][i] * d;
            col[(size_t) i * stride] = s;
            dot += s * qs[hi][i];
        }
        o[(size_t) h * S + j] = dot;
    }
}

/// `ggml_ssm_conv`: out[c] = sum_i inp[i][c] * kW[c*d_conv + i], inp = [conv_state | x], and the state slides.
/// One thread per channel; d_conv is 4, so the serial work per thread is a 4-tap dot and an unrolled shift.
void gdn_conv_kernel(float* __restrict__ cs, const float* __restrict__ x,
                                const float* __restrict__ kW, float* __restrict__ out, int C, int dc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
    acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
    st[dc - 2] = x[c];
}

/// `build_gdn_l2_norm`, one warp per row: x / sqrt(sum(x^2) + eps) over the last axis.
///
/// The sum is accumulated in DOUBLE, as `ref/gdn.py` does (`l2_norm(q.astype(np.float64), eps)`), because the
/// reference deliberately computes this one in float64 and then casts - so matching its precision is part of
/// transcribing it.
void gdn_l2_kernel(float* __restrict__ x, int cols, float eps, double &ssum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2);
    float* p = x + (size_t) row * cols;
    double acc = 0.0;
    for (int i = item_ct1.get_local_id(2); i < cols;
         i += item_ct1.get_local_range(2)) acc += (double)p[i] * (double)p[i];
    // warp reduction
    for (int off = 16; off > 0; off >>= 1) acc += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), acc, off);

    if (item_ct1.get_local_id(2) == 0) ssum = acc;
    /*
    DPCT1065:226: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float inv = (float)(1.0 / sycl::sqrt(ssum + (double)eps));
    for (int i = item_ct1.get_local_id(2); i < cols;
         i += item_ct1.get_local_range(2)) p[i] *= inv;
}

/// y = rms_norm(o, eps) * ssm_norm * sigmoid(z), one warp per head, norm over that head's S values.
void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ ssm_norm, float* __restrict__ y, int S,
                                    float eps, double &ssum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2);
    const float* po = o + (size_t) h * S;
    const float* pz = z + (size_t) h * S;
    float* py = y + (size_t) h * S;
    double acc = 0.0;
    for (int i = item_ct1.get_local_id(2); i < S;
         i += item_ct1.get_local_range(2)) acc += (double)po[i] * (double)po[i];
    for (int off = 16; off > 0; off >>= 1) acc += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), acc, off);

    if (item_ct1.get_local_id(2) == 0) ssum = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float inv = (float)(1.0 / sycl::sqrt(ssum / (double)S + (double)eps));
    for (int i = item_ct1.get_local_id(2); i < S;
         i += item_ct1.get_local_range(2))
        py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
}

}  // namespace

void gdn_step(float *state, const float *q, const float *k, const float *v,
              const float *gate, const float *beta, float *o,
              const GdnShapes &s, void *stream) try {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n",
                     (long long) s.S);
        std::exit(1);
    }
    const dpct::dim3 grid((unsigned)((s.S + JTHREADS - 1) / JTHREADS),
                          (unsigned)((s.h_v + MAX_H - 1) / MAX_H));
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:566: 'MAX_H' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[1 /*MAX_H*/][128], 0> ks_acc_ct1(cgh);
        /*
        DPCT1101:567: 'MAX_H' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[1 /*MAX_H*/][128], 0> qs_acc_ct1(cgh);
        /*
        DPCT1101:568: 'MAX_H' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> dec_s_acc_ct1(
            sycl::range<1>(1 /*MAX_H*/), cgh);
        /*
        DPCT1101:569: 'MAX_H' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> beta_s_acc_ct1(
            sycl::range<1>(1 /*MAX_H*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, JTHREADS),
                              sycl::range<3>(1, 1, JTHREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                gdn_step_kernel(
                    state, q, k, v, gate, beta, o, (int)s.S, (int)s.h_k,
                    (int)s.h_v, ks_acc_ct1, qs_acc_ct1,
                    dec_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    beta_s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void gdn_conv_step(float *conv_state, const float *x, const float *kW,
                   float *out, int64_t channels, int64_t d_conv,
                   void *stream) try {
    if (channels <= 0 || d_conv < 1) return;
    const int blocks = (int) ((channels + 255) / 256);
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           gdn_conv_kernel(conv_state, x, kW, out,
                                           (int)channels, (int)d_conv);
                       });
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void gdn_l2_norm(float *x, int64_t rows, int64_t cols, float eps,
                 void *stream) try {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp64});

        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 0> ssum_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)rows) *
                                          sycl::range<3>(1, 1, 32),
                                      sycl::range<3>(1, 1, 32)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_l2_kernel(x, (int)cols, eps, ssum_acc_ct1);
                        });
            });
    }
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/// `beta = sigmoid(beta)`, in place, over the `h_v` per-head scalars.
///
/// **THIS IS `build_layer_attn_linear` L889, AND LEAVING IT OUT WAS THE C1 BUG.**  The reference computes
/// `beta = ggml_sigmoid(ssm_beta @ cur)` and hands the RESULT to the delta net; `gdn_step_kernel` is documented
/// as `d[h,j] = (v[h,j] - sk[h,j]) * beta[h]` and applies no sigmoid of its own, so the layer glue owes it one.
/// The engine passed the raw pre-activation, which is unbounded and signed - the delta-net write strength then
/// stops being a fraction and the whole recurrence is wrong from layer 0.
///
/// **IT SURVIVED `gdn_parity` BECAUSE THAT TEST SUPPLIES ITS OWN `beta`.**  The kernel was always right; only
/// the glue between the projection and the kernel was wrong, and `ref/gdn.py` (`beta = sigmoid(wbeta @ x)`) and
/// `ref/model.py` (`beta = G.sigmoid(proj(...))`) both get it right - so the two transcriptions agreed with
/// each other and neither was ever compared against the C++ that actually runs.
void gdn_beta_gate_kernel(float* __restrict__ beta, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) beta[i] = sigmoid_f(beta[i]);
}

void gdn_beta_gate(float *beta, int64_t h_v, void *stream) try {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, (unsigned)((n + 127) / 128)) *
                               sycl::range<3>(1, 1, 128),
                           sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           gdn_beta_gate_kernel(beta, n);
                       });
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void gdn_out_norm(const float *o, const float *z, const float *ssm_norm,
                  float *y, int64_t h_v, int64_t S, float eps,
                  void *stream) try {
    if (h_v <= 0 || S <= 0) return;
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device(),
            {sycl::aspect::fp64});

        ((sycl::queue *)((dpct::queue_ptr)stream))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<double, 0> ssum_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)h_v) *
                                          sycl::range<3>(1, 1, 32),
                                      sycl::range<3>(1, 1, 32)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gdn_out_norm_kernel(o, z, ssm_norm, y, (int)S, eps,
                                                ssum_acc_ct1);
                        });
            });
    }
    if (stream == nullptr) {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
