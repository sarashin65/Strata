// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x +=
        dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset);
    return x;
}
void gate_kernel(const float* key, const float* query, float* gate, float scale,
                 float *partials) {
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = item_ct1.get_local_id(2) + j * 512;
        const float p = d < N ? sycl::ext::intel::math::fmul_rn(
                                    key[item_ct1.get_group(2) * N + d],
                                    query[item_ct1.get_group(2) * N + d])
                              : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];

    sum = warp_sum(sum);
    const int lane = item_ct1.get_local_id(2) % 32;
    if (!lane) partials[item_ct1.get_local_id(2) / 32] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (item_ct1.get_local_id(2) == 0) {
        const float s = sycl::ext::intel::math::fmaf_rn(
            scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[item_ct1.get_group(2)] =
            1.0f / (1.0f + sycl::native::exp(
                               -sycl::ext::intel::math::fmul_rn(sign, mag)));
    }
}
void broadcast_kernel(const float* value, const float* gate, float* gated) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < D) gated[i] =
        sycl::ext::intel::math::fmul_rn(value[i % N], gate[i / N]);
}
void conv_residual_kernel(const float* history, const float* normalized,
                                    const uint16_t* weights, const float* hidden,
                                    const float* gated, float* conv, float* result) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = sycl::ext::intel::math::half2float(
            sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]));
        const float term = sycl::ext::intel::math::fmul_rn(x, w);
        sum = k == 0 ? term : sycl::ext::intel::math::fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = sycl::ext::intel::math::fadd_rn(
        hidden[c], sycl::ext::intel::math::fadd_rn(gated[c], activation));
}
// ---- the batch: T tokens, the same arithmetic per element as the kernels above
// weighted_rms_norm (native_gr_norm.cu) with the gamma row repeating every H rows (one token's H groups)
float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value +=
        dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), value, offset);
    return value;
}
void rms_rep_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                               float* __restrict__ output, float *sums) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BlockSize =
        1024; // native_gr_rms_norm_weighted's choice for 2560 columns
    const int tid = item_ct1.get_local_id(2);
    const size_t row_offset = size_t(item_ct1.get_group(2)) * N;
    input += row_offset;
    output += row_offset;
    gamma += size_t(item_ct1.get_group(2) % H) * N;
    float partial = 0.0f;
    for (int col = tid; col < N; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / N;
    const float scale = sycl::rsqrt(mean + NG_RMS_EPS);
    for (int col = tid; col < N; col += BlockSize) output[col] = scale * input[col] * gamma[col];
}
void broadcast_batch_kernel(const float* value, const float* gate, float* gated, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= size_t(T) * D) return;
    const size_t t = i / D, d = i % D;
    gated[i] = sycl::ext::intel::math::fmul_rn(value[t * N + d % N],
                                               gate[t * H + d / N]);
}
// the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
void conv_residual_batch_kernel(const float* history, const float* normalized, const uint16_t* weights,
                                           float* hidden, const float* gated, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = t - 9 + 3 * k;             // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        const float wk = sycl::ext::intel::math::half2float(
            sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]));
        const float term = sycl::ext::intel::math::fmul_rn(x, wk);
        sum = k == 0 ? term : sycl::ext::intel::math::fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    hidden[i] = sycl::ext::intel::math::fadd_rn(
        hidden[i], sycl::ext::intel::math::fadd_rn(gated[i], activation));
}
// the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
void history_batch_kernel(float* history, const float* normalized, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= D) return;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = T - HISTORY + r;
        h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
}

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    /*
    DPCT1010:311: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009:312: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001:309: The statement could not be removed.
    */
    /*
    DPCT1000:310: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(std::string("native PLE postops launch: ") +
                                    dpct::get_error_string_dummy(error));
}
} // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = static_cast<dpct::queue_ptr>(stream);
    /*
    DPCT1049:60: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    st->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> partials_acc_ct1(sycl::range<1>(32),
                                                        cgh);

        auto std_sqrt_float_N_ct3 = 1.0f / std::sqrt(float(N));

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, H) *
                                  sycl::range<3>(1, 1, 512),
                              sycl::range<3>(1, 1, 512)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gate_kernel(b.key, b.query, b.gate, std_sqrt_float_N_ct3,
                            partials_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, D / 256) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                         broadcast_kernel(value, b.gate, b.gated);
                     });
    launch_check();
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, D / 256) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                         conv_residual_kernel(history, b.normalized,
                                              w.conv1d_f16, hidden, b.gated,
                                              b.conv, b.result);
                     });
    launch_check();
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto st = static_cast<dpct::queue_ptr>(stream);
    const unsigned rows = unsigned(T) * H;
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    /*
    DPCT1049:61: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    st->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sums_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rms_rep_kernel(
                    key, w.norm_key, key,
                    sums_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1049:62: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    st->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sums_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rms_rep_kernel(
                    hidden, w.norm_query, query_norm,
                    sums_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1049:63: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    st->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> partials_acc_ct1(sycl::range<1>(32),
                                                        cgh);

        auto std_sqrt_float_N_ct3 = 1.0f / std::sqrt(float(N));

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                  sycl::range<3>(1, 1, 512),
                              sycl::range<3>(1, 1, 512)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gate_kernel(key, query_norm, gate, std_sqrt_float_N_ct3,
                            partials_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                         broadcast_batch_kernel(value, gate, gated, T);
                     });
    /*
    DPCT1049:64: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    st->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sums_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rms_rep_kernel(
                    gated, w.norm_conv, query_norm,
                    sums_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                         conv_residual_batch_kernel(history, query_norm,
                                                    w.conv1d_f16, hidden, gated,
                                                    T);
                     });
    st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, D / 256) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                         history_batch_kernel(history, query_norm, T);
                     });
    launch_check();
}
} // namespace strata::kernels
