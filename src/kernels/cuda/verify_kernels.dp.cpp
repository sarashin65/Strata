// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include "strata/kernels/verify_kernels.hpp"

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

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    /*
    DPCT1010:448: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009:449: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

bool gdn_env_is_one(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

void gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin,
                                                              float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = t_begin + item_ct1.get_group(1);
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        /*
        DPCT1118:126: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

/*
DPCT1110:127: The total declared local variable size in device function
gdn_ab_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gdn_ab_multi_kernel(const float *__restrict__ x,
                         const uint16_t *__restrict__ wa,
                         const uint16_t *__restrict__ wb,
                         const float *__restrict__ dt,
                         const float *__restrict__ ssm_a,
                         float *__restrict__ gate, float *__restrict__ beta,
                         int n, int h_v, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 8 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        /*
        DPCT1098:450: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wv = *(w4 + j);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const sycl::float4 xa =
                *reinterpret_cast<const sycl::float4 *>(xt + j * 8);
            const sycl::float4 xb =
                *reinterpret_cast<const sycl::float4 *>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u),
                              (float)(xa.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u),
                              (float)(xa.w()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u),
                              (float)(xb.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u),
                              (float)(xb.w()), a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

/*
DPCT1110:128: The total declared local variable size in device function
gdn_step_norm_multi_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gdn_step_norm_multi_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T,
    const int32_t *__restrict__ n_keep, int t_out_begin, float *sk, float *sq,
    float red[4 /*RG*/][128 /*S*/], float *wsum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118:129: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:451: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous token is done with sk/sq/red/wsum
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        /*
        DPCT1118:130: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:452: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        /*
        DPCT1118:131: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:453: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        /*
        DPCT1118:132: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:454: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        red[rg][col] = o;
        /*
        DPCT1118:133: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:455: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), sq_part,
                o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        /*
        DPCT1118:134: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:456: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
/*
 * The staged verify path keeps the per-token values in SLM before entering the
 * recurrence.  `red_kv` has two parity slots: after the kv reduction is read,
 * the same slot is reused for `o`, while the next token writes the other slot.
 * There is no per-work-item array beyond the original state fragment.
 */
void gdn_step_norm_multi_stage_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T, int t_out_begin,
    float *sk, float *sq, float *v, float *g_stage, float *beta_stage,
    float *red_kv, float *wsum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];

    // Every work-group reads exactly the same n (T), and every work-item
    // reaches this single barrier before any staged value is consumed.
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        if (tid < S) {
            sk[(size_t) t * S + tid] = ht[qk + qh * S + tid];
            sq[(size_t) t * S + tid] = ht[qh * S + tid];
            v[(size_t) t * S + tid] = ht[2 * qk + head * S + tid];
        }
        if (tid == 0) {
            g_stage[t] = sycl::native::exp(gate[(size_t) t * h_v + head]);
            beta_stage[t] = beta[(size_t) t * h_v + head];
        }
    }
    item_ct1.barrier();

    for (int t = 0; t < n; ++t) {
        const size_t sk_off = (size_t) t * S;
        const size_t red_off = (size_t) (t & 1) * RG * S;
        const float g = g_stage[t];
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[sk_off + rg * RPG + r], kv);
        red_kv[red_off + rg * S + col] = kv;
        item_ct1.barrier();

        const float kv_col = red_kv[red_off + col] +
                             red_kv[red_off + S + col] +
                             red_kv[red_off + 2 * S + col] +
                             red_kv[red_off + 3 * S + col];
        const float delta =
            (v[(size_t) t * S + col] - g * kv_col) * beta_stage[t];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r],
                             sk[sk_off + rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[sk_off + rg * RPG + r], o);
        }
        // The kv slot is no longer read after this point.  Reusing it for o
        // keeps the local allocation at two parity slots.
        red_kv[red_off + rg * S + col] = o;
        item_ct1.barrier();

        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_kv[red_off + col] +
                  red_kv[red_off + S + col] +
                  red_kv[red_off + 2 * S + col] +
                  red_kv[red_off + 3 * S + col]) *
                 sycl::rsqrt((float) S);
            sq_part = oc * oc;
        }
        const bool emit = t >= t_out_begin;
        const size_t wsum_off = (size_t) (t & 1) * (S * RG / 32);
        if (emit) {
            for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
                dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    sq_part, o2);
            if ((tid & 31) == 0)
                wsum[wsum_off + (tid >> 5)] = sq_part;
        }
        // emit is work-group uniform; every work-item takes this barrier.
        // wsum is parity-buffered because rg == 0 reads it after this
        // barrier while other work-items may already start the next token.
        item_ct1.barrier();
        if (emit && rg == 0) {
            const float ss = wsum[wsum_off + 0] + wsum[wsum_off + 1] +
                             wsum[wsum_off + 2] + wsum[wsum_off + 3];
            const float scale = sycl::rsqrt(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t) t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
}

/*
 * The lean commit path has the same SLM prefetch and kv recurrence as the
 * staged verify path, but its caller discards y.  It therefore stops after
 * the kv reduction and state update; one barrier remains per token.
 */
void gdn_step_norm_multi_lean_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    int h_k, int h_v, const int32_t *__restrict__ n_keep,
    float *sk, float *sq, float *v, float *g_stage, float *beta_stage,
    float *red_kv) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = *n_keep;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];

    // n_keep is device data but is WG-uniform by contract.
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        if (tid < S) {
            sk[(size_t) t * S + tid] = ht[qk + qh * S + tid];
            sq[(size_t) t * S + tid] = ht[qh * S + tid];
            v[(size_t) t * S + tid] = ht[2 * qk + head * S + tid];
        }
        if (tid == 0) {
            g_stage[t] = sycl::native::exp(gate[(size_t) t * h_v + head]);
            beta_stage[t] = beta[(size_t) t * h_v + head];
        }
    }
    item_ct1.barrier();

    for (int t = 0; t < n; ++t) {
        const size_t sk_off = (size_t) t * S;
        const size_t red_off = (size_t) (t & 1) * RG * S;
        const float g = g_stage[t];
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[sk_off + rg * RPG + r], kv);
        red_kv[red_off + rg * S + col] = kv;
        item_ct1.barrier();

        const float kv_col = red_kv[red_off + col] +
                             red_kv[red_off + S + col] +
                             red_kv[red_off + 2 * S + col] +
                             red_kv[red_off + 3 * S + col];
        const float delta =
            (v[(size_t) t * S + col] - g * kv_col) * beta_stage[t];
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            s[r] = sycl::fma(g, s[r],
                             sk[sk_off + rg * RPG + r] * delta);
    }
    if (n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product =
        sycl::ext::intel::math::fmul_rn((float)(code + code_bias), sc[group]);
    out[(size_t)t * n + i] =
        sycl::ext::intel::math::fadd_rn(product, of ? of[group] : 0.0f);
}

void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        dst[i] = src[(size_t) idx * stride + i];
}

void fetch_blobs_kernel(const unsigned long long *__restrict__ src,
                        const int32_t *__restrict__ n,
                        sycl::uint4 *__restrict__ dst, long long per) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long)*n * per;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4 *)src[k])[off];
    }
}

void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = n * row_e;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) ids[i] = table[ids[i]];
}

void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs, float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = item_ct1.get_local_id(2); i < n_vocab;
         i += item_ct1.get_local_range(2)) s += sycl::native::exp(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), s, o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = s;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int)(item_ct1.get_local_range(2) >> 5); ++w) tot +=
            part[w];
        probs[t] = 1.0f / tot;
    }
}

void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = *row_dev;
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < stride; i += (int64_t)item_ct1.get_group_range(2) *
                          item_ct1.get_local_range(2))
        R_dst[i] = R_src[(size_t) row * stride + i];
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
    }
}

void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        auto blob_bytes_ct3 = (long long)(blob_bytes / 16);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 48 * 8) *
                                               sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             fetch_blobs_kernel(src, n, (sycl::uint4 *)dst,
                                                blob_bytes_ct3);
                         });
    });
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                           rebase_ptrs_kernel(ptr, n, (unsigned long long)base,
                                              (long long)blob_bytes);
                       });
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, (unsigned)n_tok,
                               (unsigned)((n_embd * hc + 255) / 256)) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                add_streams_broadcast_kernel(h, e, R, n_embd, hc);
            });
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    /*
    DPCT1049:135: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                                         sycl::range<3>(1, 1, 1024)),
                       [=](sycl::nd_item<3> item_ct1) {
                           ident_hits_kernel(ids, n, slot, dst, count);
                       });
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 16) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           mtp_select_kernel(R_src, R_stride, ids, row_dev,
                                             R_dst, tok_dst, out, j, probs,
                                             out_p);
                       });
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    dpct::queue_ptr s = (dpct::queue_ptr)stream;
    if (row_bytes % 16 == 0)
        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 16;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 48 * 8) *
                                                   sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 gather_rows_kernel((const sycl::uint4 *)src,
                                                    row_bytes_ct1, ids, n,
                                                    (sycl::uint4 *)dst);
                             });
        });
    else if (row_bytes % 4 == 0)
        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 4;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 48 * 8) *
                                                   sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 gather_rows_kernel((const uint32_t *)src,
                                                    row_bytes_ct1, ids, n,
                                                    (uint32_t *)dst);
                             });
        });
    else
        s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 48 * 8) *
                                              sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            gather_rows_kernel(src, row_bytes, ids, n, dst);
                        });
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 64),
                                         sycl::range<3>(1, 1, 64)),
                       [=](sycl::nd_item<3> item_ct1) {
                           map_ids_kernel(ids, table, n);
                       });
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    /*
    DPCT1049:136: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> part_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_rows) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                row_top_prob_kernel(
                    logits, n_vocab, ids, probs,
                    part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    check("row_top_prob");
}

namespace {
void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    /*
    DPCT1065:457: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (unsigned)n, 8) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           window_ids_kernel(steps, window, ids,
                                             (long long)ids_stride);
                       });
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 64),
                                         sycl::range<3>(1, 1, 64)),
                       [=](sycl::nd_item<3> item_ct1) {
                           dense_steps_kernel(cells, n, steps);
                       });
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:560: 'S / 32' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> part_acc_ct1(
            sycl::range<1>(4 /*S / 32*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, (unsigned)n_tok, (unsigned)(channels / S)) *
                    sycl::range<3>(1, 1, S),
                sycl::range<3>(1, 1, S)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_conv_l2_multi_kernel(
                    history, qkv, conv_w, h, channels, qk_heads, eps, t_begin,
                    part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (unsigned)((channels + 255) / 256)) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                gdn_conv_commit_kernel(history, qkv, channels, n_keep);
            });
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (unsigned)((2 * h_v + 7) / 8)) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_ab_multi_kernel(x, w_alpha, w_beta, dt, ssm_a, gate, beta,
                                    n_embd, h_v, n_tok);
            });
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxWindow) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    const bool stage_on =
        n_keep == nullptr && gdn_env_is_one("STRATA_GDN_STAGE");
    const bool lean_on =
        n_keep != nullptr && gdn_env_is_one("STRATA_GDN_LEAN_COMMIT");
    if (stage_on) {
        ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sk_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> sq_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> v_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> g_acc_ct1(
                sycl::range<1>(kVerifyMaxWindow), cgh);
            sycl::local_accessor<float, 1> beta_acc_ct1(
                sycl::range<1>(kVerifyMaxWindow), cgh);
            sycl::local_accessor<float, 1> red_kv_acc_ct1(
                sycl::range<1>((size_t) 2 * RG * S), cgh);
            sycl::local_accessor<float, 1> wsum_acc_ct1(
                sycl::range<1>((size_t) 2 * S * RG / 32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned) h_v) *
                                      sycl::range<3>(1, RG, S),
                                  sycl::range<3>(1, RG, S)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_stage_kernel(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, t_out_begin,
                            sk_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            sq_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            v_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            g_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            beta_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            red_kv_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            wsum_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get());
                    });
        });
        check("gdn_step_norm_multi_stage");
        return;
    }
    if (lean_on) {
        ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sk_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> sq_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> v_acc_ct1(
                sycl::range<1>((size_t) kVerifyMaxWindow * S), cgh);
            sycl::local_accessor<float, 1> g_acc_ct1(
                sycl::range<1>(kVerifyMaxWindow), cgh);
            sycl::local_accessor<float, 1> beta_acc_ct1(
                sycl::range<1>(kVerifyMaxWindow), cgh);
            sycl::local_accessor<float, 1> red_kv_acc_ct1(
                sycl::range<1>((size_t) 2 * RG * S), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned) h_v) *
                                      sycl::range<3>(1, RG, S),
                                  sycl::range<3>(1, RG, S)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_lean_kernel(
                            state, h, conv_channels, gate, beta, h_k, h_v,
                            n_keep,
                            sk_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            sq_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            v_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            g_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            beta_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get(),
                            red_kv_acc_ct1
                                .get_multi_ptr<
                                    sycl::access::decorated::no>()
                                .get());
                    });
        });
        check("gdn_step_norm_multi_lean");
        return;
    }
    /*
    DPCT1049:137: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:561: 'S' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> sk_acc_ct1(sycl::range<1>(128 /*S*/),
                                                  cgh);
        /*
        DPCT1101:562: 'S' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> sq_acc_ct1(sycl::range<1>(128 /*S*/),
                                                  cgh);
        /*
        DPCT1101:563: 'RG' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:564: 'S' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0> red_acc_ct1(cgh);
        /*
        DPCT1101:565: 'S * RG / 32' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> wsum_acc_ct1(
            sycl::range<1>(16 /*S * RG / 32*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)h_v) *
                                  sycl::range<3>(1, RG, S),
                              sycl::range<3>(1, RG, S)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_step_norm_multi_kernel(
                    state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k,
                    h_v, n_tok, n_keep, t_out_begin,
                    sk_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    sq_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    red_acc_ct1,
                    wsum_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    check("gdn_step_norm_multi");
}

namespace {
void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    sx::annotated_ptr<uint32_t, uncached_l1l3> flag_read(const_cast<uint32_t *>(flag));
    while (*flag_read < value) {
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
}  // namespace

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                wait_flag_ge_kernel(flag, value);
            });
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, (unsigned)n_tok,
                                             (unsigned)((n + 255) / 256)) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                embedding_gather_dev_kernel(codes, scales, offsets, tokens, n,
                                            code_bits, code_bias, group_elems,
                                            row_codes, row_groups, out);
            });
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, (unsigned)n_tok,
                               (unsigned)((n_embd * hc + 255) / 256)) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                broadcast_streams_kernel(x, R, n_embd, hc);
            });
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           copy_indexed_kernel(dst, src, stride, index, n);
                       });
    check("copy_indexed");
}

}  // namespace strata::kernels
