// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa_select.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__dpct_inline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

void block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                      (item_ct1.get_local_id(2) >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}
constexpr int MQ = 8;

void block_scores_multi_kernel(const float* __restrict__ pooled,
                               const float* __restrict__ dead,
                               const float* __restrict__ q_idx,
                               const int32_t* __restrict__ steps, int nq,
                               int64_t max_blocks, float* __restrict__ out,
                               float* qs, int64_t* s_nkv, int64_t* s_nbid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    for (int i = t; i < nq * IDX_HEADS * IDX_DIM; i += 256) qs[i] = q_idx[i];
    if (t < nq) {
        s_nkv[t] = steps[t * kStepCount + kStepNKv];
        s_nbid[t] = steps[t * kStepCount + kStepNBid];
    }
    item_ct1.barrier();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q)
        top = s_nbid[q] > top ? s_nbid[q] : top;
    const int lane = t & 31;
    // The stride is the full grid width; using this group's id would make group 0 loop forever.
    const int64_t group = (int64_t)item_ct1.get_group(2);
    const int64_t wstride =
        (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = group * SCORE_WARPS + (t >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp =
            *reinterpret_cast<const sycl::float4 *>(
                pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd =
            *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() +
                          k4.z() * q4.z() + k4.w() * q4.w();
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    d += dpct::permute_sub_group_by_xor(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}


/*
DPCT1110:82: The total declared local variable size in device function
block_topk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void block_topk_kernel(const float *__restrict__ scores,
                       const int32_t *__restrict__ steps, int64_t max_blocks,
                       int64_t cap, int32_t *__restrict__ ids, int *hist,
                       int *s_a, int *s_b, int &s_digit, int &s_above) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        /*
        DPCT1118:83: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:389: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], w);
        }
        /*
        DPCT1118:84: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:390: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        /*
        DPCT1118:85: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:391: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        /*
        DPCT1118:86: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:392: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    /*
    DPCT1065:384: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    /*
    DPCT1065:385: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    /*
    DPCT1065:386: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    s_a[t] = sel;
    /*
    DPCT1065:387: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    /*
    DPCT1065:388: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
constexpr size_t TK_REG_LOCAL_BYTES =
    (TK_T / 32) * 256 * sizeof(int) + 33 * sizeof(int) +
    2 * sizeof(int) + 1024;

int block_excl_scan(int v, int* s_warp, int& total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    item_ct1.barrier();
    int w = warp == 0 ? s_warp[lane] : 0;
    int z = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::shift_sub_group_left(
            sycl::ext::oneapi::this_work_item::get_sub_group(), z, o);
        if (warp == 0 && lane >= o) z += y;
    }
    if (warp == 0) {
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    item_ct1.barrier();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    item_ct1.barrier();
    return r;
}

void block_topk_reg_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ steps,
    int64_t max_blocks, int64_t cap, int32_t *__restrict__ ids, int *hist,
    int *s_warp, int &s_digit, int &s_above) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    const int lane = t & 31, warp = t >> 5;
    const bool active = n_kv > width;
    if (!active) {
        for (int64_t j = t; j < n_kv; j += TK_T)
            out[j] = (int32_t)j;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = active ? n_bid + 1 : 0;
    const int64_t per = (nb + TK_T - 1) / TK_T;
    const int64_t b0 = (int64_t)t * per;
    const int64_t b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[TK_PER];
#pragma unroll
    for (int j = 0; j < TK_PER; ++j)
        key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int {
        return b < n_bid ? R : (int)(n_kv - n_bid * R);
    };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32)
            hist[warp * 256 + i] = 0;
        // The CUDA source uses __syncwarp(): this reset is warp-local.
        sycl::group_barrier(
            sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask =
            shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < TK_PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = key[j];
            if ((k & hi_mask) == (prefix & hi_mask))
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[warp * 256 + ((k >> shift) & 255)], w);
        }
        item_ct1.barrier();
        if (t < 256) {
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2)
                s += hist[w2 * 256 + t];
            hist[t] = s;
        }
        item_ct1.barrier();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        item_ct1.barrier();
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < TK_PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = key[j];
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    int total;
    const int eq_before = block_excl_scan(eq, s_warp, total);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int)my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, total);
    int64_t eq_left = my_eq;
    if (active) {
#pragma unroll
        for (int j = 0; j < TK_PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = key[j];
            if (k > thr) {
                for (int c = 0; c < w; ++c)
                    out[wpos++] = (int32_t)(b * R + c);
            } else if (k == thr) {
                for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                    out[wpos++] = (int32_t)(b * R + c);
            }
        }
    }
}


}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    static const bool scores_multi = [] {
        const char* v = std::getenv("STRATA_QSA_SCORES_MULTI");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (scores_multi && nq <= MQ) {
        ((sycl::queue *)((dpct::queue_ptr)stream))->submit(
            [&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> qs_acc_ct1(
                    sycl::range<1>(MQ * IDX_HEADS * IDX_DIM), cgh);
                sycl::local_accessor<int64_t, 1> s_nkv_acc_ct1(
                    sycl::range<1>(MQ), cgh);
                sycl::local_accessor<int64_t, 1> s_nbid_acc_ct1(
                    sycl::range<1>(MQ), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, 256 * 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_multi_kernel(
                                pooled, dead, q_idx, steps, (int)nq,
                                max_blocks, scores,
                                qs_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                s_nkv_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                s_nbid_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                        });
            });
        return;
    }

    const dpct::dim3 grid(
        (unsigned)((max_blocks + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned)nq);
    ((sycl::queue *)((dpct::queue_ptr)stream))
        ->parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SCORE_WARPS * 32),
                              sycl::range<3>(1, 1, SCORE_WARPS * 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_kernel(pooled, dead, q_idx, steps, max_blocks,
                                    scores);
            });
    /*
    DPCT1010:393: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009:394: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    static const bool topk_reg = [] {
        const char* v = std::getenv("STRATA_QSA_TOPK_REG");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (topk_reg && max_blocks <= (int64_t)TK_T * TK_PER) {
        const sycl::device topk_reg_device =
            ((sycl::queue *)((dpct::queue_ptr)stream))->get_device();
        const bool topk_reg_device_ok =
            topk_reg_device
                    .get_info<sycl::info::device::max_work_group_size>() >=
                (size_t)TK_T &&
            topk_reg_device.get_info<sycl::info::device::local_mem_size>() >=
                TK_REG_LOCAL_BYTES;
        if (topk_reg_device_ok) {
            ((sycl::queue *)((dpct::queue_ptr)stream))->submit(
                [&](sycl::handler &cgh) {
                    sycl::local_accessor<int, 1> hist_acc_ct1(
                        sycl::range<1>((TK_T / 32) * 256), cgh);
                    sycl::local_accessor<int, 1> s_warp_acc_ct1(
                        sycl::range<1>(33), cgh);
                    sycl::local_accessor<int, 0> s_digit_acc_ct1(cgh);
                    sycl::local_accessor<int, 0> s_above_acc_ct1(cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, (unsigned)nq) *
                                sycl::range<3>(1, 1, TK_T),
                            sycl::range<3>(1, 1, TK_T)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_topk_reg_kernel(
                                    scores, steps, max_blocks, cap, ids,
                                    hist_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    s_warp_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    s_digit_acc_ct1, s_above_acc_ct1);
                            });
                });
            return;
        }
    }

    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> hist_acc_ct1(sycl::range<1>(256), cgh);
        /*
        DPCT1101:508: 'TOPK_T' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<int, 1> s_a_acc_ct1(sycl::range<1>(256 /*TOPK_T*/),
                                                 cgh);
        /*
        DPCT1101:509: 'TOPK_T' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<int, 1> s_b_acc_ct1(sycl::range<1>(256 /*TOPK_T*/),
                                                 cgh);
        sycl::local_accessor<int, 0> s_digit_acc_ct1(cgh);
        sycl::local_accessor<int, 0> s_above_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)nq) *
                                  sycl::range<3>(1, 1, TOPK_T),
                              sycl::range<3>(1, 1, TOPK_T)),
            [=](sycl::nd_item<3> item_ct1) {
                block_topk_kernel(
                    scores, steps, max_blocks, cap, ids,
                    hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_a_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_b_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_digit_acc_ct1, s_above_acc_ct1);
            });
    });
    /*
    DPCT1010:395: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009:396: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

}  // namespace strata::kernels
