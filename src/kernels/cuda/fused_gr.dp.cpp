// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

static bool gr_split_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("STRATA_GR_SPLIT");
        return value == nullptr or value[0] != 48;
    }();
    return enabled;
}
static bool gr_norm_ilp_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("STRATA_GR_NORM_ILP");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
static bool gr_tred_on() {
    static const bool enabled = [] {
        const char *value = std::getenv("STRATA_GR_TRED");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
// V (power of two, 2..32) values reduced at once.  Each transposed step
// pairs lane L with L^o exactly as warp_sum does: keep + received is the
// scalar warp_sum addition for the same value and in the same order.
template <int V>
__dpct_inline__ void warp_sum_multi(float (&v)[V], int lane) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int o = 16 >> st, w = V >> st;           // both constant after unrolling
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
template <int V> __dpct_inline__ int warp_sum_multi_index(int lane) {
    int idx = 0;
#pragma unroll
    for (int st = 0; (V >> st) > 1; ++st) idx = 2 * idx + ((lane >> (4 - st)) & 1);
    return idx;
}

__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// 8 bf16 packed in a uint4 against 8 floats.
__dpct_inline__ float dot8(const sycl::uint4 w, const float *x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), (float)(x[2 * j]),
                        acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u),
                        (float)(x[2 * j + 1]), acc);
    }
    return acc;
}

/*
DPCT1110:14: The total declared local variable size in device function
gr_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_kernel(FusedGrArgs a, float *xn, float part[8 /*WARPS*/][4 /*HC*/],
                    float *s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        if (item_ct1.get_group(2) == 0) a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:204: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    /*
    DPCT1065:205: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    /*
    DPCT1098:206: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    for (int j = lane; j < D / 8; j += 32) acc += dot8(*(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + sycl::native::exp(-x));
    }
}

void gr_up_kernel(FusedGrArgs a, float *lo,
                                             float g[4/*HC*/][32/*UP_COLS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    /*
    DPCT1065:207: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(a.w_up + (size_t)i * LR);
        /*
        DPCT1098:209: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float acc = dot8(*(w4 + lane), lo + lane * 8);
        /*
        DPCT1098:210: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        if (lane < LR / 8 - 32) acc +=
            dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = sycl::fma((float)(a.bo_prev[d0 + dd]),
                               2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    /*
    DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
/*
DPCT1110:15: The total declared local variable size in device function
gr_norm_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_norm_multi_kernel(GrMulti m, float part[8 /*WARPS*/][4 /*HC*/],
                          float *s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:211: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:212: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}
// Norm variant that stages every R, bo_prev, and w_norm chunk before computing.
void gr_norm_multi_ilp_kernel(GrMulti m, float part[8 /*WARPS*/][4 /*HC*/],
                              float *s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;

    sycl::float4 r[10], b[10], g[10], prod[10];
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const int d = i - (i / N) * N;
        r[k] = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            b[k] = *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
        }
        g[k] = *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
    }
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const int c = i / N;
        if (a.apply) {
            r[k].x() = sycl::fma((float)(b[k].x()), gw[c], r[k].x());
            r[k].y() = sycl::fma((float)(b[k].y()), gw[c], r[k].y());
            r[k].z() = sycl::fma((float)(b[k].z()), gw[c], r[k].z());
            r[k].w() = sycl::fma((float)(b[k].w()), gw[c], r[k].w());
        }
        float sq =
            r[k].x() * r[k].x() + r[k].y() * r[k].y() + r[k].z() * r[k].z() + r[k].w() * r[k].w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        prod[k] = sycl::float4(
            r[k].x() * g[k].x(), r[k].y() * g[k].y(), r[k].z() * g[k].z(), r[k].w() * g[k].w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:213: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:214: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const int c = i / N;
        const float s = s_rs[c];
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            prod[k].x() * s, prod[k].y() * s, prod[k].z() * s, prod[k].w() * s);
    }
}


constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's 10 weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
/*
DPCT1110:16: The total declared local variable size in device function
gr_down_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_multi_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto tile = (float *)dpct_local; // [T][TILE]
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILE) {
        sycl::uint4 wv[TQ];
        if (active) {
#pragma unroll
            /*
            DPCT1098:215: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < TQ; ++q)
                wv[q] = *(w4 + base / 8 + lane + 32 * q);
        }
        /*
        DPCT1118:17: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:213: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile is consumed
        const sycl::float4 *src4 = reinterpret_cast<const sycl::float4 *>(m.xn);
        sycl::float4 *tile4 = reinterpret_cast<sycl::float4 *>(tile);
        for (int i = t; i < T * (TILE / 4); i += THREADS) {
            const int k = i / (TILE / 4), off = i - k * (TILE / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        /*
        DPCT1118:18: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:214: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILE + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

template <bool TRED>
void gr_down_split_multi_kernel(GrMulti m, float *partial) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int group = item_ct1.get_group(2);
    const bool inject_block = group >= LR;
    const int row = inject_block ? group - LR : group;
    const bool active = !inject_block || m.a[0].w_inject != nullptr;
    const uint16_t *wbase = active ? (inject_block ? m.a[0].w_inject : m.a[0].w_down) : m.a[0].w_down;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wbase + (size_t)(active ? row : 0) * D);
    for (int s = 5 * warp; s < 5 * warp + 5; ++s) {
        const int base = (s / TQ) * TILE;
        const int q = s % TQ;
        const int j = lane + 32 * q;
        const sycl::uint4 wv = active ? *(w4 + base / 8 + j) : sycl::uint4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= m.T) break;
            partial[(k * 40 + s) * 32 + lane] = dot8(
                wv, m.xn + (size_t)k * D + base + j * 8);
        }
    }
    item_ct1.barrier();
    if constexpr (TRED) {
        // Keep the same 40 terms and order; only the warp assigned to token k changes.
        if (!active) return;
        if (warp < m.T) {
            const int k = warp;
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 40; ++s)
                acc += partial[(k * 40 + s) * 32 + lane];
            const float sum = warp_sum(acc);
            if (lane == k) {
                if (inject_block) {
                    m.a[k].inject_out[row] = sum;
                } else {
                    const float x = sum / (float)HC;
                    m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
                }
            }
        }
    } else {
        if (!active || warp != 0) return;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= m.T) break;
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 40; ++s)
                acc += partial[(k * 40 + s) * 32 + lane];
            const float sum = warp_sum(acc);
            if (lane == k) {
                if (inject_block) {
                    m.a[k].inject_out[row] = sum;
                } else {
                    const float x = sum / (float)HC;
                    m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
                }
            }
        }
    }
}


constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
/*
DPCT1110:19: The total declared local variable size in device function
gr_up_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
template <bool TRED>

void gr_up_multi_kernel(
    GrMulti m, float lo[8 /*kFusedGrMaxT*/][320 /*LR*/],
    float g[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    /*
    DPCT1065:216: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098:218: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098:219: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
        if constexpr (TRED) {
            float v[kFusedGrMaxT];
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) {
                float acc = 0.0f;
                if (k < T) {
                    acc = dot8(wa, lo[k] + lane * 8);
                    if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
                }
                v[k] = acc;
            }
            // The transposed stages retain each token's warp_sum pair/order; select only copies v[0].
            warp_sum_multi<kFusedGrMaxT>(v, lane);
            const int src = ((lane >> 2) & 1) << 4 | ((lane >> 1) & 1) << 3 | (lane & 1) << 2;
            mine = dpct::select_from_sub_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), v[0], src);
        } else {
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) {
                if (k >= T) break;
                float acc = dot8(wa, lo[k] + lane * 8);
                if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
                acc = warp_sum(acc);
                if (lane == k) mine = acc;
            }
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }

    }
    /*
    DPCT1065:217: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    const bool ilp = gr_norm_ilp_enabled();
    const bool tred = gr_tred_on();

    dpct::queue_ptr st = (dpct::queue_ptr)stream;
    st->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:570: 'WARPS' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:571: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[8 /*WARPS*/][4 /*HC*/], 0> part_acc_ct1(cgh);
        /*
        DPCT1101:572: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> s_rs_acc_ct1(sycl::range<1>(4 /*HC*/),
                                                    cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_tok) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                if (ilp) {
                    gr_norm_multi_ilp_kernel(
                        m, part_acc_ct1,
                        s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                } else {
                    gr_norm_multi_kernel(
                        m, part_acc_ct1,
                        s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                }
            });
    });
    static bool attr = false;
    if (!attr) {
        /*
        DPCT1026:220: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
        attr = true;
    }
if (gr_split_enabled()) {
        st->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> split_acc(sycl::range<1>((size_t)n_tok * 40u * 32u), cgh);
            if (tred) {
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (LR + HC) * THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_split_multi_kernel<true>(
                            m, split_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            } else {
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, (LR + HC) * THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_split_multi_kernel<false>(
                            m, split_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            }
        });

    } else {
        st->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:573: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>((size_t)n_tok * TILE * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_multi_kernel(
                    m, dpct_local_acc_ct1
                           .get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
    }
    st->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:574: 'kFusedGrMaxT' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:575: 'LR' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[8 /*kFusedGrMaxT*/][320 /*LR*/], 0>
            lo_acc_ct1(cgh);
        /*
        DPCT1101:576: 'kFusedGrMaxT' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:577: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:578: 'UPM_COLS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<
            float[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/], 0>
            g_acc_ct1(cgh);

        if (tred) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, UPM_BLOCKS) *
                                       sycl::range<3>(1, 1, THREADS),
                                   sycl::range<3>(1, 1, THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<true>(m, lo_acc_ct1, g_acc_ct1);
                    });
        } else {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, UPM_BLOCKS) *
                                       sycl::range<3>(1, 1, THREADS),
                                   sycl::range<3>(1, 1, THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<false>(m, lo_acc_ct1, g_acc_ct1);
                    });
        }
    });
    /*
    DPCT1010:221: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    const bool split = gr_split_enabled();
    dpct::queue_ptr st = (dpct::queue_ptr)stream;
    if (!split || a.xn == nullptr) {
        st->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:579: 'D' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> xn_acc_ct1(sycl::range<1>(10240 /*D*/),
                                                  cgh);
        /*
        DPCT1101:580: 'WARPS' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:581: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[8 /*WARPS*/][4 /*HC*/], 0> part_acc_ct1(cgh);
        /*
        DPCT1101:582: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> s_rs_acc_ct1(sycl::range<1>(4 /*HC*/),
                                                    cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_kernel(
                    a,
                    xn_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    part_acc_ct1,
                    s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    } else {
        GrMulti m;
        m.a[0] = a;
        m.xn = a.xn;
        m.T = 1;
        st->submit([=](sycl::handler &cgh) {
            sycl::local_accessor<float[8 /*WARPS*/][4 /*HC*/], 0> part_acc_ct1(cgh);
            sycl::local_accessor<float, 1> s_rs_acc_ct1(sycl::range<1>(4 /*HC*/), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS),
                                  sycl::range<3>(1, 1, THREADS)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    gr_norm_multi_kernel(
                        m, part_acc_ct1,
                        s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
        st->submit([=](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> split_acc(sycl::range<1>(40u * 32u), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (LR + HC) * THREADS),
                                  sycl::range<3>(1, 1, THREADS)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    gr_down_split_multi_kernel<false>(
                        m, split_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
    st->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:583: 'LR' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> lo_acc_ct1(sycl::range<1>(320 /*LR*/),
                                                  cgh);
        /*
        DPCT1101:584: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:585: 'UP_COLS' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*HC*/][32 /*UP_COLS*/], 0> g_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, UP_BLOCKS) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_kernel(
                    a,
                    lo_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    g_acc_ct1);
            });
    });
    /*
    DPCT1010:223: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

}  // namespace strata::kernels
