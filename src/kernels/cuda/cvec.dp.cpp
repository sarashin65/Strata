// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/cvec.hpp"

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;
int* g_on = nullptr;
bool g_on_host = false;

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// one block per (stream, token): the pending write, then h . v over the stream, then the update
/*
DPCT1110:2: The total declared local variable size in device function
cvec_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void cvec_kernel(float *__restrict__ R, const float *__restrict__ dir,
                 const float *__restrict__ s_l, const int *__restrict__ on,
                 int mode, int64_t layer, int n, int hc, int64_t r_ld,
                 const float *__restrict__ bo, int64_t bo_ld,
                 const float *__restrict__ inj, int64_t inj_ld, int write,
                 float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2);
    const int64_t t = item_ct1.get_group(1);
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = sycl::fma((float)(b[d]), (float)w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = sycl::fma(xv, (float)(v[d]), dot);
        }
    }
    if (steer && mode == 0) {

#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dot, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = dot;
        /*
        DPCT1118:3: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:178: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) < 32) {
            float p = item_ct1.get_local_id(2) < THREADS / 32
                          ? part[item_ct1.get_local_id(2)]
                          : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p +=
                dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), p, o);
            if (item_ct1.get_local_id(2) == 0) part[0] = p;
        }
        /*
        DPCT1118:4: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:179: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv =
                mode == 0 ? sycl::fma(-dot, (float)(v[d]), xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float> &dir, const std::vector<float> &s,
                 int mode, int first, int last, int64_t n_embd, int64_t hc,
                 std::string &err) try {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    float* d_dir = nullptr;
    float* d_s = nullptr;
    int* d_on = nullptr;
    const int one = 1;
    if (DPCT_CHECK_ERROR(d_dir = sycl::malloc_device<float>(
                             dir.size(), dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(d_s = sycl::malloc_device<float>(
                             s.size(), dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(d_on = sycl::malloc_device<int>(
                             1, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(d_dir, dir.data(), dir.size() * sizeof(float))
                .wait()) != 0 ||
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_s, s.data(), s.size() * sizeof(float))
                             .wait()) != 0 ||
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_on, &one, sizeof(int))
                             .wait()) != 0) {
        err = "control vector: device allocation failed";
        return false;
    }
    g_cvec.dir = d_dir;
    g_cvec.s = d_s;
    g_cvec.on = d_on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    g_on = d_on;
    g_on_host = true;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cvec_set_enabled(bool on) {
    if (g_on == nullptr || on == g_on_host) return;
    dpct::get_current_device()
        .queues_wait_and_throw(); // nothing in flight may still read the flag
    const int v = on ? 1 : 0;
    dpct::get_in_order_queue().memcpy(g_on, &v, sizeof(int)).wait();
    g_on_host = on;
}

bool cvec_enabled() { return g_on != nullptr && g_on_host; }

void cvec_apply(float *R, int64_t layer, int64_t T, int64_t r_ld,
                const float *bo, int64_t bo_ld, const float *inj,
                int64_t inj_ld, bool write, void *stream) try {
    if (!g_cvec.loaded() || T < 1) return;
    const dpct::dim3 grid((unsigned)g_cvec.hc, (unsigned)T);
    ((sycl::queue *)((dpct::queue_ptr)stream))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:554: 'THREADS / 32' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        sycl::local_accessor<float, 1> part_acc_ct1(
            sycl::range<1>(8 /*THREADS / 32*/), cgh);

        auto g_cvec_dir_ct1 = g_cvec.dir;
        auto g_cvec_s_ct2 = g_cvec.s;
        auto g_cvec_on_ct3 = g_cvec.on;
        auto g_cvec_mode_ct4 = g_cvec.mode;
        auto g_cvec_n_embd_ct6 = (int)g_cvec.n_embd;
        auto g_cvec_hc_ct7 = (int)g_cvec.hc;

        cgh.parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                cvec_kernel(
                    R, g_cvec_dir_ct1, g_cvec_s_ct2, g_cvec_on_ct3,
                    g_cvec_mode_ct4, layer, g_cvec_n_embd_ct6, g_cvec_hc_ct7,
                    r_ld, bo, bo_ld, inj, inj_ld, write ? 1 : 0,
                    part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:180: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaPeekAtLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (0 != 0) throw std::runtime_error("cvec_apply: launch failed");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
