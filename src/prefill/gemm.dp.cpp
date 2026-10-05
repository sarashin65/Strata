// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <dpct/blas_utils.hpp>

#include <cstdio>
#include <cstdlib>
#include <dpct/lib_common_utils.hpp>

namespace strata::prefill {
namespace {

void ck(int s, const char *what) {
    if (s != 0) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) delete ((dpct::blas::descriptor_ptr)handle_);
    if (!external_) {
        if (scratch_) dpct::dpct_free(scratch_, dpct::get_in_order_queue());
        if (workspace_) dpct::dpct_free(workspace_, dpct::get_in_order_queue());
    }
}

bool Gemm::init_external(void *stream, uint16_t *scratch, int64_t scratch_elems,
                         void *workspace, size_t ws_bytes,
                         std::string &err) try {
    dpct::blas::descriptor_ptr h = nullptr;
    if (DPCT_CHECK_ERROR(h = new dpct::blas::descriptor()) != 0) {
        err = "prefill gemm: cublasCreate failed"; return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    h->set_queue((dpct::queue_ptr)stream);
    workspace_ = workspace;
    /*
    DPCT1026:459: The call to cublasSetWorkspace was removed because this
    functionality is redundant in SYCL.
    */
    h->set_math_mode(dpct::blas::math_mode::mm_default);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    /*
    DPCT1026:460: The call to cublasSetWorkspace was removed because this
    functionality is redundant in SYCL.
    */
}

bool Gemm::init(void *stream, int64_t scratch_elems, std::string &err) try {
    dpct::blas::descriptor_ptr h = nullptr;
    if (DPCT_CHECK_ERROR(h = new dpct::blas::descriptor()) != 0) {
        err = "prefill gemm: cublasCreate failed"; return false;
    }
    handle_ = h;
    stream_ = stream;
    h->set_queue((dpct::queue_ptr)stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (DPCT_CHECK_ERROR(workspace_ = (void *)sycl::malloc_device(
                             ws, dpct::get_in_order_queue())) != 0) {
        err = "prefill gemm: workspace"; return false;
    }
    /*
    DPCT1026:461: The call to cublasSetWorkspace was removed because this
    functionality is redundant in SYCL.
    */
    h->set_math_mode(dpct::blas::math_mode::mm_default);
    if (scratch_elems > 0 &&
        DPCT_CHECK_ERROR(
            scratch_ = (uint16_t *)sycl::malloc_device(
                (size_t)scratch_elems * 2, dpct::get_in_order_queue())) != 0) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
           dpct::library_data_t::real_bfloat16, (int)K, X,
           dpct::library_data_t::real_bfloat16, (int)K, &beta, Y,
           dpct::library_data_t::real_float, (int)ldy,
           dpct::compute_type::f32)),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
           dpct::library_data_t::real_half, (int)K, X,
           dpct::library_data_t::real_half, (int)K, &beta, Y,
           dpct::library_data_t::real_float, (int)ldy,
           dpct::compute_type::f32)),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

void Gemm::native_reordered_q6(const uint16_t* X, const void* W_reordered, float* Y, int64_t T, int64_t N, int64_t K,
                               int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16_q6_reordered(W_reordered, r0, n, N, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16_q6_reordered(W_reordered, 0, N, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}
void Gemm::native_reordered(const uint16_t* X, int ggml_type, const void* W_reordered, float* Y,
                            int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long)K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16_reordered(ggml_type, W_reordered, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16_reordered(ggml_type, W_reordered, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
