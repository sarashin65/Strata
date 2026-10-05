// src/prefill/ggml_cuda_host.dp.cpp
//
// Host-side adapter for the SYCL MMVQ path.  The former implementation
// exposed llama.cpp's CUDA context and cache pool to CUDA MMQ templates.  The
// replacement keeps the same lifetime boundary without including CUDA headers:
// llama.cpp's SYCL pool interface owns device USM allocations and the context
// uses the pinned in-order queue as its default stream.
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include <sycl/sycl.hpp>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common-upstream.h"

#include "ggml-sycl/common.hpp"
#include "strata/prefill/sycl_mmq_adapter.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::prefill::mmq::detail {
namespace {

class CachingPool final : public ggml_sycl_pool {
public:
    explicit CachingPool(dpct::queue_ptr stream) : stream_(stream) {}

    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto & buffer : buffers_) {
            if (!buffer.used && buffer.size >= size) {
                buffer.used = true;
                *actual_size = buffer.size;
                return buffer.ptr;
            }
        }

        if (size == 0) {
            *actual_size = 0;
            return nullptr;
        }

        void * ptr = sycl::malloc_device(size, *stream_);
        if (ptr == nullptr) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                  "SYCL MMQ pool allocation failed");
        }
        buffers_.push_back({ptr, size, true});
        *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t) override {
        if (ptr == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        for (auto & buffer : buffers_) {
            if (buffer.ptr == ptr) {
                buffer.used = false;
                return;
            }
        }
    }

    ~CachingPool() override {
        for (const auto & buffer : buffers_) {
            if (buffer.ptr != nullptr) {
                sycl::free(buffer.ptr, *stream_);
            }
        }
    }

private:
    struct Buffer {
        void * ptr;
        size_t size;
        bool used;
    };

    dpct::queue_ptr stream_;
    std::vector<Buffer> buffers_;
    std::mutex mu_;
};

struct SyclContext {
    dpct::queue_ptr stream;
    std::unique_ptr<CachingPool> pool;

    explicit SyclContext(dpct::queue_ptr q) : stream(q), pool(q != nullptr ? new CachingPool(q) : nullptr) {
        if (q == nullptr) std::abort();
    }
};

}  // namespace

void * new_context(void * stream) {
    if (stream == nullptr) std::abort();
    return new SyclContext(static_cast<dpct::queue_ptr>(stream));
}

void delete_context(void * ctx) noexcept {
    delete static_cast<SyclContext *>(ctx);
}

dpct::queue_ptr context_stream(void * ctx) noexcept {
    if (ctx == nullptr) std::abort();
    return static_cast<SyclContext *>(ctx)->stream;
}

}  // namespace strata::prefill::mmq::detail
