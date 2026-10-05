#pragma once

#include "ggml-sycl/dpct/helper.hpp"

namespace strata::prefill::mmq::detail {

// Opaque host-side context replacing the CUDA backend context used only by the
// borrowed MMQ launch path. The implementation owns a llama.cpp SYCL pool.
void * new_context(void * stream);
void delete_context(void * ctx) noexcept;
dpct::queue_ptr context_stream(void * ctx) noexcept;

}  // namespace strata::prefill::mmq::detail
