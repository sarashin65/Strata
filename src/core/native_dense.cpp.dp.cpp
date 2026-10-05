#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <climits>
#include <exception>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
namespace {
bool eligible(const std::string& name, bool include_ple_key) {
    if (name.rfind("blk.", 0) != 0) return false;
    if (include_ple_key && name == "blk.1.ple_key.weight") return true;
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree {
    void
    operator()(void *p) const {
        if (p) dpct::dpct_free(p, dpct::get_in_order_queue());
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr reordered;
    bool reorder_only_safe;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor.name, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

NativeDense::~NativeDense() {
    if (scratch_) dpct::dpct_free(scratch_, dpct::get_in_order_queue());
    for (void *p : reordered_weights_)
        DPCT_CHECK_ERROR(dpct::dpct_free(p, dpct::get_in_order_queue()));
    for (void *p : weights_)
        DPCT_CHECK_ERROR(dpct::dpct_free(p, dpct::get_in_order_queue()));
}

bool NativeDense::load(const std::vector<std::string> &shards,
                       WeightTable &table, std::string &err,
                       bool include_ple_key) try {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor.name, include_ple_key)) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status =
                    DPCT_CHECK_ERROR(allocation = (void *)sycl::malloc_device(
                                         bytes, dpct::get_in_order_queue()));
                DevicePtr data(allocation);
                if (status == 0)
                    status = DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue()
                            .memcpy(data.get(), gguf.tensor_data(tensor), bytes)
                            .wait());

                // Keep the authoritative GGUF copy and make the plane-major Q6_K copy once at startup.
                // The opt-in copy has the same byte count; unsupported dimensions leave reordered null so
                // native_mmvq_with_reorder falls through to the original path.
                DevicePtr reordered;
                if (status == 0 && tensor.type != 13 && strata::kernels::native_reorder_enabled((int) tensor.type) &&
                    strata::kernels::native_reorder_shape((int) tensor.type, (int) ref.ne0, (int) ref.ne1, 1)) {
                    const std::size_t reordered_bytes =
                        strata::kernels::native_reordered_bytes((int) tensor.type, (int) ref.ne0, (int) ref.ne1);
                    std::vector<uint8_t> host_reordered(reordered_bytes);
                    strata::kernels::native_reorder_host(
                        (int) tensor.type, gguf.tensor_data(tensor), host_reordered.data(),
                        (int) ref.ne0, (int) ref.ne1);
                    void* reordered_allocation = nullptr;
                    status = DPCT_CHECK_ERROR(reordered_allocation = (void *)sycl::malloc_device(
                                                   reordered_bytes, dpct::get_in_order_queue()));
                    reordered.reset(reordered_allocation);
                    if (status == 0)
                        status = DPCT_CHECK_ERROR(
                            dpct::get_in_order_queue()
                                .memcpy(reordered.get(), host_reordered.data(), reordered_bytes)
                                .wait());
                }
                /*
                DPCT1000:83: Error handling if-stmt was detected but could not
                be rewritten.
                */
                if (status != 0) {
                    /*
                    DPCT1009:84: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1001:82: The statement could not be removed.
                    */
                    err = "native dense upload " + tensor.name + ": " +
                          dpct::get_error_string_dummy(status);
                        return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                const bool reorder_only_safe = tensor.type == 14 || tensor.type == 12 ||
                tensor.type == 13 || tensor.type == 23;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes,
                                          std::move(data), std::move(reordered),
                                          reorder_only_safe});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status =
            DPCT_CHECK_ERROR(allocation = (void *)sycl::malloc_device(
                                 strata::kernels::native_q8_1_bytes(max_in),
                                 dpct::get_in_order_queue()));
        DevicePtr scratch(allocation);
        /*
        DPCT1009:85: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        /*
        DPCT1001:80: The statement could not be removed.
        */
        /*
        DPCT1000:81: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            err = std::string("native dense scratch: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        reordered_weights_.reserve(pending.size());
        for (auto& item : pending) {
            const bool release_original =
                item.reorder_only_safe && item.reordered &&
                strata::kernels::native_reorder_enabled(item.type) &&
                strata::kernels::native_mmvq_multi_exact() &&
                strata::kernels::native_reorder_shape(
                    item.type, static_cast<int>(item.ref->ne0), static_cast<int>(item.ref->ne1), 8);
            if (release_original) {
                item.ref->native_data = item.reordered.get();
                item.data.reset();
            } else {
                item.ref->native_data = item.data.get();
            }
            item.ref->native_reordered_data = item.reordered.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
            reordered_weights_.push_back(item.reordered.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
} // namespace strata::core

