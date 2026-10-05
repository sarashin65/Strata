#pragma once

#include <cstdint>

namespace strata::kernels {

/// One captured device-side flag wait.  The host owns a mapped array of these records.
struct DeviceWaitStat {
    uint64_t start;
    uint64_t end;
    uint64_t spins;
    uint32_t layer;
    uint32_t group;
    uint32_t kind;
    uint32_t value;
};

}  // namespace strata::kernels
