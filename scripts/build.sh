#!/usr/bin/env bash
# Build the published tree with oneAPI and preserve a working executable on failure.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ONEAPI_SET_VARS="${ONEAPI_SET_VARS:-/opt/intel/oneapi/setvars.sh}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
GGML_DIR="${STRATA_GGML_DIR:-}"
JOBS="${JOBS:-4}"

usage() {
    cat <<'EOF'
Usage: scripts/build.sh --ggml-dir PATH [--build-dir PATH] [--jobs N]

Prerequisite: source-compatible Intel oneAPI setvars.sh, icpx, CMake 3.24+,
oneMKL, and a llama.cpp checkout containing ggml/src/ggml-sycl.
EOF
}

while (($#)); do
    case "$1" in
        --ggml-dir)
            [[ $# -ge 2 ]] || { echo "error: --ggml-dir needs a path" >&2; exit 2; }
            GGML_DIR=$2; shift 2 ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "error: --build-dir needs a path" >&2; exit 2; }
            BUILD_DIR=$2; shift 2 ;;
        --jobs)
            [[ $# -ge 2 ]] || { echo "error: --jobs needs a positive integer" >&2; exit 2; }
            JOBS=$2; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if [[ -z "$GGML_DIR" ]]; then
    echo "error: ggml_dir_present=0; pass --ggml-dir PATH" >&2
    exit 2
fi
if [[ ! -d "$GGML_DIR/ggml/src/ggml-sycl" ]]; then
    echo "error: ggml_sycl_present=0; missing $GGML_DIR/ggml/src/ggml-sycl" >&2
    exit 3
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: jobs_valid=0; expected a positive integer, got '$JOBS'" >&2
    exit 2
fi
if [[ ! -r "$ONEAPI_SET_VARS" ]]; then
    echo "error: setvars_present=0; cannot read $ONEAPI_SET_VARS" >&2
    exit 4
fi

mkdir -p "$BUILD_DIR" || {
    echo "error: build_dir_created=0; cannot create $BUILD_DIR" >&2
    exit 5
}
RAW_LOG="$BUILD_DIR/build.raw.log"
BINARY="$BUILD_DIR/strata-sycl"
PREVIOUS_BINARY=""
if [[ -f "$BINARY" ]]; then
    PREVIOUS_BINARY="$BUILD_DIR/.strata-sycl.previous.$$"
    if ! cp -p "$BINARY" "$PREVIOUS_BINARY"; then
        echo "error: previous_binary_backup=0; refusing to build" >&2
        exit 6
    fi
fi

restore_previous() {
    if [[ -n "$PREVIOUS_BINARY" && -f "$PREVIOUS_BINARY" ]]; then
        if cp -p "$PREVIOUS_BINARY" "$BINARY"; then
            echo "previous_binary_restored=1"
        else
            echo "previous_binary_restored=0" >&2
        fi
    else
        echo "previous_binary_restored=0"
    fi
}

fail_stage() {
    local stage=$1 rc=$2
    restore_previous
    echo "build_ok=0 stage=$stage rc=$rc raw_log=$RAW_LOG" >&2
    echo "failure_hint: inspect the numeric rc and build.raw.log" >&2
    exit "$rc"
}

: > "$RAW_LOG" || {
    echo "error: raw_log_created=0; cannot write $RAW_LOG" >&2
    restore_previous
    exit 7
}

# The llama.cpp source acquisition is intentionally documented, not guessed:
#   git clone <llama.cpp-repository-url> llama.cpp
#   cd llama.cpp
#   git checkout <hash>
#   cd ..
# The exact repository and commit hash are unconfirmed in this publication.

# setvars.sh is sourced before CMake so icpx, SYCL, and oneMKL are discoverable.
set +e
source "$ONEAPI_SET_VARS"
setvars_status=$?
set -e
if ((setvars_status != 0)); then
    fail_stage setvars "$setvars_status"
fi

# With the default checkout and build directory, the copy-paste commands are:
#   cmake -S . -B build -DSTRATA_GGML_DIR=<reader-llama.cpp>
#   cmake --build build --parallel 4
set +e
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DSTRATA_GGML_DIR="$GGML_DIR" 2>&1 | tee "$RAW_LOG"
cmake_statuses=("${PIPESTATUS[@]}")
set -e
cmake_rc=${cmake_statuses[0]}
tee_rc=${cmake_statuses[1]}
if ((cmake_rc != 0 || tee_rc != 0)); then
    ((cmake_rc != 0)) && fail_stage cmake-configure "$cmake_rc"
    fail_stage cmake-log "$tee_rc"
fi

set +e
cmake --build "$BUILD_DIR" --parallel "$JOBS" 2>&1 | tee -a "$RAW_LOG"
build_statuses=("${PIPESTATUS[@]}")
set -e
build_rc=${build_statuses[0]}
tee_rc=${build_statuses[1]}
if ((build_rc != 0 || tee_rc != 0)); then
    ((build_rc != 0)) && fail_stage cmake-build "$build_rc"
    fail_stage build-log "$tee_rc"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary_present=0; expected executable $BINARY" >&2
    fail_stage binary-missing 8
fi

set +e
ldd -r "$BINARY" 2>&1 | tee -a "$RAW_LOG"
ldd_statuses=("${PIPESTATUS[@]}")
set -e
ldd_rc=${ldd_statuses[0]}
tee_rc=${ldd_statuses[1]}
if ((tee_rc != 0)); then
    fail_stage ldd-log "$tee_rc"
fi
UNRESOLVED="$(awk '/undefined symbol:/{n++} END{print n+0}' "$RAW_LOG")"
printf 'ldd_rc=%d unresolved_symbols=%s\n' "$ldd_rc" "$UNRESOLVED"
if ((ldd_rc != 0)); then
    fail_stage ldd "$ldd_rc"
fi
if [[ "$UNRESOLVED" != 0 ]]; then
    fail_stage unresolved-symbols 9
fi

if [[ -n "$PREVIOUS_BINARY" && -f "$PREVIOUS_BINARY" ]]; then
    rm -f "$PREVIOUS_BINARY"
fi
printf 'build_ok=1 jobs=%s unresolved_symbols=0 binary=%s raw_log=%s\n' "$JOBS" "$BINARY" "$RAW_LOG"
