#!/usr/bin/env bash
# Run one fixed Strata configuration and extract generated token IDs from stdout.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DATA_DIR="${STRATA_DATA_DIR:-${ROOT_DIR}/data}"
ENGINE="${STRATA_ENGINE:-${ROOT_DIR}/build/strata-sycl}"
PACK_DIR="${STRATA_PACK_DIR:-${ROOT_DIR}/pack}"
SHARD1="${STRATA_SHARD1:-${ROOT_DIR}/model/shard1.gguf}"
SHARD2="${STRATA_SHARD2:-${ROOT_DIR}/model/shard2.gguf}"
PROFILE="${STRATA_PROFILE:-${DATA_DIR}/expert-profile.bin}"
MTP_DIR="${STRATA_MTP_DIR:-${ROOT_DIR}/mtp/rt}"
TOKENS_FILE="${STRATA_TOKENS_FILE:-${ROOT_DIR}/prompt.ids}"
OUTPUT_IDS="${STRATA_OUTPUT_IDS:-${ROOT_DIR}/response.ids}"
RUN_LOG="${STRATA_RUN_LOG:-${ROOT_DIR}/run.raw.log}"

usage() {
    cat <<'EOF'
Usage: scripts/run.sh [path overrides]
  --engine PATH --pack PATH --native PATH --ple-gguf PATH --mtp PATH
  --expert-profile PATH --tokens-file PATH --output-ids PATH --run-log PATH

All numeric engine flags and feature switches are fixed below. Only paths may
be changed by the caller. The input file contains whitespace- or comma-separated
integer token IDs; output IDs are extracted from lines printed as: T <id>.
EOF
}

while (($#)); do
    case "$1" in
        --engine|--pack|--native|--ple-gguf|--mtp|--expert-profile|--tokens-file|--output-ids|--run-log)
            [[ $# -ge 2 ]] || { echo "error: $1 needs a path" >&2; exit 2; }
            case "$1" in
                --engine) ENGINE=$2 ;;
                --pack) PACK_DIR=$2 ;;
                --native) SHARD1=$2 ;;
                --ple-gguf) SHARD2=$2 ;;
                --mtp) MTP_DIR=$2 ;;
                --expert-profile) PROFILE=$2 ;;
                --tokens-file) TOKENS_FILE=$2 ;;
                --output-ids) OUTPUT_IDS=$2 ;;
                --run-log) RUN_LOG=$2 ;;
            esac
            shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

missing=0
check_file() {
    local label=$1 path=$2
    if [[ ! -f "$path" ]]; then
        printf 'missing_%s=1 path=%s\n' "$label" "$path" >&2
        missing=$((missing + 1))
    fi
}
check_dir() {
    local label=$1 path=$2
    if [[ ! -d "$path" ]]; then
        printf 'missing_%s=1 path=%s\n' "$label" "$path" >&2
        missing=$((missing + 1))
    fi
}

if [[ ! -x "$ENGINE" ]]; then
    printf 'missing_engine=1 path=%s\n' "$ENGINE" >&2
    missing=$((missing + 1))
fi
check_file pack_dense "$PACK_DIR/dense.bin"
check_file pack_index "$PACK_DIR/index.txt"
check_file pack_native_experts "$PACK_DIR/native_experts.txt"
check_dir pack_tokenizer "$PACK_DIR/tokenizer"
check_file shard1 "$SHARD1"
check_file shard2 "$SHARD2"
check_file mtp_dense_index "$MTP_DIR/dense.txt"
check_file mtp_dense "$MTP_DIR/dense.bin"
check_file mtp_experts "$MTP_DIR/experts.bin"
check_file expert_profile "$PROFILE"
check_file prompt_ids "$TOKENS_FILE"
printf 'required_missing=%d\n' "$missing"
if ((missing != 0)); then
    echo "error: required input count is nonzero; no engine launch" >&2
    exit 4
fi
# The resident protocol needs a comma-separated GEN request. Keep the token file
# as the user-facing input and reject anything that is not an integer token ID.
TOKEN_LIST="$(
awk '
{
    gsub(/,/, " ")
    for (i = 1; i <= NF; ++i) {
        if ($i !~ /^-?[0-9]+$/) {
            bad = 1
            continue
        }
        if (n > 0) printf ","
        printf "%s", $i
        ++n
    }
}
END {
    printf "\n"
    if (bad || n < 1) exit 2
}
' "$TOKENS_FILE"
)"
token_parse_rc=$?
token_count="$(awk '{gsub(/,/, " "); n += NF} END{print n+0}' "$TOKENS_FILE")"
printf 'token_parse_rc=%d token_ids=%s\n' "$token_parse_rc" "$token_count"
if ((token_parse_rc != 0 || token_count < 1)); then
    echo "error: token_ids_parse_failed=$token_parse_rc; expected whitespace/comma-separated integers" >&2
    exit 4
fi

mkdir -p "$(dirname -- "$OUTPUT_IDS")" "$(dirname -- "$RUN_LOG")" || {
    echo "error: output_parent_created=0" >&2
    exit 5
}
: > "$OUTPUT_IDS" || {
    echo "error: output_ids_created=0 path=$OUTPUT_IDS" >&2
    exit 5
}

# These are the exact fixed switches from the publication specification.
export STRATA_PREFILL_MMQ=0
export STRATA_WATCHDOG_S=180
export STRATA_WIDE_LOOKUP=1
export STRATA_DEC_BATCH=1
export STRATA_PCIE_SKIP=1
export STRATA_GDN_STAGE=1
export STRATA_GDN_LEAN_COMMIT=1
export STRATA_Q5K_QMINUS=1
export STRATA_MISS_GATHER=1
export STRATA_PLAN_COPY_WIDE=1
export STRATA_GR_NORM_ILP=1
export STRATA_IQ_PACK=0
export STRATA_DQ_VEC=0
export STRATA_MMVQ_ROT=0
export STRATA_GR_TRED=0
export STRATA_PF_HC_FUSE=0
export STRATA_QSA_SCORES_MULTI=0
export STRATA_QSA_TOPK_REG=0

set +e
{
    printf 'GEN 32000 %s\n' "$TOKEN_LIST"
    printf 'QUIT\n'
} | "$ENGINE" --serve \
    --pack "$PACK_DIR" --native "$SHARD1" --ple-gguf "$SHARD2" \
    --max-new 32000 --max-context 131072 --spec 16 --spec-min-p 0.5 --prefill 8192 \
    --expert-profile "$PROFILE" --expert-cache auto --mtp "$MTP_DIR" \
    --pcie-mode kernel --pcie-frac 0 --eos-ids 248046 --prompt-cache 6 --prompt-cache-every 16384 \
    --turn-token 248045 --short-read 64 --mtp-max-t 4 \
    2>&1 | tee "$RUN_LOG"
run_statuses=("${PIPESTATUS[@]}")
set -e
engine_rc=${run_statuses[1]}
tee_rc=${run_statuses[2]}

# The engine emits generated IDs as `T <id>`. Do not treat arbitrary log numbers
# as output; an empty extraction is a distinct failure from a nonzero engine rc.
awk '$1 == "T" && $2 ~ /^-?[0-9]+$/ { print $2 }' "$RUN_LOG" > "$OUTPUT_IDS"
output_count="$(awk 'NF {n++} END{print n+0}' "$OUTPUT_IDS")"
printf 'engine_rc=%d tee_rc=%d output_ids=%s run_log=%s\n' "$engine_rc" "$tee_rc" "$output_count" "$RUN_LOG"
if ((engine_rc != 0)); then
    echo "error: engine_return_nonzero=$engine_rc; inspect $RUN_LOG" >&2
    exit "$engine_rc"
fi
if ((tee_rc != 0)); then
    echo "error: run_log_write_failed=$tee_rc" >&2
    exit 6
fi
if ((output_count < 1)); then
    echo "error: output_ids=0; no T <id> lines were emitted" >&2
    exit 7
fi
printf 'run_ok=1 output_ids=%s output_file=%s\n' "$output_count" "$OUTPUT_IDS"
