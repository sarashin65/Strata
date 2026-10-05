#!/usr/bin/env bash
# Prepare a native model pack and the MTP runtime directory with upstream tools.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
TOOLS_DIR="${STRATA_TOOLS_DIR:-${ROOT_DIR}/tools}"
DATA_DIR="${STRATA_DATA_DIR:-${ROOT_DIR}/data}"

usage() {
    cat <<'EOF'
Usage:
  scripts/prepare-data.sh pack --shard1 PATH --shard2 PATH --out PATH [--kind native|q2]
  scripts/prepare-data.sh mtp --out PATH

The native IQ path uses tools/iq_pack.py. The canonical Q2 path uses the
upstream strata_pack.py plus pack_index.py (the pack_*.py family) and extracts
the tokenizer. The MTP path is mtp_fetch.py -> mtp_pack.py -> mtp_rt.py.
The upstream mtp_fetch.py does not accept --checkpoint; its source is
configured in the tool, so this wrapper does not accept or use that option.
The profile and draft vocabulary are read from the upstream data directory;
this script never generates either source file.
EOF
}

fail() {
    echo "error: $*" >&2
    exit 2
}
run_checked() {
    "$@"
    local rc=$?
    if ((rc != 0)); then
        echo "tool_rc=$rc failed=$1" >&2
        exit "$rc"
    fi
}
need_tool() {
    local name=$1
    if [[ ! -f "$TOOLS_DIR/$name" ]]; then
        echo "error: tool_present=0 name=$TOOLS_DIR/$name" >&2
        exit 3
    fi
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
MODE=$1
shift
if [[ "$MODE" == -h || "$MODE" == --help ]]; then
    usage
    exit 0
fi

if [[ "$MODE" == pack ]]; then
    SHARD1=""
    SHARD2=""
    OUT=""
    KIND=native
    BASE_PACK=""
    while (($#)); do
        case "$1" in
            --shard1|--shard2|--out|--kind|--base)
                [[ $# -ge 2 ]] || fail "$1 needs a value"
                case "$1" in
                    --shard1) SHARD1=$2 ;;
                    --shard2) SHARD2=$2 ;;
                    --out) OUT=$2 ;;
                    --kind) KIND=$2 ;;
                    --base) BASE_PACK=$2 ;;
                esac
                shift 2 ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                fail "unknown pack argument: $1" ;;
        esac
    done
    [[ -f "$SHARD1" ]] || fail "shard1_present=0 path=$SHARD1"
    [[ -f "$SHARD2" ]] || fail "shard2_present=0 path=$SHARD2"
    [[ -n "$OUT" ]] || fail "out_present=0"
    [[ "$KIND" == native || "$KIND" == q2 ]] || fail "kind_valid=0 expected native or q2"

    # iq_pack.py discovers a two-shard model from the standard suffix. For
    # reader-friendly arbitrary filenames, temporary symlinks provide that
    # suffix and the generated second-shard name is rewritten to the real name.
    SHARD1="$(CDPATH= cd -- "$(dirname -- "$SHARD1")" && pwd -P)/$(basename -- "$SHARD1")"
    SHARD2="$(CDPATH= cd -- "$(dirname -- "$SHARD2")" && pwd -P)/$(basename -- "$SHARD2")"
    CANON_DIR="$(mktemp -d)" || fail "temporary_dir_created=0"
    trap 'rm -rf "$CANON_DIR"' EXIT
    CANON1="$CANON_DIR/model-00001-of-00002.gguf"
    CANON2="$CANON_DIR/model-00002-of-00002.gguf"
    ln -s "$SHARD1" "$CANON1" || fail "shard1_linked=0"
    ln -s "$SHARD2" "$CANON2" || fail "shard2_linked=0"
    mkdir -p "$OUT" || fail "pack_dir_created=0 path=$OUT"

    if [[ "$KIND" == native ]]; then
        need_tool iq_pack.py
        IQ_ARGS=("$PYTHON" "$TOOLS_DIR/iq_pack.py" --gguf "$CANON1" --out "$OUT")
        if [[ -n "$BASE_PACK" ]]; then
            [[ -d "$BASE_PACK" ]] || fail "base_pack_present=0 path=$BASE_PACK"
            IQ_ARGS+=(--base "$BASE_PACK")
        fi
        run_checked "${IQ_ARGS[@]}"
        if [[ -f "$OUT/native_experts.txt" ]]; then
            "$PYTHON" - "$OUT/native_experts.txt" "$(basename -- "$CANON2")" "$(basename -- "$SHARD2")" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old, new = sys.argv[2], sys.argv[3]
text = path.read_text(encoding="utf-8")
path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
PY
            rewrite_rc=$?
            ((rewrite_rc == 0)) || { echo "error: shard_name_rewrite_rc=$rewrite_rc" >&2; exit "$rewrite_rc"; }
        fi
        REQUIRED=("dense.bin" "index.txt" "native_experts.txt" "tokenizer/vocab.json" "tokenizer/merges.txt" "tokenizer/token_type.json" "tokenizer/tokenizer.json" "tokenizer/chat_template.jinja")
    else
        need_tool strata_pack.py
        need_tool pack_index.py
        need_tool strata_tokenizer.py
        run_checked "$PYTHON" "$TOOLS_DIR/strata_pack.py" build --gguf "$CANON1" --out "$OUT" --skip-hash
        run_checked "$PYTHON" "$TOOLS_DIR/pack_index.py" --pack "$OUT"
        run_checked "$PYTHON" "$TOOLS_DIR/strata_tokenizer.py" --gguf "$CANON1" --out "$OUT"
        REQUIRED=("dense.bin" "index.txt" "experts.bin" "tokenizer/vocab.json" "tokenizer/merges.txt" "tokenizer/token_type.json" "tokenizer/tokenizer.json" "tokenizer/chat_template.jinja")
    fi
    missing=0
    for item in "${REQUIRED[@]}"; do
        if [[ ! -e "$OUT/$item" ]]; then
            printf 'missing_pack_artifact=1 path=%s\n' "$OUT/$item" >&2
            missing=$((missing + 1))
        fi
    done
    printf 'pack_missing=%d kind=%s\n' "$missing" "$KIND"
    ((missing == 0)) || exit 6
    printf 'pack_ok=1 out=%s\n' "$OUT"
    exit 0
fi

if [[ "$MODE" == mtp ]]; then
    OUT=""
    while (($#)); do
        case "$1" in
            --out)
                [[ $# -ge 2 ]] || fail "$1 needs a value"
                OUT=$2
                shift 2 ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                fail "unknown mtp argument: $1" ;;
        esac
    done
    [[ -n "$OUT" ]] || fail "out_present=0"
    need_tool mtp_fetch.py
    need_tool mtp_pack.py
    need_tool mtp_rt.py
    mkdir -p "$OUT" || fail "mtp_dir_created=0 path=$OUT"
    FETCH_DIR="$OUT/fetch"

    # Current upstream mtp_fetch.py exposes inventory/fetch and keeps its
    # source in the tool configuration; it has no --checkpoint option.
    # Confirm the source setting before a real fetch.
    run_checked "$PYTHON" "$TOOLS_DIR/mtp_fetch.py" fetch --out "$FETCH_DIR"
    [[ -f "$FETCH_DIR/mtp-manifest.json" ]] || fail "mtp_manifest_present=0"
    FETCH_COUNT="$($PYTHON - "$FETCH_DIR/mtp-manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(len(json.load(f)))
PY
)"
    FETCH_RC=$?
    ((FETCH_RC == 0)) || { echo "error: mtp_manifest_count_rc=$FETCH_RC" >&2; exit "$FETCH_RC"; }
    printf 'mtp_fetched_tensors=%s expected=31\n' "$FETCH_COUNT"
    [[ "$FETCH_COUNT" == 31 ]] || fail "mtp_tensor_count=$FETCH_COUNT expected=31"

    MTP_GGUF="$OUT/mtp-q2_0.gguf"
    RT_DIR="$OUT/rt"
    run_checked "$PYTHON" "$TOOLS_DIR/mtp_pack.py" --src "$FETCH_DIR" --experts q2_0 --out "$MTP_GGUF"
    run_checked "$PYTHON" "$TOOLS_DIR/mtp_rt.py" --gguf "$MTP_GGUF" --out "$RT_DIR"

    DRAFT_VOCAB="${DRAFT_VOCAB:-$DATA_DIR/draft_vocab.bin}"
    if [[ ! -f "$DRAFT_VOCAB" ]]; then
        echo "error: draft_vocab_present=0 path=$DRAFT_VOCAB; source it from the upstream data directory" >&2
        exit 7
    fi
    # This copies the already-existing upstream vocabulary into the runtime
    # directory; it does not generate or replace data/draft_vocab.bin.
    cp -f "$DRAFT_VOCAB" "$RT_DIR/draft_vocab.bin" || fail "draft_vocab_copy=0"
    missing=0
    for item in dense.txt dense.bin experts.bin draft_vocab.bin; do
        if [[ ! -f "$RT_DIR/$item" ]]; then
            printf 'missing_mtp_artifact=1 path=%s\n' "$RT_DIR/$item" >&2
            missing=$((missing + 1))
        fi
    done
    printf 'mtp_missing=%d\n' "$missing"
    ((missing == 0)) || exit 8
    printf 'mtp_ok=1 out=%s\n' "$RT_DIR"
    exit 0
fi

echo "error: unknown mode: $MODE" >&2
usage >&2
exit 2
