#!/usr/bin/env bash
# Run the smallest useful public verification: GPU, cache announcement, and token round trip.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
TOOLS_DIR="${STRATA_TOOLS_DIR:-${ROOT_DIR}/tools}"
TOKENIZER_DIR="${STRATA_TOKENIZER_DIR:-${ROOT_DIR}/pack/tokenizer}"
ENGINE="${STRATA_ENGINE:-${ROOT_DIR}/build/strata-sycl}"
PACK_DIR="${STRATA_PACK_DIR:-${ROOT_DIR}/pack}"
SHARD1="${STRATA_SHARD1:-${ROOT_DIR}/model/shard1.gguf}"
SHARD2="${STRATA_SHARD2:-${ROOT_DIR}/model/shard2.gguf}"
DATA_DIR="${STRATA_DATA_DIR:-${ROOT_DIR}/data}"
PROFILE="${STRATA_PROFILE:-${DATA_DIR}/expert-profile.bin}"
MTP_DIR="${STRATA_MTP_DIR:-${ROOT_DIR}/mtp/rt}"
PROMPT_TEXT="${STRATA_PROMPT_TEXT:-${ROOT_DIR}/prompt.txt}"
PROMPT_IDS="${STRATA_PROMPT_IDS:-${ROOT_DIR}/prompt.ids}"
OUTPUT_IDS="${STRATA_OUTPUT_IDS:-${ROOT_DIR}/response.ids}"
OUTPUT_TEXT="${STRATA_OUTPUT_TEXT:-${ROOT_DIR}/response.txt}"
RUN_LOG="${STRATA_RUN_LOG:-${ROOT_DIR}/verify.raw.log}"

usage() {
    cat <<'EOF'
Usage:
  scripts/verify.sh
  scripts/verify.sh encode --tokenizer DIR --text-file PATH --ids-file PATH
  scripts/verify.sh decode --tokenizer DIR --ids-file PATH --text-file PATH

encode reads `--text-file` as input and writes `--ids-file` as output.
decode reads `--ids-file` as input and writes `--text-file` as output.

The default action calls sycl-ls, runs a short pre-tokenized prompt through
run.sh, checks the startup cache line, then decodes the T <id> output.
EOF
}

codec() {
    local mode=$1 tokenizer=$2 text_file=$3 ids_file=$4
    PYTHONPATH="${TOOLS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON" - "$mode" "$tokenizer" "$text_file" "$ids_file" <<'PY'
import json
import pathlib
import sys

from strata_tokenizer import Tokenizer

MAX_ID = 248320

def load_tokenizer(directory: pathlib.Path) -> Tokenizer:
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        index = int(index)
        if index < 0 or index >= len(tokens):
            raise ValueError(f"vocabulary index out of range: {index}")
        tokens[index] = token
    if any(token is None for token in tokens):
        raise ValueError("vocabulary has a missing index")
    merges = [line for line in (directory / "merges.txt").read_text(encoding="utf-8").splitlines() if line]
    token_types = json.loads((directory / "token_type.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    return Tokenizer(tokens, merges, token_types,
                     pre=config.get("pre", "qwen35"),
                     special_ids=config.get("special_ids", {}))

def main() -> int:
    mode, tokenizer_path, text_path, ids_path = sys.argv[1:]
    tokenizer = load_tokenizer(pathlib.Path(tokenizer_path))
    text = pathlib.Path(text_path)
    ids = pathlib.Path(ids_path)
    if mode == "encode":
        values = tokenizer.encode(text.read_text(encoding="utf-8"), parse_special=True)
        invalid = [value for value in values if not isinstance(value, int) or value < 0 or value >= MAX_ID]
        if invalid:
            print(f"encode: ids={len(values)} invalid={len(invalid)} max={max(values)}", file=sys.stderr)
            return 3
        ids.write_text(" ".join(str(value) for value in values) + "\n", encoding="utf-8")
        print(f"encode: ids={len(values)} invalid=0 max={max(values) if values else -1}")
        return 0 if len(values) >= 1 else 4
    if mode == "decode":
        fields = ids.read_text(encoding="utf-8").replace(",", " ").split()
        if not fields:
            print("decode: ids=0 chars=0", file=sys.stderr)
            return 4
        try:
            values = [int(field, 10) for field in fields]
        except ValueError as error:
            print(f"decode: ids={len(fields)} invalid=1 error={error}", file=sys.stderr)
            return 3
        invalid = [value for value in values if value < 0 or value >= MAX_ID]
        if invalid:
            print(f"decode: ids={len(values)} invalid={len(invalid)} max={max(values)}", file=sys.stderr)
            return 3
        result = tokenizer.decode(values)
        text.write_text(result, encoding="utf-8")
        print(f"decode: ids={len(values)} invalid=0 chars={len(result)}")
        return 0 if len(result) >= 1 else 5
    print(f"unknown codec mode: {mode}", file=sys.stderr)
    return 2

raise SystemExit(main())
PY
    local rc=$?
    return "$rc"
}

codec_mode="${1:-}"
if [[ "$codec_mode" == encode || "$codec_mode" == decode ]]; then
    shift
    TEXT_FILE="$PROMPT_TEXT"
    IDS_FILE="$PROMPT_IDS"
    while (($#)); do
        case "$1" in
            --tokenizer|--text-file|--ids-file)
                [[ $# -ge 2 ]] || { echo "error: $1 needs a path" >&2; exit 2; }
                case "$1" in
                    --tokenizer) TOKENIZER_DIR=$2 ;;
                    --text-file) TEXT_FILE=$2 ;;
                    --ids-file) IDS_FILE=$2 ;;
                esac
                shift 2 ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                echo "error: unknown codec argument: $1" >&2
                usage >&2
                exit 2 ;;
        esac
    done
    for item in vocab.json merges.txt token_type.json tokenizer.json chat_template.jinja; do
        [[ -f "$TOKENIZER_DIR/$item" ]] || { echo "error: tokenizer_missing=1 path=$TOKENIZER_DIR/$item" >&2; exit 3; }
    done
    if [[ "$codec_mode" == encode ]]; then
        [[ -f "$TEXT_FILE" ]] || { echo "error: text_file_present=0 path=$TEXT_FILE" >&2; exit 3; }
    else
        [[ -f "$IDS_FILE" ]] || { echo "error: ids_file_present=0 path=$IDS_FILE" >&2; exit 3; }
    fi
    codec "$codec_mode" "$TOKENIZER_DIR" "$TEXT_FILE" "$IDS_FILE"
    exit $?
fi

if [[ $# -gt 0 && "$1" != check && "$1" != --help && "$1" != -h ]]; then
    echo "error: unknown verify argument: $1" >&2
    usage >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    usage
    exit 0
fi

if [[ ! -f "$PROMPT_TEXT" ]]; then
    printf '%s\n' 'short English verification prompt' > "$PROMPT_TEXT" || {
        echo "error: prompt_created=0 path=$PROMPT_TEXT" >&2
        exit 3
    }
fi

set +e
GPU_LIST="$(sycl-ls 2>&1)"
SYCL_RC=$?
set +e
printf '%s\n' "$GPU_LIST"
GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | awk 'tolower($0) ~ /gpu/ {n++} END{print n+0}')"
printf 'sycl_rc=%d gpu_lines=%s\n' "$SYCL_RC" "$GPU_COUNT"
if ((SYCL_RC != 0 || GPU_COUNT < 1)); then
    echo "error: GPU visibility check failed; sycl_rc=$SYCL_RC gpu_lines=$GPU_COUNT" >&2
    exit 4
fi

"$0" encode --tokenizer "$TOKENIZER_DIR" --text-file "$PROMPT_TEXT" --ids-file "$PROMPT_IDS"
encode_rc=$?
((encode_rc == 0)) || { echo "error: encode_rc=$encode_rc" >&2; exit "$encode_rc"; }

"$SCRIPT_DIR/run.sh" \
    --engine "$ENGINE" --pack "$PACK_DIR" --native "$SHARD1" --ple-gguf "$SHARD2" \
    --mtp "$MTP_DIR" --expert-profile "$PROFILE" --tokens-file "$PROMPT_IDS" \
    --output-ids "$OUTPUT_IDS" --run-log "$RUN_LOG"
run_rc=$?
if ((run_rc != 0)); then
    echo "error: run_rc=$run_rc; inspect $RUN_LOG" >&2
    exit "$run_rc"
fi

CACHE_LINES="$(awk '/expert_slots=[0-9]+|expert cache|expert-cache/ {n++} END{print n+0}' "$RUN_LOG")"
OUTPUT_COUNT="$(awk 'NF {n++} END{print n+0}' "$OUTPUT_IDS")"
printf 'expert_cache_lines=%s output_ids=%s\n' "$CACHE_LINES" "$OUTPUT_COUNT"
if ((CACHE_LINES < 1)); then
    echo "error: expert_cache_announcement=0; expected an expert_slots or expert cache line" >&2
    exit 5
fi
if ((OUTPUT_COUNT < 1)); then
    echo "error: output_ids=0; engine produced no token IDs" >&2
    exit 6
fi

"$0" decode --tokenizer "$TOKENIZER_DIR" --ids-file "$OUTPUT_IDS" --text-file "$OUTPUT_TEXT"
decode_rc=$?
((decode_rc == 0)) || { echo "error: decode_rc=$decode_rc" >&2; exit "$decode_rc"; }
printf '%s\n' 'decoded_output_begin'
cat "$OUTPUT_TEXT"
printf '%s\n' 'decoded_output_end'
printf 'verify_ok=1 sycl_gpu_lines=%s expert_cache_lines=%s output_ids=%s\n' "$GPU_COUNT" "$CACHE_LINES" "$OUTPUT_COUNT"
