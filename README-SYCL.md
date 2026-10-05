# Strata SYCL public cookbook

This is a recipe to follow, not a verified installer, for the converted Strata
tree. It is not an upstream Strata release. The public tree is a SYCLomatic
CUDA-to-SYCL conversion followed by manual SYCL and oneAPI fixes. The commands
below are written for a reader who changes only the paths at the top of the
scripts.

This distribution has not been built, and none of its scripts have been run.
The numbers below are measurements from the development machine (Arc Pro B70);
build, link, and runtime behavior in your environment, on other cards, and with
other quantizations are unconfirmed.

## 1. Public tree and checkout

The base is the `main` branch of the `sarashin65/Strata` fork at `236d5f2`,
the upstream Engine 0.1.17 generation. The publication branch is
`sycl-port`. The converted source tree is the primary artifact; SYCLomatic is
not required for an ordinary checkout and build.

```sh
git clone -b sycl-port https://github.com/sarashin65/Strata.git
cd Strata
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

The fork base, branch existence, and the final clean-tree manifest remain
unconfirmed until a publication pass. Do not silently substitute the latest
upstream Engine generation.

## 2. Environment and prerequisites

The measured environment was Linux with `xe`, Level Zero/NEO, Intel oneAPI
2026.0, `icpx`, oneMKL, and `sycl-ls` reporting the tested B70 device. The
machine had 94 GB RAM, about 76 GB of disk, and 32 GB VRAM. Those numbers are
B70 measurements and prerequisites for the described run, not portability
claims.

Only Arc Pro B70 (BMG-G31) is confirmed. Every other Intel Arc model, driver,
oneAPI release, operating system, RAM layout, and VRAM layout is **unconfirmed**;
this document does not say that Arc in general works.

Run the environment setup before CMake or the executable:

```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

There is no SYCL installer. Upstream `setup.py` and `setup.sh` are NVIDIA
oriented and are not usable as SYCL installers. Install the compiler, runtime,
oneMKL, and Python prerequisites by the reader's normal system method.

No CUDA toolkit is required. The tree ships both ggml headers the port uses: the
SYCLomatic-converted `third_party/ggml/ggml-common.h` (it uses `sycl::half` and
`sycl::half2` and does not include `<cuda_fp16.h>`) for the SYCL translation
units, and `third_party/ggml/ggml-common-upstream.h`, the unconverted copy, for
the four conversion-host translation units. Substituting either copy for the
other breaks the build (`fatal error: 'cuda_fp16.h' file not found`, or
`type 'const ggml_half2' (aka 'const __half2') does not provide a subscript
operator`, or `use of undeclared identifier 'dpct'`). If you deliberately want
to supply a separate CUDA include directory, `scripts/build.sh --cuda-include
DIR` forwards it as `-DSTRATA_CUDA_INCLUDE_DIR=DIR`; the measured build needs
none.

## 3. Inputs and excluded artifacts

The measured configuration uses the following reader-supplied materials. The
availability notes are publication status, not download claims.

| Item | Measured contents | Availability |
|---|---|---|
| Model | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, two GGUF shards, approximately 75.8 GB total | Download source unconfirmed: `<MODEL-DOWNLOAD-URL>` |
| Native IQ3 pack | `dense.bin` (approximately 1.5 GB), `index.txt`, `native_experts.txt`, and `tokenizer/` with `vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json`, and `chat_template.jinja` | Generate from the two model shards with `prepare-data.sh`; the reader's tool version may produce a different layout |
| MTP head | `dense.bin`, `dense.txt`, `experts.bin` (approximately 708 MB), and `draft_vocab.bin` | BF16 checkpoint source unconfirmed: `<MTP-BF16-CHECKPOINT-URL>` |
| Existing upstream data | `data/expert-profile.bin` and `data/draft_vocab.bin` | Already present in the upstream repository; the reader does not create them. Identity with the measurement inputs is unconfirmed |

The model weights are not distributed here. Other quantizations and models are
**unmeasured and unconfirmed**. Do not reuse the published performance numbers
for them.

The upstream tree already contains the data files listed above; this recipe
does not generate them. The profile shipped by upstream has not been proven
identical to the profile used for the published measurement. GGUF files,
generated packs, generated MTP runtime files, build directories, executables,
objects, JIT caches, logs, credentials, and machine-specific service settings
are not publication artifacts.

The pack's `tokenizer/` directory is part of the generated pack because the
engine accepts token IDs, not text. `vocab.json` and `merges.txt` are the two
tokenizer inputs needed for token conversion; `token_type.json`, `tokenizer.json`,
and `chat_template.jinja` are the three additional files present in the
measured pack and checked by `verify.sh`. A missing artifact is reported as
`missing_pack_artifact=1 path=...`, which distinguishes an incomplete pack
from a different upstream-tool layout.

## 4. Optional SYCLomatic maintenance conversion

A normal reader starts from the converted `sycl-port` tree and skips
SYCLomatic. Re-run conversion only when maintaining a new CUDA base. Use a
clean source snapshot, a SYCLomatic `c2s` build, CUDA headers appropriate to
that snapshot, and the project conversion options; review every diagnostic
and every generated file before replacing the published tree.

The converted tree contains machine-generated `.dp.cpp` translation units and
manual SYCL/oneAPI fixes. The public build does not claim that a fresh
conversion is byte-identical. The exact SYCLomatic build, CUDA header set, and
regeneration manifest are **unconfirmed**.

## 5. Build

First obtain a local llama.cpp checkout. The exact repository and commit used
for the original measurement are **unconfirmed**, so the placeholder must be
replaced rather than guessed:

```sh
git clone <llama.cpp-repository-url> llama.cpp
cd llama.cpp
git checkout <hash>
test -d ggml/src/ggml-sycl
cd ..
source /opt/intel/oneapi/setvars.sh
```

The `ggml/src/ggml-sycl` directory is mandatory. The exact copy-and-checkout
sequence is also available in `scripts/build.sh`; it refuses a missing SYCL
backend with `ggml_sycl_present=0`.

From the Strata checkout, the copy-paste build entry point is:

```sh
scripts/build.sh --ggml-dir "$PWD/llama.cpp" --build-dir "$PWD/build" --jobs 4
```

The oneAPI compilers must be selected explicitly. The script passes
`-DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx` and
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` (the configuration the published numbers
came from), and stops with `compiler_not_icpx=1` if CMake selects anything
else. Without those flags CMake picks the system `c++` and the first
translation unit fails at once with
`c++: error: unrecognized command-line option '-fsycl'`.

The script is intended to source oneAPI, run CMake, build with four jobs by
default, and run `ldd -r`. Its intended success markers are `build_ok=1` and
`unresolved_symbols=0`; these statuses are unconfirmed until a reader builds
the package. Configure, build, missing-backend, missing-executable, `ldd`, and
unresolved-symbol failures have distinct numeric/status messages. The raw
transcript is `build.raw.log` inside the build directory. A previous
`strata-sycl` executable is backed up before the build and restored on a
failure; no build was run while preparing this publication.

The public CMake file has an empty optional `STRATA_CUDA_INCLUDE_DIR` by
default. Whether converted code genuinely needs CUDA headers is
**unconfirmed**. oneMKL lookup prefers `$MKLROOT` and otherwise uses the
conventional oneAPI location; the clean configure and link are **unconfirmed**.

## 6. Prepare pack and MTP

The reader supplies two GGUF shard paths and an output directory. The default
native IQ path uses upstream `tools/iq_pack.py`; its two-shard discovery is
made explicit by the wrapper and its result is expected to contain:

- `dense.bin` (approximately 1.5 GB);
- `index.txt`;
- `native_experts.txt`; and
- `tokenizer/{vocab.json, merges.txt, token_type.json, tokenizer.json, chat_template.jinja}`.

The alternative canonical Q2 path uses `strata_pack.py` and `pack_index.py`,
which are members of the upstream `pack_*.py` tool family.

```sh
scripts/prepare-data.sh pack \
  --shard1 "$PWD/model/shard1.gguf" \
  --shard2 "$PWD/model/shard2.gguf" \
  --out "$PWD/pack"
```

For a canonical Q2 pack only, select `--kind q2`; it is not the IQ3_XXS
measurement path. A pack failure prints `pack_missing=N`; success prints
`pack_ok=1`. The reader's pack layout remains unconfirmed until the upstream
tool version is checked.

The MTP head is fetched and repacked with the upstream sequence
`tools/mtp_fetch.py` then `mtp_pack.py`; `mtp_rt.py` emits the runtime files
consumed by the engine. The fetch acceptance count is exactly 31 tensors, and
the runtime acceptance set is `dense.txt`, `dense.bin`, `experts.bin`, and
`draft_vocab.bin`.

```sh
scripts/prepare-data.sh mtp \
  --out "$PWD/mtp"
```

The current upstream `mtp_fetch.py` has `inventory`/`fetch` commands and its
source location is configured in the tool; it does not expose a
`--checkpoint` switch, so this wrapper does not accept or use that argument.
The source setting and fetch result must be confirmed before a real fetch. The
wrapper copies the already-existing upstream `data/draft_vocab.bin` into the
MTP runtime folder; it does not create the upstream data file. It never
creates an expert profile.

## 7. Fixed launch

Change paths only in the first variables of `scripts/run.sh`, or use its path
options. The numeric arguments and feature switches are fixed to the v2
condition. `STRATA_PREFILL_MMQ=0` is mandatory.

The fixed engine invocation written by `run.sh` is:

```text
--serve
--pack <pack-dir> --native <shard1.gguf> --ple-gguf <shard2.gguf>
--max-new 32000 --max-context 131072 --spec 16 --spec-min-p 0.5 --prefill 8192
--expert-profile data/expert-profile.bin --expert-cache auto --mtp <mtp-dir>
--pcie-mode kernel --pcie-frac 0 --eos-ids 248046 --prompt-cache 6 --prompt-cache-every 16384
--turn-token 248045 --short-read 64 --mtp-max-t 4
```

Before this invocation, `run.sh` reads the `--tokens-file` path override and
writes the token IDs to standard input as one `GEN 32000 <comma-separated IDs>`
line followed by `QUIT`. It does not pass `--tokens-file` to the engine. This
launch shape and the engine's ID-stream and output protocol are script
expectations, not results confirmed by running this publication.

The environment block is fixed as follows:

```text
STRATA_PREFILL_MMQ=0
STRATA_WATCHDOG_S=180
ON:  WIDE_LOOKUP, DEC_BATCH, PCIE_SKIP, GDN_STAGE, GDN_LEAN_COMMIT, Q5K_QMINUS, MISS_GATHER, PLAN_COPY_WIDE, GR_NORM_ILP
OFF: IQ_PACK, DQ_VEC, MMVQ_ROT, GR_TRED, PF_HC_FUSE, QSA_SCORES_MULTI, QSA_TOPK_REG
```

`run.sh` is written to materialize the ON values as `1` and the OFF values as
`0`, keep the listed argument values unchanged, read `--tokens-file`, and
extract engine lines matching `T <id>` into `response.ids`. It checks the pack,
two GGUF shards, profile, MTP files, and input ID file first, then prints
`required_missing=N`. These checks and output are unconfirmed until a reader
runs the package. It does not silently substitute a short-generation
configuration.

## 8. Verify token IDs

The engine input and output contract is an integer ID stream assumed by these
scripts. A text prompt is intended to be encoded with the pack's `tokenizer/`,
passed to the engine, and decoded with the same files. `verify.sh` imports the
upstream tokenizer implementation while checking `vocab.json`, `merges.txt`,
`token_type.json`, `tokenizer.json`, and `chat_template.jinja` from the pack.
It rejects an empty ID list, non-integer IDs, and IDs at or above 248320.

The full minimum check is:

```sh
printf '%s\n' 'short English verification prompt' > "$PWD/prompt.txt"
scripts/verify.sh encode --tokenizer "$PWD/pack/tokenizer" --text-file "$PWD/prompt.txt" --ids-file "$PWD/prompt.ids"
scripts/run.sh --tokens-file "$PWD/prompt.ids" --output-ids "$PWD/response.ids" --pack "$PWD/pack"
scripts/verify.sh decode --tokenizer "$PWD/pack/tokenizer" --ids-file "$PWD/response.ids" --text-file "$PWD/response.txt"
```

The default `verify.sh` is designed to run `sycl-ls`, require at least one GPU
line, check the startup log for an `expert_slots`/expert-cache announcement,
print numeric counts for encoded and generated IDs, and print decoded text
between `decoded_output_begin` and `decoded_output_end`. These runtime checks
and their stage failures are unconfirmed until the pack and engine are
available and the command is run.

## 9. Troubleshooting and low-VRAM operation

Start diagnosis with the numeric stage message and its raw log. A nonzero
`sycl-ls` status or zero GPU lines is an environment/runtime problem, not
proof of a source defect. A missing `ggml-sycl` directory is a llama.cpp
checkout problem. A nonzero CMake/build/`ldd -r` status is a build problem;
`unresolved_symbols=0` is required before runtime investigation.

`c++: error: unrecognized command-line option '-fsycl'` on the first
translation unit is not a source defect: CMake selected the system compiler.
Reconfigure with `-DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx` after
sourcing oneAPI, or let `scripts/build.sh` do it and report
`compiler_not_icpx=1`.

`fatal error: 'cuda_fp16.h' file not found`,
`type 'const ggml_half2' (aka 'const __half2') does not provide a subscript
operator`, or `use of undeclared identifier 'dpct'` means one of the two
shipped `third_party/ggml/` headers was replaced by the other copy (section 2).
Restore both; no CUDA toolkit is needed for the converted tree.
`scripts/build.sh` only forwards `--cuda-include DIR` when a directory is given
or auto-detected.

At runtime, keep `--expert-cache auto`, the expert profile, kernel PCIe mode,
and `--pcie-frac 0` exactly as shown. The cache can lend prompt buffers; the
startup report names the resulting expert slots, borrowed prompt slots, and
free VRAM. Smaller-VRAM cards may need a card-specific profile and different
placement, but no such card has been measured here. Do not infer a minimum
VRAM value or a speed from the B70 run.

`STRATA_WATCHDOG_S=180` is a watchdog prerequisite, not a performance claim.
If a long prompt stops or becomes extremely slow, keep the diagnosis at the
same unresolved transport/stall class until a log identifies the cause; do
not present a cause as established by this cookbook.

## 10. Measurements, limits, and unconfirmed items

The only published performance and placement values are from one B70 (32 GB)
run with IQ3_XXS, an 84,834-token input, 34,186 generated tokens, an
approximately 119,000-token sequence, decode context 85K--119K,
`--max-context 131072`, window 16, output limit 32,000, every argument and
ON/OFF setting in section 7, and one execution:

| Published value | Condition |
|---|---|
| **80.8 tok/s** | Long real task, one execution; the adopted decode figure, not a short-prompt representative. |
| **12.37 ms/token** | The reciprocal of 80.8 tok/s from that same execution, not a separate measurement. |
| **843.6 tok/s prefill** | Same B70, IQ3_XXS, 84,834-token input and real task; not a claim about filling a 128K input. A colder-state measurement is unconfirmed. |
| **14,118 slots (22.93 GiB)** | Startup expert cache; the prompt path borrowed **2,121 slots (3.44 GiB)** and reported **490 MiB** free VRAM. This is placement, not performance. |

The output difference is **a middle 0.03% in a window-8 versus window-16
comparison**. Variance under the same settings is unmeasured. This does not
promise bit-identical output.

Everything below is **unconfirmed**: other Intel Arc models; other
quantizations or models; upstream-latest compatibility; exact llama.cpp
source and commit; clean CMake/link success; the necessity of CUDA headers;
AOT/JIT choices beyond the tested B70 target; identity of the upstream expert
profile with the measurement profile; fetch/checksum/provider terms; and the
legal details of distributing generated conversion artifacts. No installer,
portability guarantee, or performance extrapolation is implied.
