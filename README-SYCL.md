<h1 align="center">Strata on Intel Arc — SYCL port</h1>

<p align="center"><b>An experimental port of <a href="https://github.com/Niko1221/Strata">Strata</a> that runs the engine on
Intel Arc GPUs through SYCL / oneAPI instead of CUDA.</b><br>
Tested on one Arc Pro B70 (32 GB) · Linux only · not upstream, no installer yet</p>

**This is not the original Strata.** It is a fork for people who have an Intel Arc card and want to see whether the
engine can run there. For NVIDIA cards use [Niko1221/Strata](https://github.com/Niko1221/Strata) — it is faster, it is
one click to install, and it is the version that is maintained.

> **This page is a write-up, not a release.** The converted source tree is not published here yet — the measurements
> and commands below come from the machine the port was developed on. Read the build section as a recipe to follow,
> not as something you can download and run. Nothing here changes upstream's install instructions.

- **What is the same:** the engine's behaviour — model loading, the expert cache in VRAM, the PLE n-gram table on the
  SSD, MTP speculative decoding, prompt chunking. The port changes *how the CUDA code talks to the GPU*, not what it
  computes.
- **What is different:** every `.cu` file is a SYCLomatic conversion of the CUDA original, plus a list of hand fixes
  (below). Two kernels that the engine borrows from llama.cpp were replaced with llama.cpp's own SYCL
  implementations, and the BLAS path goes to oneMKL.
- **What it is not:** finished. Read [Known limits](#known-limits) before you spend an evening on it.

---

## Does it work? — measured numbers

One machine, one model size (IQ3_XXS), a real 8,989-token coding prompt:

| | this port | 
| --- | ---: |
| First token (8,989-token prompt) | **13.9 s** |
| Reading the prompt (prefill) | **656 tokens/s** (median of 5 runs, 5/5 completed) |
| Writing the answer (decode) | **50.9 tokens/s** |
| Decode with a 128K context window | 41.4 tokens/s |
| Output length | 3,000 tokens written to completion, stops on `<|im_end|>` |

Test machine: Arc Pro B70 (32 GB, BMG-G31), Ryzen 9 5900X, 94 GB DDR4-2667, SATA SSD, Linux.
For context, upstream on an RTX 5070 with DDR5-5200 reports 62 tokens/s (decode) and 1,110 tokens/s (prefill) for the
same quantization — **different machine and different memory, so do not compare the two columns**. Memory bandwidth
decides this engine's speed.

It writes usable code, not just tokens: asked to fix a real bug in a real repository, it produced a 4-file patch
(a helper function, its call sites, and tests). Hand review found it matched the intended design, with one cosmetic
difference in a test fixture.

## Known limits

- **There is no installer.** `setup.py` / `setup.sh` are NVIDIA-only and will stop at "no NVIDIA GPU found". Building
  this port is manual today — see [Install](#install).
- **Only one GPU has been tried:** Arc Pro B70 (BMG-G31). The kernels are compiled for `bmg_g31`; other Arc cards are
  untested (they might need a different AOT target, or might just work).
- **Linux only.** No Windows build has been attempted.
- **The fast prefill path is behind an environment variable** (`STRATA_PREFILL_MMQ=0`, FP16 oneMKL GEMM). Without it,
  prefill is 115 tokens/s instead of 656. It should become the default.
- **A 120K-token prompt has not been fed through.** A 131,072-token context *window* with an 8,989-token prompt is
  verified; the input side of a 120K prompt is not.
- **`--eos-ids` works but is not listed in `--help`.** Without it, generation runs to `--max-new` after the model has
  already emitted `<|im_end|>`.
- **No test suite, no CI.** Numeric checks were done against the CUDA build and against exhaustive CPU-side
  recomputation for the one place where a rounding change was made (see below).
- **Output can differ run to run** even with identical flags, when the adaptive expert cache is on: experts computed
  on the CPU and on the GPU round slightly differently. Bit-exact reproducibility is not a property of this engine.

## Requirements

- **GPU:** Intel Arc (tested: Arc Pro B70, 32 GB). Needs a working Level Zero stack:
  `sycl-ls` should list `[level_zero:gpu] … Arc(TM) Pro B70`.
- **Driver:** kernel driver `xe` + Intel compute runtime (NEO). Tested with NEO 26.31 / Level Zero 1.17.
- **Intel oneAPI 2026.0** — `icpx` and oneMKL. `source /opt/intel/oneapi/setvars.sh` before *anything*, including
  `--help`; without it the engine reports "No device of requested type available" and exits 134. This is the most
  common false alarm — it means the environment, not the GPU.
- **RAM:** the model's first shard plus ~10 GB (this port loaded 40 GiB of weights; the box had 94 GB).
- **Disk:** ~76 GB for the IQ3_XXS model (47 GB shard 1 + 29 GB shard 2) and 32 GB of VRAM.
- **Model files:** a Qwen3.8-Flash-Next **GSQ-RCO** GGUF (2 shards). The engine reads 8 expert quantization types
  (`Q2_0`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`) — check the expert tensors, not the
  whole-file histogram, before downloading.

## Install

Nothing here is packaged. The honest version of "install" is three steps, and step 1 assumes you already have the
upstream sources.

```
1. convert   upstream CUDA sources  ->  SYCL sources          (SYCLomatic / c2s)
2. build     SYCL sources + llama.cpp SYCL backend + oneMKL    (cmake + icpx)
3. run       the binary with a runtime pack, an expert profile and an MTP draft head
```

A `setup-intel.py` (or a flag on the existing `setup.py`) that does all three would be the right thing to have. It does
not exist yet.

### Step 1 — convert the CUDA sources with SYCLomatic

Get a SYCLomatic daily build ([oneapi-src/SYCLomatic releases](https://github.com/oneapi-src/SYCLomatic/releases)) and
run `c2s` on a clean `git archive` copy of the upstream tree. Two passes, because the language mode must match the file
type:

```sh
# .cu files — CUDA mode (do NOT pass -xc++ here)
c2s --in-root=<clean> --out-root=<out> \
    --cuda-include-path=<cuda>/include \
    --use-experimental-features=graph \
    --use-dpcpp-extensions=intel_device_math \
    --extra-arg=-I<stub-headers>            `# see note below` \
    $(<list of .cu files>)

# .cpp / .hpp files — C++ mode
c2s ... --extra-arg=-xc++ $(<list of .cpp/.hpp files>)
```

Three things worth knowing, each of which cost real time here:

- **`--extra-arg=-xc++` on a `.cu` file silently skips kernel conversion.** The output looks converted but still
  contains `<<<...>>>` and `blockIdx`; nothing compiles. This is a flag mistake, not a limitation of the tool.
- **Count `fatal error` in the conversion log before trusting the output.** dpct (≈clang 21) understands CUDA 12.9.
  Against CUDA 13 headers a missing header aborts analysis *silently* and the file comes out largely unconverted. The
  fix that worked here was an empty `#pragma once` file with the missing header's name plus `--extra-arg=-I<dir>`;
  it beat downgrading the CUDA headers (6 leftover CUDA lines vs 17).
- **`--use-experimental-features=graph` matters.** Without it, all 158 `cudaGraph*` diagnostics are left unhandled.
  With it (and the message the diagnostic itself points at), graph usage converts cleanly.

### Step 2 — build

The port's `CMakeLists.txt` differs from upstream in three places:

- `-fsycl -O1 -fp-model=precise -ffp-contract=off -Wno-overriding-option` for host and device. icpx's default is
  speed-first float (`-ffp-contract=fast`, reassociation); the engine has comments stating that it intends bit-exact
  results, so precise float is the sane default here. Note that `-ffp-model=precise` alone still leaves fused
  multiplies in, hence `-ffp-contract=off`.
- `STRATA_GGML_DIR` must point at a llama.cpp checkout (the same pinned commit upstream uses) **that has
  `ggml/src/ggml-sycl`** — the SYCL backend is where the port's quantized matrix kernels come from.
- oneMKL is linked explicitly (this is what turns the link error `oneapi::mkl::blas::column_major::gemm` into a working
  binary):

```cmake
set(MKL_LINK dynamic CACHE STRING "" FORCE)
set(MKL_SYCL_LINK dynamic CACHE STRING "" FORCE)
set(MKL_INTERFACE ilp64 CACHE STRING "" FORCE)
set(MKL_SYCL_THREADING sequential CACHE STRING "" FORCE)
find_package(MKL CONFIG REQUIRED PATHS "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl" NO_DEFAULT_PATH)
target_link_libraries(<target> PRIVATE ... MKL::MKL_SYCL)
```

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S <port-src> -B <build-dir> -DSTRATA_GGML_DIR=<llama.cpp>
cmake --build <build-dir> --parallel 4        # 26 s incremental, ~20 min from scratch on 12 cores
```

Check the link, not just the exit code: `rc=0` plus zero unresolved symbols from `ldd -r`. **Run builds one at a time
and keep a copy of the previous binary** — `ld` deletes the output when a link fails, so a failed experiment can leave
you with no executable at all. The first run after every build is a JIT warm-up and is much slower; discard it.

### Step 3 — runtime files

The engine is the only part that is CUDA-specific; the Python tools are shared with upstream and are used unchanged:

- **pack directory** (`dense.bin`, `index.txt`, `native_experts.txt`, `tokenizer/`) — produced by the project's tools
  from the GGUF. It carries the tokenizer, which matters, because the engine itself only accepts token IDs.
- **expert profile** (`data/expert-profile.bin`) — which experts to keep resident; the project's `make_profile` tool.
- **MTP draft head** — not in the distributed GGUF. Fetch the 31 `mtp.*` tensors from the BF16 checkpoint
  (`tools/mtp_fetch.py`, ~5 GB) and repack them (`tools/mtp_pack.py`). Engine flags expect `<dir>/{dense.txt,dense.bin,
  experts.bin}`.
- **`draft_vocab.bin`** — put this in the same directory. The draft head then covers 40,525 rows instead of all
  248,320 vocabulary entries. No code change, ~1.5 ms per round.

## Run

```sh
source /opt/intel/oneapi/setvars.sh
STRATA_PREFILL_MMQ=0 <build>/strata-sycl \
  --pack        <pack-dir> \
  --native      <shard1.gguf> \
  --ple-gguf    <shard2.gguf> \
  --mtp         <mtp-dir> \
  --expert-profile <profiles>/expert-profile.bin \
  --expert-cache auto \
  --tokens-file <prompt token IDs> \
  --prefill 8192 --max-context 131072 --spec 4 --max-new 3000 \
  --eos-ids 248046 \
  --pcie-mode kernel --pcie-frac 0 --stats
```

- `--tokens-file` — the engine has no tokenizer. Encode your prompt with the pack's `tokenizer/` (a pure-Python
  encoder built from `vocab.json` + `merges.txt` + the chat template is enough) and pass IDs. Decode the output IDs
  before judging the output; "it produced tokens" is not the same as "it produced an answer".
- `--prefill 8192` — large chunks are much faster (8192 → 656 tokens/s, 2048 → 429, 512 → 116), because the weights are
  dequantized once per chunk instead of once per step.
- `--spec 4` — speculative window. 4 beats 2 on long generations; the effect of the window depends on the output
  length, so measure it on your own workload.
- `--pcie-frac 0` — keep this. Pinned host memory is not available here, so `--pcie-mode direct` reads unregistered
  memory and the run dies.

## What the port had to change (the interesting part)

The conversion is mechanical; this list is the judgement that was left, roughly in order of how much it mattered. If
you are porting a similar engine, these are the traps:

- **The GPU↔CPU flag rendezvous.** The engine has a graphics kernel spin on a host-memory flag while the CPU computes
  experts, so the GPU has to see a host write *without* waiting for a cache line to be evicted. Ordinary reads fail
  about half the time on this device; bypassing both L1 and L3 (`annotated_ptr` +
  `read_hint<cache_control<uncached, L1, L3>>`) failed zero times in a full sweep. Bypassing only L1 fails always and
  only L3 fails half the time. This was measured before the port was designed, and it decided the design.
- **Graphs are worth keeping.** Direct submission had a higher failure rate than the captured graph, and the graph
  conversion held up. Do not replace `cudaGraph*` with sequential launches for convenience.
- **`dpct::sync_barrier` is not `cudaEventRecord`.** When the queue is the default one it waits for *every* queue on
  the device — a host-wide serialization that the original code never had (and a deadlock shape). The engine's
  per-layer records were turned back into non-blocking device-side barriers to stay faithful.
- **`DPCT_CHECK_ERROR(expr)` throws away `expr`'s value.** `q = DPCT_CHECK_ERROR(cs->ext_oneapi_empty())` is always
  `0`, so a "has the GPU finished?" test always says yes. The first layer then decided the flag would never arrive and
  gave up. The comparison's direction was irrelevant — the return value was the bug.
- **`cudaHostRegister` converted to a no-op that returns success.** Logs said `cudaHostRegister PORTABLE ok` while
  dozens of GiB of expert memory were in fact not visible to the GPU. Anything that depended on that registration
  (`--pcie-mode direct`, `--pcie-frac > 0`) then hung or died with SIGBUS. Default to the safe path and never treat a
  no-op as success.
- **Staging ring depth.** The host↔GPU staging ring was too shallow for one layer's worth of jobs; deepening it to
  512 removed the hangs completely (cost: ~1.2 GB of pinned memory). The engine's own stall reports named the exact
  stage and job.
- **The "all layers in one order" prefill path.** For chunks ≥ 2048 the engine switches to a path that queues every
  layer's experts at once; here it never completed. Gating it behind an environment variable and defaulting it off
  turned "hangs after 120 s" into 5/5 completed runs at 8192-token chunks. The same defect also appeared as extreme
  slowness (3.5 tokens/s) in other runs, so treat "slow" and "stuck" as the same bug until proven otherwise.
- **Rounding intrinsics are slow *and* wrong.** dpct maps `__fmaf_rn` and friends to device math library calls
  (helper function, ~90 lines of expansions); the SYCL equivalent `sycl::fma` is ~2× faster on the hot layer, is
  required by the SYCL 2020 spec to round correctly, and an exhaustive audit (4,194,304 cases, integer-only reference)
  found the hardware path exact and the library call off by 1 ULP in 32,888 of them. So this replacement is a
  correctness fix, not a speed hack — audit any such change the same way before accepting it.
- **`dpct::dp4a` is emulated in software.** Where the integer dot product did become a real `dp4a` instruction, the
  engine got no faster: the cost is in packing/masks/loads, not in the multiply-accumulate. Worth knowing before
  spending a week on it.
- **Device math intrinsics in a shared header break the host link.** Including them unconditionally gave every host
  object file its own definition (`multiple definition` at link time, after 100% of translation units compiled). Put
  the include inside the device-only guard.
- **Two copies of dpct in one binary.** The vendored dpct and the one inside llama.cpp's SYCL backend define the same
  globals with different layouts; the linker keeps one, the other reads its fields from an adjacent object. Adding a
  global to that header makes it crash. The fix is to stop calling the second copy's queue helper and pass the queue
  explicitly.
- **One-line fixes in two vendored headers** (a destructor using a queue that has already been destroyed → 1-line
  `dpct_free` change; a `nullptr` stream in the MTP path). Expect to keep a small patch series against the generated
  tree and against dpct's runtime headers, and keep it in one place.

Performance work on top of the port (Q6_K weight reordering, splitting the fused grouped kernel, prefetching the
n-gram table, MTP partial vocabulary) is all switchable by environment variable. Not every idea paid off — several
micro-optimizations that were 1.5× on a microbenchmark did nothing in the engine because the microbenchmark did not
use the row counts the engine actually uses.

## Credits and licence

- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 — the engine, the format, all of the design. This port
  is a derivative work; licence MIT, same as upstream.
- **llama.cpp / ggml** — the SYCL backend, used for the quantized kernels and the pinned ggml build.
- **SYCLomatic** — the CUDA→SYCL conversion, and **oneAPI / oneMKL** for the compiler, runtime and BLAS.

Questions, corrections and "that number looks wrong" are welcome as issues.
