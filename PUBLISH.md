# Publication procedure

This file is the handoff checklist for the public branch. It is a procedure,
not a release claim. The CMake file and all runtime commands are **unbuilt and
unconfirmed** in this publication pass.

## What goes on the branch

Use fork `sarashin65/Strata` `main` at `236d5f2` (the Engine 0.1.17
generation) as the base, then publish the converted `sycl-port` tree. The
publication-set inventory is **249 files**: 197 source-tree files
(`CMakeLists.txt`, `src/`, and `include/`), 45 DPCT headers, 4 scripts, and 3
public documents. The count includes regular files under `dist/`; directories
are not counted. Its required content is:

- the manifest's 197-file `CMakeLists.txt` + `src/` + `include/` source tree;
- the bundled `tools/include/dpct/` runtime headers, with their original Intel
  notices intact;
- `scripts/build.sh`, `scripts/run.sh`, `scripts/prepare-data.sh`, and
  `scripts/verify.sh`;
- the public cookbook, `NOTICE.md`, and this procedure;
- the upstream `LICENSE`, upstream `third_party/ggml/` files, and the upstream
  data files that are already part of the base tree.

The 197-file source manifest is 76 unchanged files, 34 replaced files, and 87
added converted/header files. Reconcile the final path list and byte counts
against `MANIFEST-1005.md` and `manifest-delta.tsv` before a real publication.
Do not treat a copied workspace count as the final branch count.

The four source include edits are intentional and limited to these files:

- `src/kernels/cpu/iq_avx2.cpp`
- `src/kernels/cpu/iq_avx512.cpp`
- `src/prefill/ggml_cuda_host.dp.cpp`
- `src/prefill/moe_mmq.dp.cpp`

The development llama.cpp `ggml-common.h` and upstream-main
`third_party/ggml/ggml-common.h` have matching SHA-256 content, with observed
prefix `0061131b…`; because `third_party/` remains the upstream tree, the
relative include replacement is equivalent for the measured source.

## Delete these 87 upstream files

Delete exactly the following upstream CUDA files, as listed in
`manifest-delta.tsv`; do not invent additional deletions:

```text
include/strata/kernels/native_qsa_score.hpp
src/core/concurrent_main.cpp
src/core/device.cu
src/core/expert_cache.cpp
src/core/expert_source.cpp
src/core/graph.cpp
src/core/layer.cpp
src/core/mtp.cpp
src/core/native_dense.cpp
src/core/native_head.cpp
src/core/overlap_main.cpp
src/core/pinned.cu
src/core/session.cpp
src/core/verify.cpp
src/core/weights.cpp
src/kernels/bf16_gemv_parity.cpp
src/kernels/cuda/bf16_gemv.cu
src/kernels/cuda/cvec.cu
src/kernels/cuda/dequant_bf16.cu
src/kernels/cuda/dequant_s2.cu
src/kernels/cuda/elementwise.cu
src/kernels/cuda/fused_gdn.cu
src/kernels/cuda/fused_gr.cu
src/kernels/cuda/gdn.cu
src/kernels/cuda/gr.cu
src/kernels/cuda/iq_kernels.cu
src/kernels/cuda/kv_q4.cu
src/kernels/cuda/kv_q8.cu
src/kernels/cuda/kv_stream.cu
src/kernels/cuda/native_bf16.cu
src/kernels/cuda/native_flash_attn.cu
src/kernels/cuda/native_gdn.cu
src/kernels/cuda/native_gdn_preprocess.cu
src/kernels/cuda/native_gr_norm.cu
src/kernels/cuda/native_gr_postops.cu
src/kernels/cuda/native_mmvq.cu
src/kernels/cuda/native_moe.cu
src/kernels/cuda/native_ple_postops.cu
src/kernels/cuda/native_qsa.cu
src/kernels/cuda/native_qsa_indexer.cu
src/kernels/cuda/native_qsa_score.cu
src/kernels/cuda/native_rope.cu
src/kernels/cuda/native_router.cu
src/kernels/cuda/ple.cu
src/kernels/cuda/qsa.cu
src/kernels/cuda/qsa_decode_attn.cu
src/kernels/cuda/qsa_select.cu
src/kernels/cuda/quantize_act.cu
src/kernels/cuda/rope.cu
src/kernels/cuda/router_top10.cu
src/kernels/cuda/s2_expert_grouped.cu
src/kernels/cuda/s2_gemv.cu
src/kernels/cuda/s2_gemv_fast.cu
src/kernels/cuda/s2_gemv_q8.cu
src/kernels/cuda/s2_gemv_quads.cu
src/kernels/cuda/s_gemv.cu
src/kernels/cuda/sampler.cu
src/kernels/cuda/shared_expert.cu
src/kernels/cuda/verify_kernels.cu
src/kernels/cvec_parity.cpp
src/kernels/dequant_bf16_test.cpp
src/kernels/dequant_s2_parity.cpp
src/kernels/elementwise_parity.cpp
src/kernels/gdn_parity.cpp
src/kernels/gr_parity.cpp
src/kernels/iq_parity.cpp
src/kernels/kv_q4_parity.cpp
src/kernels/kv_q8_parity.cpp
src/kernels/kv_stream_parity.cpp
src/kernels/native_expert_parity.cpp
src/kernels/ple_parity.cpp
src/kernels/qsa_parity.cpp
src/kernels/quantize_act_parity.cpp
src/kernels/rope_parity.cpp
src/kernels/router_top10_parity.cpp
src/kernels/s2_gemv_parity.cpp
src/kernels/s2_gemv_q8_parity.cpp
src/kernels/s_gemv_parity.cpp
src/kernels/s_gemv_q8k_parity.cpp
src/kernels/sampler_parity.cpp
src/kernels/shared_expert_parity.cpp
src/prefill/gemm.cu
src/prefill/ggml_cuda_host.cu
src/prefill/kernels.cu
src/prefill/moe_mmq.cu
src/prefill/prefill.cpp
src/program/generate.cpp
```

The converted `.dp.cpp` files replace the corresponding CUDA implementation
files; this is a replacement, not a second copy. Retain the changed headers
listed in the manifest.

## Do not include

Do not put any of these in the public branch or publication archive:

- build directories, executables, object files, compiler-identification
  output, JIT caches, raw logs, benchmark output, or comparison data;
- GGUF weights, generated IQ/Q2 packs, generated MTP files, credentials, or
  provider-specific service configuration;
- `tools/vision/strata_vision.cpp`, because the current public CMake does not
  reference it;
- `MainSourceFiles.yaml` or `run-build.sh`, because their development
  assumptions are not public-tree inputs;
- `.before-*`, `.orig`, `.pool-overlap-02-candidate2`, or other attempt and
  backup files;
- `STATE.md` or any private workstation state.

Keep upstream `data/expert-profile.bin`,
`data/expert-profile-coder.bin`, and `data/draft_vocab.bin` when they belong to
the upstream base. They are not reader-generated publication outputs. The
identity of the expert profile used by the published performance run remains
unconfirmed.

## Checks before a real branch operation

The allowed local acceptance checks for this package are:

The first acceptance check is the orchestrator's forbidden-path scan over the
five named public documents and scripts; it must report zero matching lines.
The source-tree check is a recursive comparison whose only expected changes
are the four include-line replacements named above; all other `dist/src/`
bytes must remain unchanged. No build, GPU check, network access, SSH session,
GitHub CLI operation, commit, or push is part of this handoff.


## Branch operation rule

When an authorized publisher performs the later branch update, stage the
manifest-approved files and review the diff before opening the publication
request. **Do not run `git push`.** This task creates a local publication set
only; it does not commit or push anything.
