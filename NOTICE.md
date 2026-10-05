# Notices

## Strata

Strata is by Niko1221. The upstream copyright notice and the complete MIT
License text remain verbatim in the branch's `LICENSE` file. This public tree
is a derivative work and is not an official upstream Strata release.

The derivative contains CUDA-to-SYCL machine conversion performed with
SYCLomatic, followed by manual fixes for SYCL, oneAPI, queue behavior, memory
movement, and oneMKL integration. The upstream MIT notice does not by itself
make a separate third-party notice disappear; review the dependency notices
before redistributing a complete checkout.

## Intel DPCT runtime headers

`tools/include/dpct/` contains the bundled Intel-derived DPCT helper headers:
45 files totaling 1,561,068 bytes. Their original header notices must remain in
place and are separate from the Strata MIT notice. The headers identify:

- Copyright (C) Intel Corporation
- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
- LLVM license information: <https://llvm.org/LICENSE.txt>

This notice records the attribution; it does not change the text embedded in
the bundled headers.

## Other components

The build uses llama.cpp/ggml, SYCLomatic, Intel oneAPI, and oneMKL. Their
exact versions, notices, and redistribution conditions are not fully verified
for this publication. Do not represent them as covered by Strata's MIT
license. Preserve each component's own notices when obtaining or redistributing
those components.

Model weights, generated packs, MTP runtime files, and provider-specific
artifacts are not included. Obtain them from their providers and follow the
providers' license and usage conditions. The upstream `data/expert-profile.bin`
and `data/draft_vocab.bin` files are inputs retained by the upstream tree;
this publication does not generate replacement versions.
