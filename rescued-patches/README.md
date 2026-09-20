# Rescued patches

Work that existed only as an uncommitted diff in a working copy, with no remote
that could receive it. Parked here because this repo is the only one in
`~/src/llama.cpp/` that pushes anywhere.

Nothing here is applied. These are rescues, not a queue.

## `rdna3-persistent-mmq-20260520.patch`

Found 2026-09-20 as uncommitted changes in `~/src/llama.cpp/src` — a plain
`ggml-org/llama.cpp` clone whose only remote is upstream, i.e. read-only. The
tree had been sitting on it since 2026-05-20 and a proposal to delete that
directory as redundant would have destroyed it.

Base: `a8681a0ed` (ggml-org master, 2026-05-20). 156 insertions across
`ggml/include/ggml.h`, `ggml/src/ggml-cuda/mmf.cu`, `ggml/src/ggml-cuda/mmq.cuh`.

What it does:

* adds `GGML_TYPE_TURBO_BGS = 16`
* adds `mul_mat_q_persistent`, a persistent matmul kernel for RDNA3 (gfx1100 —
  the 7900 XTX in this box). Each block loops and claims tiles atomically, in
  chunks to keep L2 atomic pressure down. No K-split, so no fixup kernel and no
  second launch.
* routes to it from `mmf.cu` when `GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 15`

Status: **unverified.** No measurement is recorded anywhere, and it does not
apply to this fork — `ggml-cuda/mmq.cuh` and `mmf.cu` have diverged too far
(turbo kernels, FATTN work). Reviving it means a manual port, then a benchmark
against the current mmq path, not a `git apply`.

## `src-build-script_.sh`

The one-character `_` build script from the same directory, kept for context —
it records how that tree was configured.
