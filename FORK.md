# adderek/llama.cpp — a TurboQuant fork for AMD RDNA3

This is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with a compressed KV cache
(TurboQuant) and a set of ROCm fixes and kernels for one machine: two **Radeon RX 7900 XTX**
cards (gfx1100, RDNA3) on Linux with ROCm 7.2.4. It is used daily to serve local models, and
everything below was measured there.

> **Hardware scope.** Built and tested **only on RX 7900 XTX (gfx1100), ROCm 7.2.4, Linux**.
> See [What runs where](#what-runs-where) before using it on anything else.

## Lineage

```
ggml-org/llama.cpp ──> domvox/llama.cpp-turboquant-hip ──> adderek/llama.cpp (this)
      (upstream)          (TurboQuant for HIP, gfx1100)        master
```

- [domvox](https://github.com/domvox/llama.cpp-turboquant-hip) ported TurboQuant KV
  compression to HIP. That fork has since stopped following upstream.
- This fork took it over, merges **upstream directly** every few weeks (last: 2026-09-28,
  `4364bf723`), and has grown well beyond it: ~100 commits of its own.
- Branch `master` is the one in use. `domvox-main` only tracks domvox for comparison.

## What is different from upstream

### 1. TurboQuant KV cache (`--cache-type-k/-v turbo2|turbo3|turbo4`)

K and V are rotated with a Walsh–Hadamard transform and stored as PolarQuant indices plus a
norm per 128-value block. Queries are rotated into the same basis and the attention output is
rotated back, so the model sees ordinary attention.

| type     | bits/value | vs f16 | |
|----------|-----------:|-------:|---|
| `turbo4` |      4.25  |  3.8x  | the default on this box |
| `turbo3` |      3.125 |  5.1x  | |
| `turbo2` |      2.125 |  7.5x  | |
| `q8_0` (upstream, for scale) | 8.5 | 1.9x | |

Quality has been checked with needle-in-context recall and the fork's regression tests, not
with a perplexity or benchmark study; pick the type per model.

Heads that are not a multiple of 128 are zero-padded in the cache (for example MLA's 576
becomes 640). `TURBO_LAYER_ADAPTIVE` can give chosen layers a wider type, see
[environment variables](#environment-variables).

Every attention path has to rotate Q itself. Supported and covered by tests:

| attention path | example models | status |
|----------------|----------------|--------|
| dense / GQA | Llama, Qwen3, the attention layers of Qwen3.5/3.6 hybrids | yes |
| sliding-window (iSWA) | Gemma 3/4 | has the rotation, not covered by the fork tests |
| QSA (sparse top-k) | Qwen3.8-Flash-Next (`qwen4exp`) | yes — fixed a silent wrong-basis bug, see [docs/fork](docs/fork/BUG_TURBOQUANT_QSA_Q_ROTATION.md) |
| MLA | DeepSeek-V2 family | yes, but on CUDA/HIP its FlashAttention runs on the **CPU** (no GPU kernel for head 640) |
| DSA (MLA + indexer) | DeepSeek-V3.2, GLM-5 (`glm-dsa`) | yes, same CPU caveat as MLA |
| DeepSeek-V4 (`deepseek4`) | DeepSeek-V4-Flash | **no** — the graph has no turbo handling and asserts |

Indexer caches (QSA, DSA) always stay f16: their keys are read back with `GET_ROWS`, which
has no turbo implementation.

K and V may use different types: FlashAttention has kernels for turbo K with q8_0 or f16 V and
the other way round, and for turbo2/3/4 mixed with each other. `llama-bench` accepts the turbo
types in `-ctk` / `-ctv`.

### 2. FlashAttention kernels for turbo on RDNA3

- **Prefill** (more than 8 query rows): turbo K/V are dequantized to f16 once per layer and the
  regular WMMA/tile kernels run on them, instead of the VEC kernel, which is built for a few
  query rows. `93fa67132`.
- **Decode** (up to 8 query rows, GQA): a TILE kernel reads turbo4 blocks straight into
  shared memory, so one pass serves every Q head of a GQA group; VEC dequantized the same
  block once per Q head. `b546d6118`, HIP only.

Measured on one 7900 XTX, turbo4, `llama-bench`, tokens/s:

| model | context | prefill before → after | decode before → after | decode, f16 cache |
|-------|--------:|-----------------------:|----------------------:|------------------:|
| Ornith-1.0-35B-A3B Q4_K_M | 16k | 875 → 2554 | | |
| | 32k | | 85.4 → 104.5 | 94.8 |
| | 64k | 277 → 1497 | 71.0 → 95.4 | 86.0 |
| Qwen3.6-27B IQ4_NL | 16k | 328 → 838 | | |
| | 64k | 112 → 595 | 25.9 → 27.5 | |

On Ornith-35B a turbo4 cache now decodes faster than an f16 one at long context, while its KV
cache takes 3.8x less memory.

### 3. MoE models larger than VRAM

- `GGML_CUDA_OP_OFFLOAD_DEVICES` picks which GPU runs the large-batch matmuls of experts kept
  in RAM. Upstream takes the first device, which on this box is the card behind a PCIe Gen3 x4
  chipset link; pointing it at the x16 card doubled prefill of Qwen3.8-Flash-Next
  (305 → 600 tokens/s). `dfa85e69b`.
- **Models larger than VRAM + RAM, streamed from NVMe** — branch
  [`moe-tier`](https://github.com/adderek/llama.cpp/tree/moe-tier), **not in `master`** yet
  (it is based on `master` before the 2026-09-28 upstream merge):
  - experts are permuted hottest-first by the routing counts in an imatrix
    (`tools/moe-tier/moe-tier.py`), then each fused expert tensor is split at load time:
    the hottest `LLAMA_MOE_HOT` experts per layer stay on the GPU, the rest go to RAM;
  - with `LLAMA_MOE_DIRECT=1` the cold experts are read from the file with O_DIRECT, by
    several threads, into an arena the process owns and locks in RAM, instead of faulting
    them in through the mmap and the page cache;
  - the page cache is kept small when the model does not fit (no whole-file prefetch at
    load, experts copied to VRAM are dropped from it).

  Measured on a 397B MoE at Q4_K_M (244 GB against ~159 GB of VRAM + RAM): decode
  0.9 tokens/s through mmap, **2.4 tokens/s with the O_DIRECT arena**; the same model at
  Q2_K, which fits, runs at 5.4. On a model that fits in VRAM + RAM, per-expert placement
  gave +30% decode over `--n-cpu-moe` at 25% of the experts in VRAM (Ornith-35B).
  Environment variables and usage: `tools/moe-tier/README.md` on that branch.

### 4. Server behaviour that differs from upstream

- **Reasoning is off unless asked for.** With the default `--reasoning auto`, upstream enables
  thinking whenever the chat template supports it; here it needs `--reasoning on`
  (`99792a7cd`). Reason: Qwen3.5/3.6 with thinking enabled put the whole answer into
  `reasoning_content` and left `content` empty for agents. Set it explicitly when moving
  presets between the two.
- **Speculative decoding parameters can be set per request** (draft-model parameters, n-gram
  sizes and the speculation type); upstream has this disabled. `901f0234f`.
- Log lines carry **wall-clock time** (`hh:mm:ss.mmm`) instead of time since start, and slot
  lines show **busy/total slots** (`| 3/4 |`). `6b8a014a9`.
- Stall watchdog and the prompt-cache fault fix: see section 6 below.

### 5. Models and formats

- **K2-Horizon** (MoVA) architecture and converter. `f11f1c137`.
- Qwen3.8-Flash-Next (`qwen4exp`) arrived from upstream; the fork adds turbo support on its
  QSA path.
- **TurboQuant weight types `TQ3_1S` / `TQ4_1S`** (WHT-rotated 3- and 4-bit Lloyd-Max, 32
  values per block), inherited from domvox: CUDA/HIP can load and run them (a fused mat-vec
  kernel; `TQ4_1S` is converted to q8_0 at load). `llama-quantize` cannot produce them, and
  they are untested here.
- The multimodal loader fails the load when the vision encoder's compute buffers do not fit,
  instead of crashing on the first image. `a04a4cf4d`.

### 6. Robustness fixes (ROCm)

- **GPU memory fault on prompt-cache restore**: glibc returned freed heap chunks to the kernel
  and tore down pages a DMA copy was still reading. The server now keeps the heap mapped
  (`LLAMA_KEEP_HEAP_MAPPED=0` opts out). `89231e3b4`, [write-up](docs/fork/BUG_GPU_FAULT_prompt_cache.md).
- **turbo4 decode hang**: a cross-stream race in the turbo quantization kernel launches.
  `1a1bbb8fc`, [write-up](docs/fork/BUG_GPU_HANG_turbo4.md).
- **Abort under CUDA-graph capture**: hipCUB's segmented sort is illegal inside a capture and
  killed qwen4exp decode. HIP builds no longer use hipCUB, as upstream. `3093ada73`.
- **Server stall watchdog** (`LLAMA_STALL_WATCHDOG_SECS`): dumps all thread backtraces when a
  slot makes no progress, plus wall-clock timestamps and busy counters in the logs. `6b8a014a9`.
- `--hugepages`: back model weights with 2 MiB hugetlb pages (Linux, mmap path). By Jeremiah
  Blanchard, proposed upstream, not in upstream `master`.

### 7. Smaller RDNA3 changes

IQ1_M in the MMQ kernels (then disabled there for correctness), BF16 mat-vec tuning, 256-wide
heads in the MMA FlashAttention kernel, mixed q8_0/q4_0 K/V FlashAttention. History and
status: [docs/fork/ROCM_RDNA3_PLAN.md](docs/fork/ROCM_RDNA3_PLAN.md).

Decode is launch-bound on this card (about 1600 kernels per token, GPU idle half the time), so
the mat-vec path keeps the last two q8_1 quantizations of its inputs and reuses one when a later
mat-vec in the same graph evaluation reads the same tensor, or a reshape of it (Q/K/V projections,
routed and shared expert gate/up), and a fused rms_norm * weight writes that q8_1 copy itself
when a mat-vec will read it. On Ornith-1.0-35B-A3B that removes 230 of 351 `quantize_q8_1`
launches per token (1622 -> 1391 kernels), +7–8% decode; the output is bit-identical.
`GGML_CUDA_MMVQ_SRC1_CACHE=0` turns it off; it is also off under `GGML_CUDA_GRAPH_OPT=1`.
The shared expert gate of qwen3next/qwen35moe (`sigmoid(gate) * shexp`, then the adds into the
routed output and the residual) runs as one kernel instead of three: another 80 launches per token
on the same model (1391 -> 1311), about +3% decode, bit-identical (`test-sigmoid-gate-fusion`).
The residual add in front of such an rms_norm is computed in the same kernel as well (1311 -> 1281).
Both together: 84.4 -> 87.9 t/s at depth 0, 81.0 -> 84.5 at 16384 (tg128, one 7900 XTX).
`softplus(alpha + dt) * a` of the linear-attention layers is one kernel too (1281 -> 1251, no
broadcast add or mul left in decode): 88.4 -> 90.0 t/s (tg256), bit-identical (`test-add-unary-mul-fusion`).
The GATED_DELTA_NET kernel applies the sigmoid of beta itself (1251 -> 1221, no standalone
sigmoid left): +1.1% (tg512), bit-identical (`test-gdn-beta-sigmoid-fusion`).

On RDNA3 the MMA FlashAttention kernel (upstream) summed P*V in f16 for every head size but 80
and 112. Over a long context that sum overflows or loses precision: attention spread evenly
over 32k positions came out 20-67% wrong, and on real Qwen3.8-27B data at 25k positions the
output was off by NMSE 2e-3 against the CPU (VEC kernel: 1e-5). The fork keeps it in f32 (NMSE
4e-8): perplexity at 32k context 2.4127 -> 2.4097 (Qwen3.8-27B), 2.8822 -> 2.8714 (Ministral-8B),
for 0-1.4% of prefill speed with 256-wide heads and up to 3.7% with 128-wide heads at depth 16k.

Also on RDNA3, Q2_K, Q6_K and IQ2 weights left MMQ above 128 rows for a hipBLAS GEMM with an f16
output, which overflowed on large activations (Ornith-1.5-397B answered only "/" at ubatch 512).
They now stay on MMQ, and the remaining hipBLAS f16 path writes f32 (-1 to -2% prefill).
The Q4_1 / Q5_1 / Q8_1 dot products multiplied the block scales in half2 as well; the block
minimum times the activation sum overflowed in the mat-vec path. They multiply in f32 now.
`GGML_TEST_INPUT_SCALE=N` widens every input of `test-backend-ops` by N, which is how that one
was found (at 100 it was the only op that went to inf where the CPU stayed finite).

## What runs where

| | RX 7900 XTX (gfx1100) | other RDNA3 (gfx1101/1102) | RDNA4 / CDNA | NVIDIA (CUDA) | CPU | Metal / Vulkan / SYCL |
|---|---|---|---|---|---|---|
| everything from upstream | yes | as upstream | as upstream | as upstream | yes | as upstream |
| turbo KV cache | **tested** | likely, untested (set `AMDGPU_TARGETS`) | untested; HIP quantize kernels assume wave32 | never built; may not compile | yes (reference, slow) | **no implementation** — falls back to CPU or fails |
| turbo decode TILE kernel | **tested** | likely | untested | no (HIP only; VEC used) | — | — |
| kernel thresholds | tuned here | not tuned | not tuned | — | — | — |

Non-turbo fork features (op-offload device choice, `--hugepages`, server watchdog, K2-Horizon,
the prompt-cache fix) are not tied to RDNA3.

## Build

Used on this box (the `_` script in the repository root wraps this, plus ccache and an archive
of the previous binaries):

```sh
cmake -B build -G Ninja \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600"
cmake --build build -j
```

ROCm is held at 7.2.4 on purpose: newer RCCL cannot be patched to run on a card without PCIe
atomics, which the chipset-attached card lacks.

Typical run:

```sh
llama-server -m model.gguf -ngl 99 -fa on --cache-type-k turbo4 --cache-type-v turbo4 -c 131072
```

## Environment variables

| variable | default | effect |
|----------|---------|--------|
| `GGML_CUDA_OP_OFFLOAD_DEVICES` | all | comma-separated device indices allowed to take offloaded large-batch ops (MoE experts in RAM). Only meaningful when devices are visible under their native numbers |
| `GGML_CUDA_TURBO_TILE` | `1` | `0` sends turbo4 decode back to the VEC kernel |
| `GGML_CUDA_TURBO_VEC_MAX_BATCH` | `8` | above this many query rows, turbo K/V are dequantized to f16 for FlashAttention |
| `TURBO_LAYER_ADAPTIVE` | `0` | per-layer cache types: `1` q8_0 for the first and last 4 layers, `2` q8_0 for the last 8, `5`–`7` mixed turbo/q8_0 V (`7` = q8_0 V on the first and last 2 layers, turbo2 elsewhere) |
| `TURBO_INNERQ` | off | calibrate a per-channel scale over this many tokens before quantizing (`TURBO_INNERQ_STRENGTH`, 0–1, default 0.5) |
| `LLAMA_STALL_WATCHDOG_SECS` | off | server: dump backtraces after this many seconds without progress (`LLAMA_STALL_WATCHDOG_GDB=0`: log only) |
| `LLAMA_KEEP_HEAP_MAPPED` | `1` | server: `0` lets glibc return freed heap to the kernel again (reintroduces the prompt-cache fault) |
| `LLAMA_MOE_HOT` and the other `LLAMA_MOE_*` | off | per-expert MoE tiering: hot experts on the GPU, cold ones in RAM or paged from the file with O_DIRECT; models permuted with `tools/moe-tier`. Variables and measurements: [tools/moe-tier/README.md](tools/moe-tier/README.md) |
| `GGML_CUDA_MMVQ_SRC1_CACHE` | `1` | `0` turns off the reuse of q8_1 mat-vec inputs (and the fused rms_norm writing them) |

`GGML_CUDA_GRAPH_OPT=1` (upstream, experimental) is not safe on this machine: on
gemma-4-26B-A4B greedy output differs from run to run, with fusion disabled as well, and is
stable again with `GGML_CUDA_DISABLE_GRAPHS=1` or without it, so the concurrent streams it adds
inside CUDA graphs race on HIP. Ornith (no three-way attn_norm fork) is unaffected. Nothing here
sets it (checked 2026-09-30).

## Tests added by the fork

Registered in ctest; each lives in its own file so upstream merges do not conflict.

| test | guards |
|------|--------|
| `test-turbo-kv-{dense,gqa,qsa,mla,dsa}` | attention output with a turbo cache stays within a few times the q4_0 error of an f16 cache on every attention path; a missing Q rotation shows as 34–51x. Prefill and 8 decode steps |
| `test-turbo-kv-{gqa,qsa}-vec` | the same with the VEC kernel forced |
| `test-fattn-turbo4` | the turbo4 TILE kernel against the CPU backend: head 128/256, GQA 1–8, 1–8 query rows |
| `test-op-offload-devices` | `GGML_CUDA_OP_OFFLOAD_DEVICES` restricts op offload |
| `test-top-k-graph-capture` | top-k over long rows survives CUDA-graph capture |
| `test-mmvq-src1-cache` | a reused q8_1 mat-vec input is never stale: across weights, slots, MUL_MAT_ID, reshapes, in-place writes, graph evaluations, and from the fused rms_norm (with and without the residual add) |
| `test-sigmoid-gate-fusion` | fused `sigmoid(gate) * x + adds` gives the bits of the unfused kernels |
| `test-add-unary-mul-fusion` | fused `op(x + bias) * g` gives the bits of the unfused kernels |
| `test-gdn-beta-sigmoid-fusion` | GATED_DELTA_NET applying `sigmoid(beta)` itself gives the bits of the unfused kernels, snapshot-copy fusion included |
| `test-fattn-long-kv` | FlashAttention evenly spread over 32k KV positions returns the value, not inf or a 20-67% error: the RDNA3 MMA kernel used to sum P*V in f16 |
| `test-mul-mat-f16-range` | mat-muls whose results pass the f16 range stay finite: Q6_K / Q2_K / F16 at large batch (the RDNA3 hipBLAS path wrote f16), Q4_1 / Q5_1 at one row (half2 block-scale product) |
| `test-moe-tier` | a tiny qwen3moe split into hot and cold experts, through the arena and windowed paths and `moe-tier.py permute`, gives the logits of the untiered model (bit-identical on CPU, NMSE < 1e-12 on each GPU and both) |

`test-fattn-turbo4`, `test-op-offload-devices`, `test-top-k-graph-capture` and the four fusion
and cache tests above skip without a CUDA/HIP device; `test-turbo-kv-*` also run on the CPU backend.

## More

- [docs/fork/](docs/fork/) — bug write-ups and the RDNA3 plan.
- [rescued-patches/](rescued-patches/) — uncommitted work recovered from other checkouts, not applied.
- `scripts/capture_hang.sh`, `scripts/repro_turbo4_hang.*` — GPU-hang forensics.
