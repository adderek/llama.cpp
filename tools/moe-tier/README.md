# moe-tier

Place the experts of a MoE layer on different backends by how often the router picks
them: the hottest in VRAM, the rest in a CPU buffer that stays in the mmap and can page
in from disk.

## Prepare a model

```
moe-tier.py stats   IMATRIX                 # per-expert routing counts as CSV, skew table on stderr
moe-tier.py permute MODEL IMATRIX OUT       # rewrite MODEL with the experts sorted hottest first
```

The counts come from the `.counts` tensors that `llama-imatrix` already writes for MoE
tensors, so no extra measurement is needed. `permute` reorders the expert dimension at
byte level, so it works on any quantisation: quantise first, permute last. An MTP block
the imatrix run never reached is left as it is.

The permutation is what makes the rest work: after it, "the hottest N experts" is a
prefix of each fused expert tensor, so it can be a separate tensor without touching the
file layout.

## Run

| variable | meaning |
|---|---|
| `LLAMA_MOE_HOT=N` | keep the first N experts of every MoE layer where the layer lives (GPU, `-ot` applies); the rest become `*_exps.weight.cold` in a plain CPU buffer |
| `LLAMA_MOE_WARM=M` | of the cold experts, expect the first M to be reused; everything past them is dropped from the page cache after use |
| `LLAMA_MOE_DIRECT=1` | page the cold experts from the file with O_DIRECT instead of reading them through the mmap; needs the settings below |
| `LLAMA_MOE_ARENA=N` | experts pinned in the RAM arena (N x 61 layers x slab bytes, so size it against free RAM) |
| `LLAMA_MOE_FRAMES=F` | replaceable frames on top of those; the arena path is used only while the ubatch routes to at most F experts, so prompt processing needs a small `-ub` |
| `LLAMA_MOE_ARENA_HEADROOM=G` | GiB of RAM that must stay available after the arena is locked (default 32); below that the load fails at once, since with less it hung in the GPU driver |
| `LLAMA_MOE_OVERLAP=1` | prefill with more experts than frames: split the frames in two halves and read the next window of experts into one half while the other computes; windows get half as wide |
| `LLAMA_MOE_READERS=R` | threads issuing the page-ins (default 8); a single stream leaves most of the NVMe unused |
| `LLAMA_MOE_STATS=1` | log hit/miss and where the routed experts fall in the permuted order |
| `LLAMA_MOE_TRIM_EVERY=T` | drop those ranges every T decoded tokens (default 8) |
| `GGML_MMID_PREFETCH=1` | read the experts a `mul_mat_id` needs in whole slabs instead of 64 KiB faults |
| `LLAMA_NO_MMAP_PREFETCH=1` | do not ask the kernel to read the whole file at load; implied by `LLAMA_MOE_HOT` |

`gpu-check.sh MODEL [OUT_DIR]` runs a KLD check against the unsplit model and a
`llama-bench` sweep over several values of N.

Not supported with the split: per-expert scales (`*_exps_s`) and LoRA on expert tensors.

## What to expect

On a model that fits in VRAM+RAM this beats the per-layer `--n-cpu-moe` split at the
same VRAM, the more so the less VRAM there is (+30% decode at 25% of experts in VRAM on
ornith-1.0-35B). On a model that spills to disk it does not help: see the 397B section
of `MOE-EXPERT-TIERING.md` in the workspace, where the limit is page-fault reclaim
rather than expert placement.
