# Bug: GPU memory access fault on prompt-cache restore (glibc unmaps DMA source)

**Status:** root-caused, fixed, verified 2026-07-27.
**Symptom:** `llama-server` dies with

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7f5065c00000.
Reason: Page not present or supervisor privilege.
```

always within a millisecond of a prompt-cache save/restore
(`srv get_availabl: updating prompt cache` / `load: - found better prompt`).
Not model-specific: hit on Qwen3-Coder-Next-UD-IQ4_XS *and* on the dense
ornith-1.0-35b-Q4_K_M, both `--cache-type-k/v turbo4 -c 131072 --fit on`,
4 slots, `kv_unified`. Five crashes in one afternoon; reproducible in ~3 min.

---

## 1. Evidence

**Kernel log** (`journalctl -k`, the decisive part — `dmesg` needs root):

```
amdgpu 0000:06:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:593)
   Process llama-server pid 33884
   in page starting at address 0x00007f5065c00000 from client 10
GCVM_L2_PROTECTION_FAULT_STATUS:0x00801A31
   Faulty UTCL2 client ID: SDMA0 (0xd)   <- DMA copy engine, not a shader
   PERMISSION_FAULTS: 0x3
   RW: 0x0                                <- a READ
   MORE_FAULTS: 0x1                       ("1035 callbacks suppressed")
```

Consecutive pages (`…c00000, c01000, c02000 …`) — a linear copy walking into
memory the GPU is no longer allowed to touch.

**Core dumps** (`coredumpctl`, cores are kept — use them, they hold the answer):

- The faulting VA is *host* memory holding a llama-server state buffer. In the
  18:38 crash it sat inside a hole in the address space exactly `0xF1AD000`
  = 241.677 MiB wide — the size of the prompt-cache entry that had just been
  restored (`saving prompt with length 26663, total state size = 241.674 MiB`).
- In the 19:32 crash the main thread was caught **inside `munmap()`**, called
  from `server_prompt_cache::load` (freeing a 62.8 MiB context checkpoint),
  while the faulting VA was in a *still-mapped* neighbouring region — i.e. the
  fault is not merely "freed too early", the GPU-side mapping of a nearby live
  buffer had been invalidated too.
- Main thread in the other cores: `get_available_slot` → `server_prompt_cache::load`
  → (returned) `launch_slot_with_task` → `common_sampler_init`. The abort always
  comes from the HSA fault handler thread, so the stack shows where the host had
  moved on to, not where the offending copy was issued.

## 2. Root cause

The state buffers (`server_prompt::data.main`, `common_prompt_checkpoint::data_tgt`)
are ordinary `std::vector<uint8_t>` heap allocations, 60-250 MiB each, and the
backend DMAs **directly out of them** (`llama_state_seq_set_data_ext` →
`ggml_backend_tensor_set` → `hipMemcpyAsync` H2D per KV layer).

glibc gives allocations that size their own `mmap`, and returns them to the
kernel on `free()` (and trims the arena around them). That teardown also drops
the GPU driver's mapping of those pages — including, as the 19:32 core shows,
for buffers *adjacent* to the one being freed. The SDMA engine, still walking
the transfer, then hits a page the kernel has taken away → VM fault → abort.

Note that the per-copy `cudaStreamSynchronize(cudaStreamPerThread)` in
`ggml_backend_cuda_buffer_set_tensor` does **not** save us here: the hazard is
the mapping being torn down under the copy engine, not simply a copy that has
not finished yet.

## 3. Fix

`tools/server/server.cpp` — `keep_heap_mapped()`, called first thing in
`llama_server()`:

```c
mallopt(M_MMAP_MAX,       0);       // large allocations do not get their own mmap
mallopt(M_TRIM_THRESHOLD, INT_MAX); // the arena top is never returned to the kernel
```

Freed state buffers stay in the allocator's free lists and are reused, so no
mapping the GPU knows about is ever destroyed. Opt out with
`LLAMA_KEEP_HEAP_MAPPED=0`. Equivalent without a rebuild:
`MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2147483647 llama-server ...`.

Three supporting changes, none of which fix the crash on their own but all of
which close real windows found while chasing it:

- `src/llama-context.cpp`: `state_seq_get_data`/`state_seq_set_data` now destroy
  the deferred-IO object and `synchronize()` before returning, so the caller's
  buffer is not still referenced by an in-flight transfer when it returns.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_backend_cuda_synchronize` also waits
  on `cudaStreamPerThread` (where `set_tensor`/`get_tensor` issue their copies),
  after selecting the backend's device — previously it only waited on the
  backend stream, so `synchronize()` did not cover state transfers at all.
- `tools/server/server-task.{h,cpp}`: a restored state buffer is `retire()`d
  instead of being freed the instant the restore returns; retired buffers are
  released one cache operation later.

## 4. Verification

Repro harness: `scripts/repro_prompt_cache_fault.py` — six conversations against
four slots, so every round forces an LRU slot switch, a full state save and a
restore of a returning conversation.

```
# server: go-big.sh with ornith-1.0-35b-Q4_K_M (turbo4 KV, -c 131072, 4 slots)
python3 scripts/repro_prompt_cache_fault.py --convs 6 --rounds 10 --filler 250
```

| build | requests | restores | result |
|---|---|---|---|
| before the fix | 26 | 5 | GPU fault, core dumped |
| `mallopt` fix, no env vars | 60 | 54 | clean, no faults |
| `mallopt` fix, longer soak | 69 | 124 | clean, no faults (run cut short when the 8081 occupant was swapped, not by a failure) |
| env-var equivalent | 36 | 30 | clean, no faults |

Host RSS with `M_MMAP_MAX=0` measured 9.1 GiB about 60 requests in (the model is
mmap'd; the cache states are the only large churn). That is a single sample, not
a trend — worth watching on a long-running server, since holding the heap is the
one cost of this fix.

## 5. Residual risk / follow-ups

- The proper fix is to allocate state buffers from **pinned host memory**
  (`ggml_backend_dev_host_buffer_type`, i.e. `hipHostMalloc`) instead of the
  general heap, which removes the dependency on allocator behaviour entirely.
  That needs a custom allocator threaded through `server_prompt` and
  `common_prompt_checkpoint`; `mallopt` buys the same safety for a few lines.
- Unrelated to `BUG_GPU_HANG_turbo4.md` (that one was a cross-stream race in the
  turbo-quant HIP kernels, fixed 2026-07-06). turbo4 is not implicated here —
  the state transfer is a raw byte copy and the dense Q4_K_M model faults the
  same way.
