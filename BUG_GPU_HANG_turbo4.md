# Bug handoff: GPU/HSA lost-wakeup hang (turbo4 KV path, RDNA3)

**Status:** root-caused to the GPU/ROCm layer, not yet fixed.
**Symptom:** llama-server wedges mid-generation — produces zero tokens, never
releases the slot. Recoverable only by killing the client connection (or the
server). Reproduced once under normal load.

---

## 1. What was observed

A `llama-server` instance (the IQ4 instruct model on port 8081) stopped
generating mid-request and hung for >5 minutes. Server log ended at:

```
17:28:19.074 I slot create_check: id  2 | task 2008 | 1/3 | created context checkpoint 4 of 4 (pos_min = 29362, pos_max = 29362, n_tokens = 29363, size = 149.626 MiB)
```

…then total silence. No `print_timing`, no `release`, no further tokens. The
previous task (1946, 17 tokens) completed normally just before. Task 2008 was a
~29k-token prompt decode.

### Exact launch command (the wedged server)

```
llama-server -m /home/adderek/LLM/instruct/bartowski/Qwen_Qwen3.6-27B-IQ4_NL.gguf \
  --port 8081 \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  -c 200000 -fa on -ngl 99 -np 3 -b 4096 -ub 4096 -t 12 \
  --offline --models-max 1 --cont-batching --tools all --jinja \
  --cache-ram 8192 --temperature 0.10 --kv-offload \
  --ctx-checkpoints 4 --reasoning-budget 256 \
  --spec-type ngram-cache --spec-draft-n-max 16
```

The fork-custom flags (NOT upstream) are the suspects:
`--cache-type-k/v turbo4`, `--spec-type ngram-cache`, `--ctx-checkpoints`,
`--reasoning-budget`.

---

## 2. Evidence — this is a GPU/HSA lost-wakeup, NOT a C++ logic deadlock

Captured live from the hung process (PID was 371633) via `gdb -p ... -batch
-ex "thread apply all bt"` and `/proc`:

**Thread backtraces:**

- **Thread 1 (main, LWP 371633):**
  `server_queue::start_loop` → `pthread_cond_clockwait`
  → idle, timed-polling the task queue. NOT inside `llama_decode`/`update_slots`.

- **Thread 36 (compute, LWP 371642):**
  `ioctl` → `libhsa-runtime64.so.1` (ROCm)
  → blocked waiting on a GPU event/signal.

- **Thread 35 (LWP 371643):**
  `kfd_wait_on_events` (AMD KFD driver event wait), same ROCm path.

- Remaining ~32 threads: `__futex_wait` — idle threadpool, parked.

**GPU state at hang time (`rocm-smi`):**

```
GPU%  = 0%       <- GPU is IDLE, doing no compute
SCLK  = 24 MHz   <- clocked down (idle)
MCLK  = 96 MHz
VRAM% = 97%      <- model still resident
Temp  = 46 C, Power = 26 W   <- idle power
```

**CPU sampling (`/proc/<tid>/stat` utime delta over 1s):**
- During the active phase the process burned ~170% CPU for ~30 min wall
  (HSA busy-polling before falling into the blocking wait).
- Once quiesced: both spinning threads → **0 ticks/s** (fully blocked).

**Kernel logs:** `dmesg` showed no `amdgpu` ring timeout / VM fault / GPU reset
(checked without sudo — re-check WITH sudo, see §5).

### Interpretation

The compute thread submitted GPU work, then armed an HSA wait on a completion
signal. The GPU went **idle** (0%, downclocked) — meaning the work either
finished without signaling, or the doorbell/signal was lost — but the host
remains blocked in `hsaKmtWaitOnEvent` forever. Classic **lost-wakeup / signal
race** at the HSA/HIP synchronization boundary.

It is **not**:
- a mutex deadlock (those show `futex_wait`, S-state; no `llama`/`update_slots`
  frames in the running stacks).
- a GPU crash/hang (rocm-smi responsive, no reset, no dmesg fault, VRAM intact).
- the spec-decode checkpoint-restore livelock (no `common_speculative` /
  `update_slots` frames in the live stack — ruled out).

Recoverable: when the client TCP connection was closed, the server released the
slot — consistent with a transient signal race rather than a permanent wedge.

---

## 3. Prime suspect

`ggml/src/ggml-turbo-quant-hip.hip` — the fork's custom RDNA3 KV-cache
dequant/quant HIP kernels, selected by `--cache-type-k/v turbo4`.

Hypothesis: the turbo4 dequant kernel dispatch sets up its HIP stream / HSA
completion signal incorrectly under some condition hit on the large (29k-token)
decode after a context-checkpoint restore — e.g. a kernel that early-returns on
an empty/edge tile without signaling, a missing `hipStreamSynchronize` /
event-record ordering, or a doorbell race. Result: GPU idles, host waits forever.

Secondary (lower priority) candidates if turbo4 is cleared by bisect:
- ngram-cache speculative path (`common/ngram-*.cpp`, `--spec-type ngram-cache`).
- context-checkpoint save/restore interacting with KV offload
  (`--ctx-checkpoints`, `--kv-offload`).

Related fork-custom files to read:
- `ggml/src/ggml-turbo-quant-hip.hip`  (HIP kernels — main suspect)
- `ggml/src/ggml-turbo-quant.c`        (host-side turbo quant)
- `ggml/src/ggml-cuda/turbo-innerq.cuh`
- `ggml/src/ggml-cuda/dequantize.cuh`
- `tools/server/server-context.cpp`    (decode loop ~L3140 checkpoint, L3201 `llama_decode`)

---

## 4. Reproduction + bisect plan

### 4a. Reproduce
Drive the server with a ~17k→29k token context that triggers a context
checkpoint restore, then a follow-up decode (the agent workload did this). A
single large decode after `--ctx-checkpoints` rollover is the trigger window.
Aim for a deterministic repro script that POSTs the same prompt sequence.

### 4b. Bisect the trigger — toggle ONE custom flag at a time, rerun repro:

| Variant | Change | If it stops hanging → culprit |
|---|---|---|
| A | `--cache-type-k/v turbo4` → `q8_0` (or `f16`) | turbo4 HIP kernel **(test first)** |
| B | `--spec-type ngram-cache` → `none` | ngram speculative path |
| C | `--ctx-checkpoints 4` → `0` | checkpoint save/restore |
| D | `--kv-offload` removed | KV offload × turbo4 interaction |

Start with A. It is the most likely and the cleanest signal.

### 4c. Instrument once the trigger is isolated
Run with ROCm tracing to capture the stuck dispatch and the un-signaled event:

```
AMD_LOG_LEVEL=3 HSA_ENABLE_INTERRUPT=0 \
  llama-server ... 2> hsa-trace.log
```

(Also try `HSA_ENABLE_INTERRUPT=1` — flips between interrupt and polled
completion; a behavior change here strongly implies a signal/doorbell race.)

For a kernel-level stack at hang time, attach `rocgdb` (ROCm gdb) instead of
plain gdb — it can show device-side state:

```
rocgdb -p <pid> -batch -ex "info threads" -ex "thread apply all bt"
```

---

## 5. Privileged steps you'll need

These require elevation (the diagnostics, NOT the build/edit):

- `sudo dmesg | grep -iE 'amdgpu|ring.*timeout|VM_L2|gpu reset|kfd'` — confirm
  whether the kernel logged a GPU fault (definitive).
- Attaching a debugger to the *already-running* server (not a child of the
  debugger) is gated by `kernel.yama.ptrace_scope` (currently `1`). Either run
  the debugger via `sudo`, or temporarily `sudo sysctl kernel.yama.ptrace_scope=0`
  (revert after).
- If a future repro truly wedges the GPU ring (dmesg fault), recovery may need a
  GPU reset / module reload (`sudo modprobe -r amdgpu` … or reboot).

**Do NOT run the coding/build work as root** — see §7.

---

## 6. Definition of done

- Deterministic repro script.
- Bisect identifies which custom flag (expected: `turbo4`).
- Root cause in the HIP dispatch/sync path identified (the specific
  kernel/stream/event-signal bug).
- Fix: correct the kernel dispatch / add the missing synchronization / fix the
  signal race.
- Verify: repro no longer hangs across N runs; tokens/s unchanged; correctness
  unchanged (logits/output match a `q8_0` KV baseline within tolerance).

## 6b. Repro progress 2026-07-05 (narrowed, not yet caught)

Built a repro harness (`scripts/repro_turbo4_hang.py` + `scripts/repro_turbo4_hang.sh`)
and llama-serve configs (`ornith-9b-turbo4-repro.env`, `qwen36-turbo4-repro.env`).
Two key findings that tighten the repro conditions:

1. **The context-checkpoint RESTORE path is hybrid/SWA/recurrent-model ONLY.**
   A dense model (tried Ornith-9B-IQ4_NL first) NEVER triggers a restore — on a
   divergent prompt it just truncates the KV cache token-by-token, so
   `restored context checkpoint` never fires (confirmed: 0 restores over many
   divergent rounds, despite checkpoints being created). The restore path
   exists precisely because hybrid/linear-attention state can't be un-rolled.
   So any repro MUST use the original hang model class — Qwen3.6-27B
   (Qwen3-Next hybrid attention) or similar — not a dense model. This also
   means the bug can only ever bite hybrid-attention models with turbo4 KV.

2. **Forcing a RESTORE (not a full do_reset) is the fiddly part.** With
   Qwen3.6-27B + turbo4 + `--ctx-checkpoints 8 --checkpoint-min-step 2048`,
   the current repro prompt (big shared "alpha" prefix + alternating
   divergent "beta"/"gamma" tail) hits the `do_reset` full-reprocess branch
   (`server-context.cpp:3432`, logs `erased invalidated context checkpoint
   ... pos_next = 0`), NOT the `load_tgt` restore branch (line 3420). Reason:
   the divergence lands past the span the 8 checkpoints cover, OR the
   checkpoint-selection predicate (line 3411 `cur.pos_max > pos_next` /
   line 3414 `cur.pos_min < pos_min_thold`) rejects them. To actually hit the
   restore branch, the NEXT repro attempt should: put the divergence point
   EARLIER (well within the last `n_ctx_checkpoints * checkpoint_min_step`
   window, e.g. diverge ~4-8k tokens from the end of a ~30k prefix), and/or
   raise `--ctx-checkpoints` and lower `--checkpoint-min-step` so a checkpoint
   reliably sits just before the divergence. Verify success by watching for
   `restored context checkpoint` in the server log BEFORE expecting a hang —
   no restore means the trigger condition isn't met and a non-hang says
   nothing.

Not caught this session (repro didn't reach the restore path). Stopped to
prioritize the vLLM-parallel-serving strategic decision. turbo3 remains the
working default, so this stays low-priority. The narrowing above (hybrid-only,
must hit load_tgt) should make the next attempt much faster.

## 6c. Repro progress 2026-07-05 continued — restore path finally triggered, hang still not conclusively caught

Follow-up session, same day. Goal: get the repro to actually hit the
`load_tgt` restore branch (6b left off before that was achieved), then push
toward the original wedge's concurrency (`-np 3` + `--spec-type ngram-cache`).

**Fix to the repro methodology (root cause of 6b's failure):** context
checkpoints are created ONLY at user-message turn boundaries, or near prompt
end (`server-context.cpp:3643-3647`, `is_user_start` / `near_prompt_end`) —
never sprinkled through the middle of one large single-turn prompt. 6b's
single-turn "big prefix + divergent tail" design could therefore only ever
produce ONE checkpoint, at the very end — so a divergence anywhere earlier
always lands *before* the only checkpoint and gets rejected, falling to
`do_reset` (full reprocess) instead of `load_tgt` (restore).

**Fix: multi-turn conversation.** Rewrote `scripts/repro_turbo4_hang.py` (v3)
to use a fixed N-turn prefix (identical every round, so it's cached and each
turn boundary gets its own checkpoint), followed by a large *varying* final
turn each round. This reliably produces multiple checkpoints across the
conversation, so the varying final turn's divergence point has a checkpoint
just before it.

**Result: restores now trigger repeatedly and cleanly.** Confirmed via
`restored context checkpoint` log lines (18+ single-client, 12+ under 3-way
concurrency) — each one a genuine restore-then-large-decode cycle on turbo4
KV, the exact precondition from the original hang. No hang across ~20+ single-
client cycles.

**Escalated to match the original wedge more closely:** added
`--spec-type ngram-cache --spec-draft-n-max 16 --reasoning-budget 256
--cont-batching -b 4096 -ub 4096 --cache-ram 8192` to
`configs/qwen36-turbo4-repro.env`, and ran **3 concurrent clients** (`-np 3`,
matching the original `-np 3`), each an independent tagged conversation so
each lands on its own slot and does its own restore+decode cycles
concurrently — this is the closest reproduction of the original conditions
achieved so far.

**Inconclusive result, session ended before full analysis:** under this 3-way
concurrent load, one client (`C`) reported "DID NOT RETURN within 240s" while
the other two (`A`, `B`) were still succeeding around ~47s/round each. GPU1
utilization read 83% at a nearby check. By the time the situation could be
re-inspected, the server had recovered: `/health` responds `ok`, GPU1 reads
0% (idle, not stuck at load with 0% — the actual hang signature per §2 is
0% GPU + a *permanently pending* request). This is most consistent with
**client C simply exceeding its 240s client-side timeout under heavy 3-way
restore+decode contention** (a real slowdown, not a proven permanent wedge) —
but this was NOT verified live (no `gdb -p`/`sudo dmesg` capture was taken at
the moment of the timeout, which is the decisive evidence per §2/§5). Treat
this run as **inconclusive, not a confirmed repro** — the true hang (server
never recovers, GPU stays at 0% with a request permanently stuck) has still
only been observed the one original time.

**Next steps for whoever picks this back up:**
1. Re-run the same 3-concurrent-client setup
   (`configs/qwen36-turbo4-repro.env` + 3x tagged
   `scripts/repro_turbo4_hang.py --tag A/B/C`), but the INSTANT any client
   reports "DID NOT RETURN", immediately (before touching anything else):
   - `rocm-smi --showuse` (repeat 2-3x over a few seconds — 0% and *staying*
     0% while the request is still pending is the tell; transient dips are
     normal)
   - `gdb -p <server_pid> -batch -ex "thread apply all bt"` and look for the
     `libhsa-runtime64.so.1` / `kfd_wait_on_events` pattern from §2
   - `curl -m 5 http://localhost:<port>/health` — if this also stops
     responding, that's a stronger signal than a single client timing out
   - Only THEN consider it a repro; if GPU% is nonzero or recovers within a
     few more seconds, it was just contention, raise `--timeout` and retry.
2. If genuinely caught: this is finally the point to do the `sudo dmesg` /
   `rocgdb` capture from §5 and the bisect table in §4b (start with variant A:
   `turbo4` → `q8_0`, keeping everything else — spec-decode, checkpoints,
   concurrency — identical, to isolate whether turbo4 itself or the
   spec-decode/checkpoint interaction is the true culprit).
3. Scripts as of this session: `scripts/repro_turbo4_hang.py` (v3, multi-turn,
   `--tag` for concurrent independent clients, distinguishes a fast HTTP 400
   config error from a real timeout-hang), `configs/qwen36-turbo4-repro.env`
   (turbo4 + `-np 3` + ngram-cache spec-decode + 8 checkpoints, matches the
   original wedge's flag set).

turbo3 remains the working production default in the meantime (see
`fractal-production-serving` memory) — this stays a background investigation,
not a blocker.

## 6d. ROOT CAUSE FOUND, 2026-07-06 — cross-stream race in the HIP quantize path

Caught the hang live under the exact 3-concurrent-client setup from §6c
(recalibrated to `--turn-items 220 --final-items 700` to fit the 43776
tok/slot ceiling — the earlier 900/1800 sizes from §6c blew that budget
immediately with a config error, not a hang). Two of three clients timed out
identically; `rocm-smi` showed GPU1 busy (7% -> 100% -> 100%), NOT idle at 0%
as originally assumed in §2 — that earlier "idle GPU" read was misleading.

**`gdb -p <pid> -batch -ex "thread apply all bt"` on the wedged server, the
decisive frame (Thread 1, the ONLY thread running `server_queue::start_loop` /
`update_slots`):**
```
#0-#12  (deep in libamdhip64.so / libhsa-runtime64.so — blocked in a stream wait)
#13 ggml_backend_cuda_buffer_get_tensor(...)       [ggml-cuda.cu:744]
#14 llama_io_write_host::~llama_io_write_host()
#15 llama_context::state_seq_get_data(...)
#16 common_prompt_checkpoint::update_tgt(...)
#17 server_context_impl::create_checkpoint(...)
#18 server_context_impl::pre_decode()::{lambda#5}
#19 server_context_impl::iterate(...)
#20 server_context_impl::pre_decode()
#21 server_context_impl::update_slots()
#22 server_queue::start_loop(...)
```

`ggml-cuda.cu:744` is:
```cpp
static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ...
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));   // <-- stuck here forever
}
```
Because llama-server's decode loop is single-threaded, this one stuck
synchronize freezes every slot on this server, not just the one being
checkpointed — matching all 3 clients (A, B, C) being affected: A and C timed
out outright, and B's two "successful" rounds took 134s and 67s for 7 tokens
(vs. the usual ~5-10s), i.e. it was stalling repeatedly on the same shared
loop, not actually fast.

**Why this specific memcpy hangs — traced to source:**
`quantize_turbo3_0` / `quantize_turbo4_0` in `ggml/src/ggml-turbo-quant.c`
(lines ~363-387, ~670-690) are wired into ggml's generic
`ggml_type_traits.from_float` interface — a plain C signature
`(src, dst, nrows, n_per_row, imatrix)` designed for synchronous CPU
reference quantization, with **no stream parameter in the interface at all**.
The fork's `#ifdef GGML_USE_HIP` fast path in that function calls straight
into `turbo3_0_quantize_hip` / `turbo4_0_quantize_hip`
(`ggml/src/ggml-turbo-quant-hip.hip:428-444`), which launch their kernels via:
```cpp
hipLaunchKernelGGL(turbo4_quantize_kernel_rdna3, grid, block, 0, 0, src, dst, nrows);
//                                                          ^shmem ^stream — hardcoded to legacy stream 0
```
There is no stream to thread through (the interface never had one), and the
wrapper returns immediately without waiting for the kernel to finish, even
though `from_float` is treated everywhere else in ggml as a synchronous,
blocking call. Contrast with upstream `ggml/src/ggml-cuda/quantize.cu`, where
every kernel launch takes an explicit `cudaStream_t stream` argument sourced
from the context (`ctx.stream()`).

So: the turbo-quant KV write happens on legacy stream 0, asynchronously, with
no wait. The checkpoint-save readback happens later on `cudaStreamPerThread`
— a third, independent stream. Nothing anywhere establishes a happens-before
relationship between "turbo4 kernel finished writing this KV block" and
"checkpoint code is about to read it back to host." That is a genuine GPU
command-queue race, not a driver mystery:
- **turbo-specific**: only these hand-rolled kernels bypass `ctx.stream()`;
  every other ggml-cuda op (matmul, attention, standard quant/dequant) is
  properly ordered on the context's real stream.
- **worse on turbo4 than turbo3**: turbo4's larger block (68B vs 50B) means a
  longer-running kernel, widening the unsynchronized race window before a
  checkpoint read can land mid-write or race the completion signal.
- **specifically at checkpoint creation**: that's the only code path that
  reads the KV buffer back to host across a *different* stream than the one
  that wrote it; ordinary decode/attention reads happen on `ctx.stream()`
  itself, which HIP does order correctly relative to same-stream writes.
- **GPU shows nonzero busy% during the hang**: other slots keep executing
  legitimately on `ctx.stream()` while thread 1 sits wedged waiting on
  `cudaStreamPerThread` for a stream-0 kernel whose completion was never
  properly synchronized.

**Proposed fix (small, low-risk, no interface plumbing needed):** since
`from_float` is already assumed synchronous everywhere it's called, just make
the HIP fast path actually synchronous — add a stream synchronize immediately
after each kernel launch in `ggml-turbo-quant-hip.hip`:
```cpp
extern "C" void turbo4_0_quantize_hip(const float * src, void * dst, int nrows) {
    dim3 grid(nrows);
    dim3 block(32);
    hipLaunchKernelGGL(turbo4_quantize_kernel_rdna3, grid, block, 0, 0, src, (uint8_t *)dst, nrows);
    hipStreamSynchronize(0);   // block until the legacy-stream kernel actually finishes
}
```
Same one-line addition in `turbo2_0_quantize_hip` and `turbo3_0_quantize_hip`
for consistency (turbo3 likely has the identical latent race, just with a
narrower window that hasn't been hit in production yet). Cost: one small sync
stall per quantize call, negligible next to decode time; benefit: eliminates
the race entirely by construction, no stream plumbing through the untouched
`from_float` interface required.

**Not yet done:** apply the fix, rebuild, and re-run the same 3-concurrent
repro to confirm no hang across a full 60-round run per client. If confirmed,
also worth timing whether the added sync measurably regresses throughput
(expected: no, since this is off the hot per-token decode path — quantization
into KV only happens once per new token written, and the kernel itself is
already fast).

## 6e. FIX CONFIRMED, 2026-07-06

Applied the `hipStreamSynchronize(0)` fix to all three wrappers
(`turbo2_0_quantize_hip`, `turbo3_0_quantize_hip`, `turbo4_0_quantize_hip` in
`ggml-turbo-quant-hip.hip`), rebuilt (`cmake --build . --target llama-server`,
needed `ninja`, `ccache`, and the `rocwmma` package reinstalled first - all
three were unexpectedly missing from the system, unrelated to this bug),
restarted the turbo4-repro server on the new binary, and re-ran the identical
3-concurrent-client repro (`--tag A/B/C`, `--turn-items 220 --final-items 700`
recalibrated to fit the 43776 tok/slot ceiling under `-np 3`).

First attempt at 200s timeout: 2 of 3 clients "timed out." Live `gdb` at that
moment showed Thread 1 in `ggml_backend_cuda_synchronize` (a normal, frequent
per-token wait for the compute stream, called from `common_sampler_sample` /
`post_decode` - NOT the checkpoint-creation frame from before) - and the
client logs showed all 3 still actively completing rounds throughout, just
slowly. So this was 3-way GPU contention exceeding an overly tight timeout,
not a hang. Confirmed by re-running with `--timeout 400`:

**Full 30-minute run, zero hangs:** client A completed 14 rounds, B 11
rounds, C 16 rounds, every single one "ok" - no timeouts, no crashes, no
runaway slowdown (round times stayed in a stable ~90-160s band under 3-way
contention the whole run, consistent with genuine GPU contention rather than
progressive wedging). This is the same flag combination
(turbo4 + `-np 3` + ngram-cache spec-decode + 8 checkpoints + `--kv-offload`)
that produced the original hang and the confirmed §6d repro.

**Status: fixed.** The cross-stream race in the turbo-quant HIP kernel launch
path (legacy stream 0, no synchronize, racing against the checkpoint
readback's `cudaStreamPerThread`) was the root cause. turbo4 is no longer
believed to hang under concurrent load + checkpointing.

**Follow-up recommended before promoting turbo4 to production:**
1. A longer soak (multi-hour, higher round count) to build more confidence
   beyond this 30-minute window.
2. Note the caveat: 3-way concurrency with large (~28k token) checkpointed
   contexts is inherently slow on this hardware regardless of the bug (one
   GPU, one decode loop, `-np 3` time-slicing) - ~100-160s/round here is
   expected contention cost, not a regression to chase.
3. turbo3 remains the current production default (smaller KV footprint, more
   context headroom) - this fix makes turbo4 *usable*, it doesn't by itself
   make turbo4 preferable to turbo3. Revisit only if a use case specifically
   wants turbo4's tradeoffs.

## 7. Note for the fixing agent

The owncoder agent that hit this already has a client-side mitigation (stream
stall watchdog + timeout + retry) so it no longer hangs forever — but that only
masks the symptom. This bug is the real defect: a wedged GPU dispatch in the
turbo4 path. Fix it at the kernel/sync level.

Build is plain CMake/HIP for RDNA3 — no root needed to compile or edit. Only the
live-GPU diagnostics in §5 need elevation, and those should be run as discrete
`sudo` commands, not by running the whole agent as root.
