# Bug: TurboQuant query rotation is applied in the `build_attn` wrapper, so any model that calls `build_attn_mha` directly reads the KV cache in the wrong basis

## Severity

**High, and silent.** No abort, no warning, no assert, no measurable slowdown.
The model loads, runs at normal speed and emits fluent, grammatical text — with
facts taken from the wrong place in the context. Every coarse health signal
(exit code, log scan, throughput, VRAM, output-vs-baseline diff) reports
success. Only a test that asks for a specific fact planted in the prompt
catches it.

## Who is affected

The bug lives at the intersection of two features that are developed
separately:

| tree | TurboQuant | `build_attn_qsa` (qwen4exp) | affected |
|---|---|---|---|
| ggml-org/llama.cpp master | no | no | no |
| unslothai/llama.cpp master | no | yes | no |
| domvox/llama.cpp-turboquant-hip | **yes** | not yet | **not yet — see below** |
| this fork (both merged) | yes | yes | yes, fixed in `ce5ba4ac9` |

domvox owns TurboQuant but has not taken qwen4exp yet, so the bug cannot
manifest there today. It will the moment that fork rebases onto an upstream
carrying qwen4exp (ggml-org/llama.cpp#27742). This report is therefore a
forward-looking one, plus a structural hazard that already exists.

## Mechanism

TurboQuant stores K in the KV cache WHT-rotated and never un-rotates it on
read — `dequantize_row_turbo4_0` says so explicitly:

> No inverse WHT, dequant stays in the rotated domain. Q is WHT-rotated by the
> graph, so `<Q_rot, K_rot>` gives correct attention scores. The inverse WHT is
> applied to the attention output via `GGML_OP_TURBO_WHT` (direction=1) in the
> graph.

That contract has two halves and they live in different places:

* **forward rotation of Q** — in `llm_graph_context::build_attn`
  (`src/llama-graph.cpp`, around the `k->type == GGML_TYPE_TURBO*` guard)
* **inverse rotation of the output** — inside `llm_graph_context::build_attn_mha`,
  keyed on `v->type`

A model that calls `build_attn_mha` **directly** gets the second half and not
the first. `<Q, K>` is then computed across two different bases: the scores are
meaningless, but every tensor shape agrees, so nothing fails. The inverse WHT
on the output then faithfully un-rotates a garbage-weighted sum of V, which is
exactly why the result reads as ordinary prose rather than noise.

There is a second trap layered on top. `build_attn_qsa` *does* contain rotation
code:

```c
if (inp->self_k_rot) {
    q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
    k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
}
```

but that is the **upstream Hadamard** mechanism, and the KV cache turns it off
when TurboQuant is active:

```
llama_kv_cache: upstream attention rotation disabled (TurboQuant uses kernel-level WHT)
```

So the function looks like it handles rotation. Under turbo, the branch is
dead and the TurboQuant equivalent was never added.

## Reproduction

Model: Qwen3.8-Flash-Next UD-IQ4_XS (arch `qwen4exp`). Prompt: ~3246 tokens of
filler with one planted fact (`The secret passphrase is HORIZON-4471.`) at
roughly 60% depth, then `Question: what is the secret passphrase?`.
`--temp 0 --seed 42`.

```sh
llama-cli -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -ngl 99 --split-mode layer --main-gpu 1 --tensor-split 0.92,1.0 \
  -ncmoe 36 --cache-type-k turbo4 --cache-type-v turbo4 -fa on \
  -b 4096 -ub 4096 -c 65536 -n 64 --temp 0 --seed 42 -st -f needle.txt
```

| config | answer |
|---|---|
| `--cache-type-* turbo4` | `The answer is **FOXTROT**.` — a filler word from the prompt |
| `--cache-type-* f16` | `The secret passphrase is **HORIZON-4471**.` |

The wrong answer is **byte-identical** on CPU, on a single card, and on a
two-card layer split, at 8192 / 16384 / 65536 context. That determinism is the
signature of a missing graph node rather than a numerical or race problem.

## Fix

Mirror the `build_attn` block in `build_attn_qsa`, immediately before the
`build_attn_mha` call, after `q`/`k`/`v` are taken from the cache:

```c
if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
    if (q->ne[0] % 128 != 0) {
        const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
        q = ggml_pad(ctx0, q, pad, 0, 0, 0);
    }
    if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx0, q); }
    ggml_tensor * innerq_scale = mctx_cur->get_turbo_innerq_scale_inv();
    q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);  // 0 = forward, 0 = auto group size
}
```

Guard on `k->type`, not `v->type`: K determines the rotation, and the two can
differ under MLA where V is a view of K with a different `ne[0]`. For a
non-turbo cache the block is a no-op, so the f16 graph is unchanged.

After the fix, turbo4 answers correctly on one card and on the production
two-card split; f16 is unchanged; and `ornith15-9B` and `qwen3.8-27B` on turbo4
produce output byte-identical to the pre-fix build (the dense path is
untouched).

Payoff for this architecture: main KV drops **1536 -> 408 MiB at 65536 context**
(3.76x). The indexer KV cache stays on f16 on purpose — its keys are read back
with `GET_ROWS`, which has no turbo support; see the guard in
`llama-memory-hybrid-idx.cpp`.

## The structural hazard (this part applies to domvox today)

The one-line fix removes one instance. The shape of the problem remains: **the
TurboQuant query rotation is opt-in by calling the right wrapper**, and nothing
enforces it. Direct callers of `build_attn_mha` outside `llama-graph.cpp` in
this tree:

| file | lines | TurboQuant handling |
|---|---|---|
| `src/models/qwen4exp.cpp` | 728 | added by `ce5ba4ac9` |
| `src/models/deepseek4.cpp` | 757, 812, 848 | **none — zero `TURBO` references in the file** |

`deepseek4.cpp` uses only the upstream Hadamard path (`llama_mul_mat_hadamard`
with `k_rot`) — the mechanism TurboQuant disables. Whether its `k_all`
(a `ggml_concat` of a cache view and a CSA tensor) actually reaches
`build_attn_mha` in a turbo basis is **unaudited**: no deepseek4 weights were
available here to test it, and this is not a claim that it is broken.

Two suggestions, in order of how much they would have helped:

1. **Assert instead of silently mis-rotating.** `build_attn_mha` already
   inspects `v->type` to decide the inverse WHT. At that point it could also
   check that a turbo K arrives with a Q that has been through
   `GGML_OP_TURBO_WHT`, and abort if not. A loud failure at graph build is
   strictly better than fluent wrong answers — the whole cost of this bug was
   that it produced neither an abort nor a warning.
2. **Move the rotation into `build_attn_mha`**, so it cannot be bypassed by
   construction, and make the wrappers stop doing it. This is the larger change
   and touches the dense path, so it wants its own measurement pass.

## Why this went unnoticed

Recorded because the failure mode, not the fix, is the interesting part. Every
check below passed while the bug was live:

| check | verdict | why it missed |
|---|---|---|
| exit code, abort/assert scan | pass | nothing failed |
| output diff vs the pre-merge build | pass | **both builds had the bug** |
| KV cache size, VRAM | pass | measures memory, not correctness |
| short generation (`The capital of France is` -> `Paris`) | pass | world knowledge, not retrieval |
| throughput | pass | within noise |
| **planted-needle retrieval in a long prompt** | **fail** | first check that asked for a fact from the context |

A regression suite for a quantized KV cache needs at least one long-context
retrieval probe. Comparing output against a baseline is not enough when the
baseline shares the defect.
