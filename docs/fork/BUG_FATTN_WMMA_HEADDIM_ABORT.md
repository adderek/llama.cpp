# Bug: `ggml_cuda_flash_attn_ext_wmma_f16` hard-aborts on unsupported head dim instead of falling back

## Severity
High for any model whose attention head dimension falls outside the WMMA
kernel's hardcoded support list combined with the conditions below — crashes
the whole server on the very first warmup decode, not a graceful degradation.

## Symptom
`llama-server` crashes immediately during model warmup with:
```
ggml/src/ggml-cuda/fattn-wmma-f16.cu:597: fatal error
```
and a full process abort (SIGABRT), taking the server down. Confirmed with
`mistralai_Mistral-Small-4-119B-2603` (bartowski IQ3_XXS quant), `-fa on`,
`--split-mode layer --tensor-split 1,1`, `--fit on --fit-target 3072,1024`,
2x RX 7900 XTX (gfx1100).

## Root cause
`ggml_cuda_flash_attn_ext_wmma_f16()` in `ggml/src/ggml-cuda/fattn-wmma-f16.cu`
takes the high-precision path (`prec != GGML_PREC_DEFAULT`) when
`Q->ne[1] <= 32 || Q->ne[0] > 128` — true during warmup, when only a handful
of tokens are being processed (`Q->ne[1]` small). Inside that branch it
`switch`es on `Q->ne[0]` (the attention head dimension) with explicit cases
only for `64, 80, 96, 112, 128, 256`:

```cpp
switch (Q->ne[0]) {
    case 64: ...
    case 80: ...
    case 96: ...
    case 112: ...
    case 128: ...
    case 256: ...
    default:
        GGML_ABORT("fatal error");
        break;
}
```

Mistral Small 4's text config reports `head_dim: 128` (confirmed via
`config.json`), which IS one of the supported cases — so the exact tensor
shape reaching this switch at crash time needs further investigation (may be
a different intermediate tensor than the top-level head dim, or an
interaction with this fork's `--fit`/layer-split placement). Have not yet
bisected further than confirming `-fa off` avoids the crash entirely by
skipping this code path.

## Impact beyond the crash
Working around it via `-fa off` has a side effect in this fork: quantized KV
cache requires flash attention (`llama_init_from_model: V cache quantization
requires flash_attn`), so `-fa off` also forces KV cache back to `f16`,
costing extra VRAM for the same context length compared to `q8_0`.

## Fix suggestion
Either extend the WMMA kernel's supported head-dim list to cover whatever
value is actually reaching this switch for this model, or — safer as a
stopgap — replace the `GGML_ABORT` default case with a fallback to the
generic (non-WMMA) flash-attention path instead of a hard crash, so an
unsupported shape degrades gracefully (slower) instead of taking the server
down.

## Files
- `ggml/src/ggml-cuda/fattn-wmma-f16.cu:597` (the abort site)
- `~/src/llama-serve/configs/mistral-small-4-119b.env` (workaround: `-fa off`
  + `f16` KV cache, documented inline)
