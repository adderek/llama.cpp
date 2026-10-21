# adderek/llama.cpp — a TurboQuant fork for AMD RDNA3

> [!NOTE]
> **This is [adderek/llama.cpp](https://github.com/adderek/llama.cpp), a fork of
> [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** tuned for AMD Radeon RX 7900 XTX
> (gfx1100, RDNA3) on ROCm 7.2.4 / Linux, and **tested only there**. Upstream is merged in
> regularly (last: 2026-10-21). Details, measurements, environment variables and **which GPUs it
> can run on** (no turbo on Metal / Vulkan / SYCL; CUDA never built): **[FORK.md](FORK.md)**.
> Everything below the upstream title is the upstream README, unchanged.

## Speculative decoding: logprobs you can trust

With speculative decoding, most generated tokens are accepted from a draft rather than sampled
one by one. A client that reads logprobs (confidence scoring, classifiers, evaluation harnesses,
agents that retry on low confidence) must get the same numbers for those tokens as without a
draft, or the probabilities silently change whenever speculation is switched on. The fork has
done this since 2026-10-02 (`b878d0765`); upstream added its own version on 2026-10-20
(#27196). The two differ where it matters:

| | upstream | this fork |
|---|---|---|
| where draft-token probs come from | read back from the verification batch after acceptance | recorded during verification, right after each position is sampled (`on_sample` callback), no extra decode |
| `post_sampling_probs: true` | draft tokens get `prob` 1.0 and an empty top list, while tokens sampled normally in the same response get real ones | supported: the sampler's candidates of that step |
| mismatch between recorded probs and accepted tokens | not checked | checked; the tokens are sent without probs and a warning is logged, never with another token's probs |
| checkpoint-restore replay | — | verifies again and refills the probs |
| cost per token with logprobs on | copies all n_vocab logits and sorts them | two vectorized passes, no copy: 1.1 ms → 45 µs per token; decode cost of logprobs 13–16% → 1.4–2.9% (`188aaff93`) |
| streamed chat with tools + logprobs | rejected with 400 | works; each row rides on the next chunk that goes out (`f79abf326`) |
| `token` / `bytes` of a row | the text emitted at that step: `""` for a stop-word prefix or half a UTF-8 character, and the next row shows both tokens' text | the token's own piece (`eba1a92ae`) |
| per-request `speculative.type` | — | `"none"` turns drafting off for that request; an unknown type returns 400 (`3e4738abf`) |

Tested by `tools/server/tests/unit/test_logprobs_speculative.py` (draft on vs off at
temperature 0: same ids, same top-k, probs within 1e-3) and `test_logprobs_stream_tools.py`;
upstream's own `test_draft_token_probs` passes on the fork as well.

## What this fork has that upstream does not

**KV cache compression**
- **TurboQuant KV cache** (`--cache-type-k/-v turbo2|turbo3|turbo4`): 2.1–4.25 bits per value,
  3.8x smaller than f16 at turbo4. Works on dense, GQA, sparse (QSA, Qwen3.8-Flash-Next), MLA
  and DSA attention; K and V may use different types.
- Correctness fixes on top of it: the QSA path read the cache in the wrong basis and silently
  answered from the wrong part of the context; GQA models with padded heads crashed.

**Speed and precision on RDNA3**
- **FlashAttention kernels for turbo caches**: 2.6–5.4x faster prefill at 16–64k context; decode
  up to 34% faster, on Ornith-35B faster than an f16 cache at long context.
- **Fewer kernel launches per decoded token**: a q8_1 activation cache shared between mat-vecs and
  fused shared-expert gate, residual add + rms_norm, softplus and GDN beta kernels; on
  Ornith-35B-A3B 1622 → 1221 kernels per token (the cache alone +7–8% decode), bit-identical
  output.
- **f32 accumulation in the MMA FlashAttention kernel**: upstream sums P*V in f16 on RDNA3, which
  goes 20–67% wrong over 32k positions; perplexity at 32k improves for 0–3.7% of prefill.
- **No f16 overflow in large matmuls**: Q2_K / Q6_K / IQ2 stay on MMQ instead of an f16-output
  hipBLAS GEMM, and Q4_1 / Q5_1 / Q8_1 multiply block scales in f32 (a 397B model answered only
  `/` before).

**MoE models larger than VRAM**
- `GGML_CUDA_OP_OFFLOAD_DEVICES` picks the GPU that runs the matmuls of experts kept in RAM
  (upstream takes the first): 2x prefill when the first card sits on a slow PCIe link.
- **Models larger than VRAM + RAM** (`LLAMA_MOE_HOT`, `LLAMA_MOE_DIRECT`, see
  `tools/moe-tier/README.md`): experts sorted hottest-first by imatrix counts, hot ones on the
  GPU, cold ones streamed from NVMe with O_DIRECT into a locked RAM arena: 2.4 instead of
  0.9 tokens/s on a 244 GB 397B model. Off unless `LLAMA_MOE_HOT` is set.

**Server and runtime**
- **Speculative decoding with correct logprobs** (above), and speculative parameters settable
  per request (draft model, n-gram sizes, type).
- **`--reasoning auto` does not enable thinking** (upstream does); use `--reasoning on`.
- Logs carry wall-clock time and busy/total slots; a stall watchdog dumps all thread backtraces
  (`LLAMA_STALL_WATCHDOG_SECS`).
- `--hugepages`: model weights on 2 MiB hugetlb pages.

**Robustness on ROCm**
- Fixed a GPU memory fault on prompt-cache restore, a turbo4 decode hang (cross-stream race) and
  an abort under CUDA-graph capture.

**Models and formats**
- TurboQuant weight types `TQ3_1S` / `TQ4_1S` (load and run; inherited from domvox, untested).
- The multimodal loader fails cleanly when the vision encoder's buffers do not fit, instead of
  crashing on the first image.

**Regression tests** for every fork feature run in ctest (`test-turbo-kv-*`, `test-fattn-turbo4`,
`test-moe-tier`, `test-mmvq-src1-cache`, ...), each in its own file.

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
