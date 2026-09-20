# ROCm RDNA 3 Enhancement Plan for RX 7900 XTX

This plan outlines the steps to implement missing ROCm features and optimizations for RDNA 3 (gfx1100) in `llama.cpp`.

## 1. Flash Attention Head Size 256 Support
- **Status:** Done
- **Goal:** Enable the optimized `MMA_F16` Flash Attention kernel for head dimensions (DKQ) of 256.
- **Impact:** Significant speedup for models like Gemma 2.

## 2. Enable `hipCUB` Support
- **Status:** Done
- **Goal:** Utilize `hipCUB` for optimized scan and reduction operations.
- **Impact:** Performance boost for `cumsum` and related ops (Mamba/SSM models).

## 3. Implement MMQ Kernel for `IQ1_M`
- **Status:** Done
- **Goal:** Add `IQ1_M` support to the Matrix-Matrix Quantized (MMQ) framework.
- **Impact:** Better performance for extreme quantization without `hipBLAS` fallbacks.

## 4. Optimize BF16/mmvf and MMVQ Heuristics
- **Status:** Done
- **Goal:** Tune batch size and warp configurations for RDNA 3.
- **Tasks:**
    - [x] Analyze `mmvf.cu` heuristics for AMD.
    - [x] Refined BF16 heuristics for RDNA3 and RDNA4 to match F16 structure.
    - [x] Added `IQ1_S` and `IQ1_M` to `MMVQ` (Matrix-Vector Quantized) path.
    - [x] Increased `MMVQ` warp count for `IQ1_S` and `IQ1_M` on RDNA 3/4 to improve GPU utilization.

## 5. Improve `hipBLASLt` Integration
- **Status:** Done
- **Goal:** Enable better kernel fusion using `hipBLASLt`.
- **Tasks:**
    - [x] Research current `hipBLASLt` limitations in `llama.cpp`.
    - [x] Added `hipblaslt` support to CMake and a `GGML_HIP_USE_HIPBLASLT` toggle.
