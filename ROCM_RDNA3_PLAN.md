# ROCm RDNA 3 Enhancement Plan for RX 7900 XTX

This plan outlines the steps to implement missing ROCm features and optimizations for RDNA 3 (gfx1100) in `llama.cpp`.

## 1. Flash Attention Head Size 256 Support
- **Status:** Done
- **Goal:** Enable the optimized `MMA_F16` Flash Attention kernel for head dimensions (DKQ) of 256.
- **Impact:** Significant speedup for models like Gemma 2.
- **Tasks:**
    - [x] Identify and modify the `DKQ` guard in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`.
    - [x] Ensure `WMMA` instructions for RDNA 3 are correctly utilized in the `DKQ=256` template.
    - [x] Verify wavefront32 compatibility for the expanded tile sizes.

## 2. Enable `hipCUB` Support
- **Status:** To Do
- **Goal:** Utilize `hipCUB` for optimized scan and reduction operations.
- **Impact:** Performance boost for `cumsum` and related ops (Mamba/SSM models).
- **Tasks:**
    - [ ] Modify `ggml/src/ggml-cuda/common.cuh` to define `GGML_CUDA_USE_CUB` for HIP builds.
    - [ ] Update build system (CMake/Makefile) to link against `hipCUB`.
    - [ ] Verify `cumsum.cu` correctly dispatches to `hipCUB`.

## 3. Implement MMQ Kernel for `IQ1_M`
- **Status:** To Do
- **Goal:** Add `IQ1_M` support to the Matrix-Matrix Quantized (MMQ) framework.
- **Impact:** Better performance for extreme quantization without `hipBLAS` fallbacks.
- **Tasks:**
    - [ ] Implement `vec_dot_iq1_m` within the MMQ template.
    - [ ] Update dispatch logic in `mmq.cu`.

## 4. Optimize BF16/mmvf Batching Heuristics
- **Status:** To Do
- **Goal:** Tune batch size thresholds for RDNA 3.
- **Tasks:**
    - [ ] Analyze `mmvf.cu` heuristics for AMD.
    - [ ] Propose and implement improved thresholds for `gfx1100`.

## 5. Improve `hipBLASLt` Integration
- **Status:** To Do
- **Goal:** Enable better kernel fusion using `hipBLASLt`.
- **Tasks:**
    - [ ] Research current `hipBLASLt` limitations in `llama.cpp`.
    - [ ] Implement missing fusion paths for RDNA 3.
