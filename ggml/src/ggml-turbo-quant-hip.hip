/**
 * @file ggml-turbo-quant-hip.hip
 * @brief HIP implementation of TURBO2_0, TURBO3_0, TURBO4_0 quantization for RDNA3.
 *
 * All three types share the same pipeline per row:
 *   L2-normalize → WHT (s1 signs, butterfly, inv_sqrt*s2 signs) → centroid → pack
 *
 * Block layouts (all assume 128-element rows, i.e. n_per_row == QK_TURBOx):
 *   TURBO2_0: [norm fp16 (2B)] [qs[32] 2-bit packed (32B)] = 34 bytes
 *   TURBO3_0: [norm fp16 (2B)] [qs[32] lower-2-bit (32B)] [signs[16] bit2 (16B)] = 50 bytes
 *   TURBO4_0: [norm fp16 (2B)] [rnorm fp16 (2B, reserved)] [qs[64] nibble (64B)] = 68 bytes
 *
 * TURBO4_0 kernel implements TURBO4_USE_4BIT=1 path only (default).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define TURBO2_D            128
#define TURBO2_QS_BYTES      32   /* 128 * 2-bit / 8 */
#define TURBO2_BLOCK_BYTES   34   /* 2 + 32 */

#define TURBO3_D            128
#define TURBO3_QS_BYTES      32   /* 128/4 */
#define TURBO3_SIGNS_BYTES   16   /* 128/8 */
#define TURBO3_BLOCK_BYTES   50   /* 2 + 32 + 16 */

#define TURBO4_D            128
#define TURBO4_QS_BYTES      64   /* 128 * 4-bit / 8 */
#define TURBO4_BLOCK_BYTES   68   /* 2 + 2 + 64 */

/* WHT sign arrays — must match CPU (turbo_cpu_s1/s2 in ggml-turbo-quant.c, seed=42) */
__device__ __constant__ float d_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

__device__ __constant__ float d_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

/* 2-bit Lloyd-Max centroids for N(0, 1/128) — must match CPU CENTROIDS_2BIT */
__device__ __constant__ float d_centroids2[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

/* 3-bit Lloyd-Max centroids for N(0, 1/128) — must match CPU CENTROIDS_3BIT */
__device__ __constant__ float d_centroids3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* 4-bit Lloyd-Max centroids for N(0, 1/128) — must match CPU CENTROIDS_4BIT */
__device__ __constant__ float d_centroids4[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

__device__ static inline int nearest_centroid_2bit_dev(float val) {
    if (val < -0.086728f) return 0;
    if (val <  0.000000f) return 1;
    if (val <  0.086728f) return 2;
    return 3;
}

__device__ static inline int nearest_centroid_3bit_dev(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

__device__ static inline int nearest_centroid_4bit_dev(float val) {
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/*
 * Shared WHT pipeline: load → L2-normalize → s1 signs → butterfly → s2/sqrt(128).
 * All three kernels (turbo2/3/4) differ only in the quantize+pack step after this.
 * Inlined per-kernel to avoid register spill from __device__ function calls with shared mem.
 */

/**
 * One block per input row, 32 threads (one RDNA3 wavefront) per block.
 * Quantizes 128 floats into one block_turbo2_0 (34 bytes).
 * Requires n_per_row == 128 (one WHT group per row).
 */
__global__ void turbo2_quantize_kernel_rdna3(
        const float * __restrict__ src,
        uint8_t     * __restrict__ dst,
        int nrows) {

    const int row = hipBlockIdx_x;
    const int tid = hipThreadIdx_x;  /* 0..31 */
    if (row >= nrows) return;

    __shared__ float buf[TURBO2_D];

    const float * row_in = src + (size_t)row * TURBO2_D;
    buf[tid]      = row_in[tid];
    buf[tid + 32] = row_in[tid + 32];
    buf[tid + 64] = row_in[tid + 64];
    buf[tid + 96] = row_in[tid + 96];
    __syncwarp();

    float norm_sq = buf[tid]      * buf[tid]
                  + buf[tid + 32] * buf[tid + 32]
                  + buf[tid + 64] * buf[tid + 64]
                  + buf[tid + 96] * buf[tid + 96];
    for (int mask = 16; mask > 0; mask >>= 1)
        norm_sq += __shfl_xor(norm_sq, mask);
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    buf[tid]      *= inv_norm;
    buf[tid + 32] *= inv_norm;
    buf[tid + 64] *= inv_norm;
    buf[tid + 96] *= inv_norm;
    __syncwarp();

    buf[tid]      *= d_s1[tid];
    buf[tid + 32] *= d_s1[tid + 32];
    buf[tid + 64] *= d_s1[tid + 64];
    buf[tid + 96] *= d_s1[tid + 96];
    __syncwarp();

    for (int h = 1; h < TURBO2_D; h <<= 1) {
        for (int p = tid; p < 64; p += 32) {
            int i = (p / h) * (2 * h) + (p % h);
            int j = i + h;
            float a = buf[i], b = buf[j];
            buf[i] = a + b;
            buf[j] = a - b;
        }
        __syncwarp();
    }

    const float inv_sqrt128 = 0.08838834764831845f;
    buf[tid]      *= inv_sqrt128 * d_s2[tid];
    buf[tid + 32] *= inv_sqrt128 * d_s2[tid + 32];
    buf[tid + 64] *= inv_sqrt128 * d_s2[tid + 64];
    buf[tid + 96] *= inv_sqrt128 * d_s2[tid + 96];
    __syncwarp();

    /* Thread tid quantizes elements [tid*4 .. tid*4+3] into 1 qs byte (2 bits each) */
    const float v0 = buf[tid * 4 + 0], v1 = buf[tid * 4 + 1];
    const float v2 = buf[tid * 4 + 2], v3 = buf[tid * 4 + 3];
    const int   i0 = nearest_centroid_2bit_dev(v0);
    const int   i1 = nearest_centroid_2bit_dev(v1);
    const int   i2 = nearest_centroid_2bit_dev(v2);
    const int   i3 = nearest_centroid_2bit_dev(v3);
    const uint8_t my_qs = (uint8_t)((i0) | (i1 << 2) | (i2 << 4) | (i3 << 6));
    float recon_sq_partial = d_centroids2[i0] * d_centroids2[i0]
                           + d_centroids2[i1] * d_centroids2[i1]
                           + d_centroids2[i2] * d_centroids2[i2]
                           + d_centroids2[i3] * d_centroids2[i3];

    float recon_sq = recon_sq_partial;
    for (int mask = 16; mask > 0; mask >>= 1)
        recon_sq += __shfl_xor(recon_sq, mask);
    const float recon_norm = sqrtf(recon_sq);
    const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    /* Write block_turbo2_0: norm(2B) + qs[32] */
    uint8_t * blk = dst + (size_t)row * TURBO2_BLOCK_BYTES;
    if (tid == 0) {
        __half h16 = __float2half(corrected);
        *((uint16_t *)blk) = __half_as_ushort(h16);
    }
    blk[2 + tid] = my_qs;
}

/**
 * One block per input row, 32 threads (one RDNA3 wavefront) per block.
 * Quantizes 128 floats into one block_turbo3_0.
 */
__global__ void turbo3_quantize_kernel_rdna3(
        const float * __restrict__ src,
        uint8_t     * __restrict__ dst,
        int nrows) {

    const int row = hipBlockIdx_x;
    const int tid = hipThreadIdx_x;  /* 0..31 */
    if (row >= nrows) return;

    __shared__ float buf[TURBO3_D];

    /* Coalesced load: thread tid owns indices tid, tid+32, tid+64, tid+96 */
    const float * row_in = src + (size_t)row * TURBO3_D;
    buf[tid]      = row_in[tid];
    buf[tid + 32] = row_in[tid + 32];
    buf[tid + 64] = row_in[tid + 64];
    buf[tid + 96] = row_in[tid + 96];
    __syncwarp();

    /* 1. L2 norm over all 128 elements via warp reduction */
    float norm_sq = buf[tid]      * buf[tid]
                  + buf[tid + 32] * buf[tid + 32]
                  + buf[tid + 64] * buf[tid + 64]
                  + buf[tid + 96] * buf[tid + 96];
    for (int mask = 16; mask > 0; mask >>= 1)
        norm_sq += __shfl_xor(norm_sq, mask);
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    /* 2. Normalize in-place */
    buf[tid]      *= inv_norm;
    buf[tid + 32] *= inv_norm;
    buf[tid + 64] *= inv_norm;
    buf[tid + 96] *= inv_norm;
    __syncwarp();

    /* 3. WHT — apply s1 signs */
    buf[tid]      *= d_s1[tid];
    buf[tid + 32] *= d_s1[tid + 32];
    buf[tid + 64] *= d_s1[tid + 64];
    buf[tid + 96] *= d_s1[tid + 96];
    __syncwarp();

    /* 4. WHT — 7-stage butterfly (size 128) */
    for (int h = 1; h < TURBO3_D; h <<= 1) {
        /*
         * 64 pairs per stage; thread tid handles pair tid and pair tid+32.
         * Pair p maps to indices: i = (p/h)*2h + (p%h), j = i+h.
         * No two pairs within a stage share an index, so no write conflicts.
         */
        for (int p = tid; p < 64; p += 32) {
            int i = (p / h) * (2 * h) + (p % h);
            int j = i + h;
            float a = buf[i], b = buf[j];
            buf[i] = a + b;
            buf[j] = a - b;
        }
        __syncwarp();
    }

    /* 5. WHT — normalize and apply s2 signs */
    const float inv_sqrt128 = 0.08838834764831845f;  /* 1/sqrt(128) */
    buf[tid]      *= inv_sqrt128 * d_s2[tid];
    buf[tid + 32] *= inv_sqrt128 * d_s2[tid + 32];
    buf[tid + 64] *= inv_sqrt128 * d_s2[tid + 64];
    buf[tid + 96] *= inv_sqrt128 * d_s2[tid + 96];
    __syncwarp();

    /*
     * 6. Quantize: thread tid handles 4 consecutive elements [tid*4 .. tid*4+3].
     *    These 4 elements pack into exactly one qs byte (byte index = tid).
     *    Sign bits for those 4 elements go into half of signs byte tid/2.
     */
    uint8_t my_qs = 0;
    uint8_t my_sign_nibble = 0;
    float   recon_sq_partial = 0.0f;

    for (int k = 0; k < 4; k++) {
        float v   = buf[tid * 4 + k];
        int   idx = nearest_centroid_3bit_dev(v);
        my_qs |= (uint8_t)((idx & 0x3) << (k * 2));
        if (idx & 0x4) my_sign_nibble |= (uint8_t)(1 << k);
        float c = d_centroids3[idx];
        recon_sq_partial += c * c;
    }

    /* 7. Warp-reduce reconstruction norm squared */
    float recon_sq = recon_sq_partial;
    for (int mask = 16; mask > 0; mask >>= 1)
        recon_sq += __shfl_xor(recon_sq, mask);

    /* 8. Corrected norm = grp_norm / recon_norm (matches CPU quantize_row_turbo3_0_ref) */
    const float recon_norm = sqrtf(recon_sq);
    const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    /* 9. Write block_turbo3_0 output */
    uint8_t * blk = dst + (size_t)row * TURBO3_BLOCK_BYTES;

    /* norm (fp16) — thread 0 only */
    if (tid == 0) {
        __half h16 = __float2half(corrected);
        *((uint16_t *)blk) = __half_as_ushort(h16);
    }

    /* qs[32]: thread tid writes byte tid — no conflicts */
    blk[2 + tid] = my_qs;

    /*
     * signs[16]: threads 2k and 2k+1 both contribute to signs byte k.
     *   Thread 2k  holds bits 0..3 (positions 0..3 in the byte).
     *   Thread 2k+1 holds bits 0..3, must shift to positions 4..7.
     * Merge via shfl_xor(lane^1) then even thread writes.
     */
    const int    bit_shift     = (tid & 1) * 4;
    const uint8_t my_shifted   = my_sign_nibble << bit_shift;
    const uint8_t partner      = (uint8_t)__shfl_xor((int)my_shifted, 1);
    const uint8_t combined     = my_shifted | partner;
    if ((tid & 1) == 0) {
        blk[2 + TURBO3_QS_BYTES + (tid >> 1)] = combined;
    }
}

/**
 * One block per input row, 32 threads (one RDNA3 wavefront) per block.
 * Quantizes 128 floats into one block_turbo4_0 (68 bytes, TURBO4_USE_4BIT=1 layout).
 * Requires n_per_row == 128 (one WHT group per row).
 */
__global__ void turbo4_quantize_kernel_rdna3(
        const float * __restrict__ src,
        uint8_t     * __restrict__ dst,
        int nrows) {

    const int row = hipBlockIdx_x;
    const int tid = hipThreadIdx_x;  /* 0..31 */
    if (row >= nrows) return;

    __shared__ float buf[TURBO4_D];

    const float * row_in = src + (size_t)row * TURBO4_D;
    buf[tid]      = row_in[tid];
    buf[tid + 32] = row_in[tid + 32];
    buf[tid + 64] = row_in[tid + 64];
    buf[tid + 96] = row_in[tid + 96];
    __syncwarp();

    float norm_sq = buf[tid]      * buf[tid]
                  + buf[tid + 32] * buf[tid + 32]
                  + buf[tid + 64] * buf[tid + 64]
                  + buf[tid + 96] * buf[tid + 96];
    for (int mask = 16; mask > 0; mask >>= 1)
        norm_sq += __shfl_xor(norm_sq, mask);
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    buf[tid]      *= inv_norm;
    buf[tid + 32] *= inv_norm;
    buf[tid + 64] *= inv_norm;
    buf[tid + 96] *= inv_norm;
    __syncwarp();

    buf[tid]      *= d_s1[tid];
    buf[tid + 32] *= d_s1[tid + 32];
    buf[tid + 64] *= d_s1[tid + 64];
    buf[tid + 96] *= d_s1[tid + 96];
    __syncwarp();

    for (int h = 1; h < TURBO4_D; h <<= 1) {
        for (int p = tid; p < 64; p += 32) {
            int i = (p / h) * (2 * h) + (p % h);
            int j = i + h;
            float a = buf[i], b = buf[j];
            buf[i] = a + b;
            buf[j] = a - b;
        }
        __syncwarp();
    }

    const float inv_sqrt128 = 0.08838834764831845f;
    buf[tid]      *= inv_sqrt128 * d_s2[tid];
    buf[tid + 32] *= inv_sqrt128 * d_s2[tid + 32];
    buf[tid + 64] *= inv_sqrt128 * d_s2[tid + 64];
    buf[tid + 96] *= inv_sqrt128 * d_s2[tid + 96];
    __syncwarp();

    /*
     * Thread tid quantizes elements [tid*4 .. tid*4+3] into 2 qs bytes (nibble packed).
     * element tid*4+0 → byte tid*2+0, bits 0-3
     * element tid*4+1 → byte tid*2+0, bits 4-7
     * element tid*4+2 → byte tid*2+1, bits 0-3
     * element tid*4+3 → byte tid*2+1, bits 4-7
     */
    const float v0 = buf[tid * 4 + 0], v1 = buf[tid * 4 + 1];
    const float v2 = buf[tid * 4 + 2], v3 = buf[tid * 4 + 3];
    const int   i0 = nearest_centroid_4bit_dev(v0);
    const int   i1 = nearest_centroid_4bit_dev(v1);
    const int   i2 = nearest_centroid_4bit_dev(v2);
    const int   i3 = nearest_centroid_4bit_dev(v3);
    const uint8_t qs_byte0 = (uint8_t)((i0 & 0xF) | ((i1 & 0xF) << 4));
    const uint8_t qs_byte1 = (uint8_t)((i2 & 0xF) | ((i3 & 0xF) << 4));
    float recon_sq_partial = d_centroids4[i0] * d_centroids4[i0]
                           + d_centroids4[i1] * d_centroids4[i1]
                           + d_centroids4[i2] * d_centroids4[i2]
                           + d_centroids4[i3] * d_centroids4[i3];

    float recon_sq = recon_sq_partial;
    for (int mask = 16; mask > 0; mask >>= 1)
        recon_sq += __shfl_xor(recon_sq, mask);
    const float recon_norm = sqrtf(recon_sq);
    const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    /* Write block_turbo4_0: norm(2B) + rnorm(2B,0) + qs[64] */
    uint8_t * blk = dst + (size_t)row * TURBO4_BLOCK_BYTES;
    if (tid == 0) {
        __half h16 = __float2half(corrected);
        *((uint16_t *)blk)       = __half_as_ushort(h16);
        *((uint16_t *)(blk + 2)) = 0;  /* rnorm reserved in 4-bit mode */
    }
    blk[4 + tid * 2]     = qs_byte0;
    blk[4 + tid * 2 + 1] = qs_byte1;
}

extern "C" void turbo2_0_quantize_hip(const float * src, void * dst, int nrows) {
    dim3 grid(nrows);
    dim3 block(32);  /* one RDNA3 wavefront */
    hipLaunchKernelGGL(turbo2_quantize_kernel_rdna3, grid, block, 0, 0,
                       src, (uint8_t *)dst, nrows);
}

extern "C" void turbo3_0_quantize_hip(const float * src, void * dst, int nrows) {
    dim3 grid(nrows);
    dim3 block(32);  /* one RDNA3 wavefront */
    hipLaunchKernelGGL(turbo3_quantize_kernel_rdna3, grid, block, 0, 0,
                       src, (uint8_t *)dst, nrows);
}

extern "C" void turbo4_0_quantize_hip(const float * src, void * dst, int nrows) {
    dim3 grid(nrows);
    dim3 block(32);  /* one RDNA3 wavefront — TURBO4_USE_4BIT=1 layout */
    hipLaunchKernelGGL(turbo4_quantize_kernel_rdna3, grid, block, 0, 0,
                       src, (uint8_t *)dst, nrows);
}

/* Keep old name as alias so the call site in ggml-turbo-quant.c continues to compile
   until it is updated to call turbo3_0_quantize_hip directly. */
extern "C" void turbo_bgs_hip_wrapper(const float * global_block, float * global_out, int d, int batch_size) {
    (void)d;  /* always TURBO3_D=128 */
    turbo3_0_quantize_hip(global_block, global_out, batch_size);
}
