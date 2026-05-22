/**
 * @file ggml-turbo-quant-hip.hip
 * @brief High-performance HIP implementation of Block-Gram-Schmidt for RDNA3.
 * 
 * Target: AMD RDNA3 (RX 7000 series)
 * Optimization: LDS Bank Conflict Avoidance, Wavefront-level Parallelism, FMA.
 */

#include <hip/hip_runtime.h>
#include <math.h>

// Stałe dopasowane do architektury RDNA3 (Wavefront size = 32)
#define B_SIZE 32
#define D_SIZE 128
#define L_PAD 33 // Padding dla uniknięcia bank conflicts w LDS (stride 33)
#define EPSILON 1e-10f

/**
 * @brief Core Kernel: Parallel Block-Gram-Schmidt
 * 
 * @param global_block Input matrix V (B x D) in VRAM
 * @param global_out   Output matrix V' (B x D) in VRAM
 * @param d            Dimension of vector (D)
 */
__global__ void turbo_bgs_kernel_rdna3(const float* __restrict__ global_block, 
                                       float* __restrict__ global_out, 
                                       int d) {
    // 1. Deklaracja pamięci lokalnej (LDS)
    // lds_block: [32][128] -> 16 KB
    __shared__ float lds_block[B_SIZE * D_SIZE];
    // lds_gram: [32][33] -> 4 KB (z paddingiem dla bank conflicts)
    __shared__ float lds_gram[B_SIZE * L_PAD];
    // lds_out: [32][128] -> 16 KB (do transformacji wektorów)
    __shared__ float lds_out[B_SIZE * D_SIZE];

    const int tid = hipThreadIdx_x; // ID wątku wewnątrz wavefrontu (0-31)

    // --- FAZA 1: Ładowanie danych i obliczanie Macierzy Grama ---
    
    // Coalesced Load: Każdy wątek ładuje swój wektor do LDS
    for (int i = tid; i < d; i += B_SIZE) {
        lds_block[tid * d + i] = global_block[tid * d + i];
    }
    __syncwarp(); 

    // Obliczanie G[i][j] = V[i] . V[j]
    // Każdy wątek i oblicza swój wiersz w macierzy Grama
    for (int j = 0; j < B_SIZE; ++j) {
        float dot = 0.0f;
        // Optymalizacja: Wykorzystanie FMA (Fused Multiply-Add) w pętli
        #pragma unroll
        for (int k = 0; k < d; ++k) {
            dot += lds_block[tid * d + k] * lds_block[j * d + k];
        }
        lds_gram[tid * L_PAD + j] = dot;
    }
    __syncwarp(); // Wszystkie elementy G muszą być gotowe w LDS

    // --- FAZA 2: Parallel Cholesky Decomposition (In-place w LDS) ---
    // Rozwiązujemy G = LL^T

    for (int i = 0; i < B_SIZE; ++i) {
        // Krok 2.1: Oblicz element przekątnej L[i][i]
        if (tid == i) {
            float sum_sq = 0.0f;
            for (int k = 0; k < i; ++k) {
                float L_ik = lds_gram[i * L_PAD + k];
                sum_sq += L_ik * L_ik;
            }
            float G_ii = lds_gram[i * L_PAD + i];
            lds_gram[i * L_PAD + i] = sqrtf(fmaxf(G_ii - sum_sq, EPSILON));
        }
        
        // Synchronizacja wavefrontu: wszystkie wątki muszą widzieć L[i][i]
        __syncwarp();

        // Krok 2.2: Oblicz elementy pod przekątną L[j][i] dla j > i
        if (tid > i) {
            float sum_prod = 0.0f;
            for (int k = 0; k < i; ++k) {
                // L[j][k] i L[i][k] są już obliczone w poprzednich iteracjach
                sum_prod += lds_gram[tid * L_PAD + k] * lds_gram[i * L_PAD + k];
            }
            float L_ii = lds_gram[i * L_PAD + i];
            lds_gram[tid * L_PAD + i] = (lds_gram[tid * L_PAD + i] - sum_prod) / L_ii;
        }
        
        // Synchronizacja: kolumna i musi być gotowa przed przejściem do i+1
        __syncwarp();
    }

    // --- FAZA 3: Transformacja Wektorów (V' = V - sum_{j<i} L_ij * V'_j) ---
    
    // Kopiujemy wektory wejściowe do lds_out, aby nie niszczyć lds_block (używanego w Fazie 1/2)
    for (int i = tid; i < d; i += B_SIZE) {
        lds_out[tid * d + i] = lds_block[tid * d + i];
    }
    __syncwarp();

    for (int i = 0; i < B_SIZE; ++i) {
        // Każdy wątek i wykonuje projekcję dla swojego wektora
        for (int j = 0; j < i; ++j) {
            float L_ij = lds_gram[i * L_PAD + j];
            // Wektor V'_j jest już zortogonalizowany i znajduje się w lds_out
            #pragma unroll
            for (int k = 0; k < d; ++k) {
                lds_out[i * d + k] -= L_ij * lds_out[j * d + k];
            }
        }
        
        // Normalizacja wektora V'_i
        float norm_sq = 0.0f;
        for (int k = 0; k < d; ++k) {
            norm_sq += lds_out[i * d + k] * lds_out[i * d + k];
        }
        float inv_norm = 1.0f / sqrtf(fmaxf(norm_sq, EPSILON));
        for (int k = 0; k < d; ++k) {
            lds_out[i * d + k] *= inv_norm;
        }
        
        // Synchronizacja: wektor V'_i musi być gotowy dla kolejnych wektorów (j > i)
        __syncwarp();
    }

    // --- FAZA 4: Zapis do pamięci globalnej ---
    for (int i = tid; i < d; i += B_SIZE) {
        global_out[tid * d + i] = lds_out[tid * d + i];
    }
}

/**
 * @brief C-compatible wrapper for HIP kernel.
 * 
 * @param global_block Pointer to input matrix in VRAM
 * @param global_out   Pointer to output matrix in VRAM
 * @param d            Vector dimension
 * @param batch_size   Number of blocks (each block is B_SIZE vectors)
 */
extern "C" void turbo_bgs_hip_wrapper(const float* global_block, float* global_out, int d, int batch_size) {
    // Grid: jeden blok na każdy zestaw 32 wektorów
    dim3 grid(batch_size); 
    // Block: jeden wavefront (32 wątki)
    dim3 block(B_SIZE); 

    hipLaunchKernelGGL(turbo_bgs_kernel_rdna3, grid, block, 0, 0, global_block, global_out, d);
}
