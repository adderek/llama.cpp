#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// fork: q8_1 src1 cache shared by MMVQ and the fused rms_norm (norm.cu). find returns the cached
// quantization of src1 from this graph evaluation or nullptr; reserve claims a slot for src1 and
// returns its buffer (the caller must fill it on this stream before anything reads it) or nullptr.
bool         ggml_cuda_mmvq_src1_cache_enabled();
size_t       ggml_cuda_mmvq_q8_1_size(const ggml_tensor * src1);
const void * ggml_cuda_mmvq_src1_cache_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, cudaStream_t stream);
void *       ggml_cuda_mmvq_src1_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, cudaStream_t stream);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
