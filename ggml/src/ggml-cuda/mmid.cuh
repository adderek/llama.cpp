#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// zero the dst rows of a partial mul_mat_id, i.e. the rows whose id is negative
void ggml_cuda_mm_ids_zero_skipped(
        const int32_t * ids, float * dst, int n_expert_used, int n_tokens, int ncols_dst,
        int si1, int sd1, int sd2, cudaStream_t stream);
