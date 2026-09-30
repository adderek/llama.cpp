#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// fork: with beta_logit, dst->src[4] is SIGMOID(beta_logit) and the kernel applies the sigmoid itself
// (see the SIGMOID -> GATED_DELTA_NET fusion); dst->src[4] is then never read.
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_tensor * beta_logit = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_tensor * beta_logit = nullptr);
