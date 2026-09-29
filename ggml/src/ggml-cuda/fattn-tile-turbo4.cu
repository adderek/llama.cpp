// fork: TILE FlashAttention reading a turbo4 KV cache directly, for small batches (decode).
//
// The VEC kernel runs one CUDA block per Q head, so with GQA every KV block is read and
// dequantized once per Q head that shares it (8x for 16 Q heads over 2 KV heads). The TILE
// kernel packs the Q heads of a GQA group into one block (ncols2) and fills its SRAM tile
// once; here that tile is filled from turbo4 blocks instead of f16 rows.

#include "common.cuh"
#include "fattn-tile.cuh"

template <int D, int cols_per_block, int ncols2>
static void launch_fattn_tile_turbo4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static_assert(cols_per_block % ncols2 == 0, "bad cols_per_block");
    constexpr int    ncols1         = cols_per_block/ncols2;
    constexpr size_t nbytes_shared  = 0;
    const int        warp_size      = 32;
    const int        cc             = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (D, D, cols_per_block, cc) / warp_size;
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(D, D, cols_per_block, cc);

    fattn_kernel_t fattn_kernel = flash_attn_tile<D, D, ncols1, ncols2, /*use_logit_softcap=*/false, /*turbo4=*/true>;
    // need_f16_K/V = false: the kernel reads turbo4 itself
    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, false, false, false, warp_size);
}

template <int D>
static void launch_fattn_tile_turbo4_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int  gqa_ratio   = Q->ne[2] / K->ne[2];
    const bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // As in the f16 TILE launcher: the smallest block that holds ncols2 Q heads x all Q rows,
    // so that a single decoded token does not leave most of an 8-column block idle.
    const int64_t n_q = Q->ne[1];
    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        launch_fattn_tile_turbo4<D, 8, 8>(ctx, dst);
    } else if (use_gqa_opt && gqa_ratio % 4 == 0) {
        if (n_q <= 1) { launch_fattn_tile_turbo4<D, 4, 4>(ctx, dst); } else { launch_fattn_tile_turbo4<D, 8, 4>(ctx, dst); }
    } else if (use_gqa_opt && gqa_ratio % 2 == 0) {
        if (n_q <= 1) { launch_fattn_tile_turbo4<D, 2, 2>(ctx, dst); }
        else if (n_q <= 2) { launch_fattn_tile_turbo4<D, 4, 2>(ctx, dst); }
        else { launch_fattn_tile_turbo4<D, 8, 2>(ctx, dst); }
    } else {
        launch_fattn_tile_turbo4<D, 8, 1>(ctx, dst);
    }
}

bool ggml_cuda_flash_attn_ext_tile_turbo4_supported(const ggml_tensor * dst) {
#if TURBO4_USE_4BIT
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return K->type == GGML_TYPE_TURBO4_0 && V->type == GGML_TYPE_TURBO4_0 &&
        (K->ne[0] == 128 || K->ne[0] == 256) && V->ne[0] == K->ne[0] && Q->ne[0] == K->ne[0] &&
        logit_softcap == 0.0f && dst->src[4] == nullptr; // no sinks: kept out until tested
#else
    GGML_UNUSED(dst);
    return false;
#endif // TURBO4_USE_4BIT
}

void ggml_cuda_flash_attn_ext_tile_turbo4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_tile_turbo4_supported(dst));
    switch (dst->src[1]->ne[0]) {
        case 128: launch_fattn_tile_turbo4_switch_ncols2<128>(ctx, dst); break;
        case 256: launch_fattn_tile_turbo4_switch_ncols2<256>(ctx, dst); break;
        default:  GGML_ABORT("fatal error");
    }
}
