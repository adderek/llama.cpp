// Fork regression test: FlashAttention with the DeepSeek MLA head sizes (K 576, V 512) must run on
// the GPU.
//
// The domvox port excluded the D=576 tile kernels from HIP builds, saying they exceed the LDS
// limit (67584 > 65536). Only the fork's D=640 variants do; D=576 compiles and is what upstream
// uses on RDNA3. With the exclusion every MLA FlashAttention on RDNA3 was "not supported" and left
// the GPU, and test-backend-ops reports that as a skip, not a failure. Here the op must be
// supported, and with Q zero every KV position gets the same weight, so the exact output is the
// mean of the V rows. V is a view of K, as in MLA. Without a CUDA/HIP device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct fcase {
    int64_t n_head;
    int64_t n_q;
    int64_t n_kv;
};

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("CUDA", 0) == 0 || name.rfind("ROCm", 0) == 0) {
            gpu = dev;
        }
    }
    if (gpu == nullptr) {
        fprintf(stderr, "no CUDA/HIP devices, skipping\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_dev_init(gpu, nullptr);

    const int64_t DK = 576;
    const int64_t DV = 512;
    // one KV head, as in MLA: decode, a short batch, and prefill
    const fcase cases[] = {
        { 16,   1, 1024 },
        { 16,   8, 1024 },
        { 64,   1, 4096 },
        { 16, 128, 4096 },
    };

    int n_fail = 0;
    for (const fcase & c : cases) {
        ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, c.n_q, c.n_head, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, c.n_kv, 1, 1);
        ggml_tensor * v = ggml_view_4d(ctx, k, DV, c.n_kv, 1, 1, k->nb[1], k->nb[2], k->nb[3], 0);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) DK), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);

        const bool supported = ggml_backend_supports_op(be, o);
        if (!supported) {
            fprintf(stderr, "heads=%lld n_q=%lld n_kv=%lld: FLASH_ATTN_EXT 576/512 not supported FAIL\n",
                    (long long) c.n_head, (long long) c.n_q, (long long) c.n_kv);
            n_fail++;
            ggml_free(ctx);
            continue;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        std::vector<float> zeros(ggml_nelements(q), 0.0f);
        ggml_backend_tensor_set(q, zeros.data(), 0, ggml_nbytes(q));

        std::vector<ggml_fp16_t> kh(ggml_nelements(k));
        std::vector<double> mean(DV, 0.0);
        uint32_t seed = 1;
        for (int64_t j = 0; j < c.n_kv; ++j) {
            for (int64_t i = 0; i < DK; ++i) {
                seed = seed*1664525u + 1013904223u;
                const ggml_fp16_t h = ggml_fp32_to_fp16(4.0f*((float) (seed >> 8) / (float) (1u << 24) - 0.5f));
                kh[j*DK + i] = h;
                if (i < DV) {
                    mean[i] += ggml_fp16_to_fp32(h) / c.n_kv;
                }
            }
        }
        ggml_backend_tensor_set(k, kh.data(), 0, ggml_nbytes(k));
        std::vector<ggml_fp16_t> mask(ggml_nelements(m), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));

        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 1;
        }
        // out: [DV, n_head, n_q]
        std::vector<float> out(ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));

        float max_err = 0.0f;
        for (size_t r = 0; r < out.size(); ++r) {
            const float x = out[r];
            max_err = std::isfinite(x) ? std::max(max_err, (float) std::fabs(x - mean[r % DV])) : INFINITY;
        }
        const bool ok = max_err < 1e-3f;
        fprintf(stderr, "heads=%lld n_q=%lld n_kv=%lld: max |out - mean(V)| = %g %s\n",
                (long long) c.n_head, (long long) c.n_q, (long long) c.n_kv, max_err, ok ? "OK" : "FAIL");
        n_fail += !ok;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(be);
    return n_fail == 0 ? 0 : 1;
}
