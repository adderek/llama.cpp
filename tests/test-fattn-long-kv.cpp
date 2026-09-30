// Fork regression test: FlashAttention over a long KV cache must not overflow its accumulators.
//
// The MMA kernel sums P*V over all KV positions before it divides by the row sum. On RDNA3 that
// sum was kept in f16, so a row that attends evenly over 32k positions with |V| = 3 reaches
// 98k, past the f16 maximum of 65504, and turns into inf; before that it loses precision
// (at 16k the f16 step is 16). Here Q is zero, so every position gets the same weight, and V is
// constant, so the exact output is that constant. The cases use the MMA path (many query rows,
// GQA) with head sizes 128 and 256 and f16 / q8_0 caches. Without a CUDA/HIP device the test
// passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct fcase {
    int64_t   D;
    int64_t   n_head;
    int64_t   n_head_kv;
    int64_t   n_q;
    int64_t   n_kv;
    ggml_type type_kv;
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

    const float v_val = 3.0f;
    const fcase cases[] = {
        { 256, 32, 8, 512, 32768, GGML_TYPE_F16  },
        { 128, 32, 8, 64, 32768, GGML_TYPE_F16  },
        { 256, 32, 8, 512, 32768, GGML_TYPE_Q8_0 },
    };

    int n_fail = 0;
    for (const fcase & c : cases) {
        ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.n_q, c.n_head, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, c.type_kv, c.D, c.n_kv, c.n_head_kv, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, c.type_kv, c.D, c.n_kv, c.n_head_kv, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) c.D), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

        std::vector<float> zeros(ggml_nelements(q), 0.0f);
        ggml_backend_tensor_set(q, zeros.data(), 0, ggml_nbytes(q));

        // K arbitrary (Q is zero, so every logit is 0), V constant
        std::vector<float> kf(ggml_nelements(k)), vf(ggml_nelements(v), v_val);
        uint32_t seed = 1;
        for (float & x : kf) {
            seed = seed*1664525u + 1013904223u;
            x = (float) (seed >> 8) / (float) (1u << 24) - 0.5f;
        }
        for (auto [t, f] : { std::pair<ggml_tensor *, std::vector<float> *>{ k, &kf }, { v, &vf } }) {
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, f->data(), data.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
            ggml_backend_tensor_set(t, data.data(), 0, data.size());
        }
        std::vector<ggml_fp16_t> mask(ggml_nelements(m), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));

        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 1;
        }
        std::vector<float> out(ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));

        float max_err = 0.0f;
        for (float x : out) {
            max_err = std::isfinite(x) ? std::max(max_err, std::fabs(x - v_val)) : INFINITY;
        }
        const bool ok = max_err < 1e-2f;
        fprintf(stderr, "D=%lld heads=%lld/%lld n_q=%lld n_kv=%lld %s: max |out - %g| = %g %s\n",
                (long long) c.D, (long long) c.n_head, (long long) c.n_head_kv, (long long) c.n_q, (long long) c.n_kv,
                ggml_type_name(c.type_kv), v_val, max_err, ok ? "OK" : "FAIL");
        n_fail += !ok;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(be);
    return n_fail == 0 ? 0 : 1;
}
