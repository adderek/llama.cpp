// Fork regression test: the fused ADD -> UNARY -> MUL must give the same bits as the unfused kernels.
//
// On CUDA/HIP the fork runs op(x + bias) * g as one kernel for softplus, sigmoid and silu, with
// bias and g either full size or one row broadcast over all rows (see add_unary_mul_kernel in
// unary.cu). In qwen3next/qwen35 that is softplus(alpha + dt) * a in every linear-attention layer.
// Each case is built twice on the GPU: as written, and with the ADD marked as an output, which
// keeps the backend from fusing it. The two must match bit for bit, and both must be close to the
// CPU backend. Inputs reach past 20, where softplus switches to the identity. Without a CUDA/HIP
// device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int64_t N = 48;

struct fcase {
    const char *  name;
    ggml_unary_op op;
    int64_t       rows;
    bool          bias_full;  // bias has every row, not one broadcast row
    bool          bias_first; // bias is src0 of the ADD
    bool          g_full;
    bool          g_first;    // g is src0 of the MUL
};

struct net {
    ggml_context * ctx;
    std::vector<ggml_tensor *> inputs;
    ggml_tensor  * out;
    ggml_cgraph  * gf;
};

static net build(const fcase & c, bool block_fusion) {
    net n;
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    n.ctx = ggml_init(ip);
    ggml_context * ctx = n.ctx;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.bias_full ? c.rows : 1);
    ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.g_full ? c.rows : 1);
    n.inputs = { x, b, g };

    // ggml_add broadcasts src1 only, so a broadcast bias as src0 needs the full-size x as src1
    ggml_tensor * s = c.bias_first && c.bias_full ? ggml_add(ctx, b, x) : ggml_add(ctx, x, b);
    if (block_fusion) {
        ggml_set_output(s);
    }
    ggml_tensor * u = ggml_unary(ctx, s, c.op);
    n.out = c.g_first && c.g_full ? ggml_mul(ctx, g, u) : ggml_mul(ctx, u, g);

    n.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(n.gf, n.out);
    return n;
}

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
    ggml_backend_t be_gpu = ggml_backend_dev_init(gpu, nullptr);
    ggml_backend_t be_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    const fcase cases[] = {
        { "softplus, one row (decode)",  GGML_UNARY_OP_SOFTPLUS, 1, false, false, false, false },
        { "softplus, rows, broadcast",   GGML_UNARY_OP_SOFTPLUS, 5, false, false, false, false },
        { "softplus, full bias and g",   GGML_UNARY_OP_SOFTPLUS, 5, true,  true,  true,  true  },
        { "sigmoid, broadcast g first",  GGML_UNARY_OP_SIGMOID,  3, false, false, true,  true  },
        { "silu, full bias",             GGML_UNARY_OP_SILU,     3, true,  false, false, false },
    };

    uint32_t seed = 4242;
    auto rnd = [&seed]() {
        seed = seed*1664525u + 1013904223u;
        return (float) (seed >> 8) / (float) (1u << 24) * 30.0f - 12.0f;
    };

    int n_fail = 0;
    for (const fcase & c : cases) {
        net nets[3] = { build(c, false), build(c, true), build(c, false) };
        ggml_backend_t bes[3] = { be_gpu, be_gpu, be_cpu };
        ggml_backend_buffer_t bufs[3];
        for (int k = 0; k < 3; ++k) {
            bufs[k] = ggml_backend_alloc_ctx_tensors(nets[k].ctx, bes[k]);
        }
        for (size_t t = 0; t < nets[0].inputs.size(); ++t) {
            std::vector<float> v(ggml_nelements(nets[0].inputs[t]));
            for (float & f : v) {
                f = rnd();
            }
            for (int k = 0; k < 3; ++k) {
                ggml_backend_tensor_set(nets[k].inputs[t], v.data(), 0, v.size()*sizeof(float));
            }
        }
        std::vector<float> res[3];
        for (int k = 0; k < 3; ++k) {
            if (ggml_backend_graph_compute(bes[k], nets[k].gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed\n", c.name);
                return 1;
            }
            res[k].resize(ggml_nelements(nets[k].out));
            ggml_backend_tensor_get(nets[k].out, res[k].data(), 0, ggml_nbytes(nets[k].out));
        }

        double err = 0.0, ref = 0.0;
        for (size_t i = 0; i < res[2].size(); ++i) {
            err += (res[0][i] - res[2][i]) * (res[0][i] - res[2][i]);
            ref += res[2][i] * res[2][i];
        }
        const bool same = memcmp(res[0].data(), res[1].data(), res[0].size()*sizeof(float)) == 0;
        if (!same || !(err / ref < 1e-10)) {
            fprintf(stderr, "%s: %s, NMSE vs CPU %g\n", c.name, same ? "same bits as unfused" : "DIFFERS from unfused", err / ref);
            n_fail++;
        }

        for (int k = 0; k < 3; ++k) {
            ggml_backend_buffer_free(bufs[k]);
            ggml_free(nets[k].ctx);
        }
    }

    if (n_fail == 0) {
        fprintf(stderr, "%s: %zu add -> unary -> mul cases match the unfused kernels bit for bit\n",
                ggml_backend_name(be_gpu), sizeof(cases)/sizeof(cases[0]));
    }
    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return n_fail == 0 ? 0 : 1;
}
