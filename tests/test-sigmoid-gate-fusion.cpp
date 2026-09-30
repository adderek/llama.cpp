// Fork regression test: the fused sigmoid gate must give the same bits as the unfused kernels.
//
// On CUDA/HIP the fork runs sigmoid(gate) * x followed by up to four adds as one kernel when the
// gate has one value per row of x (see sigmoid_gate_add_kernel in unary.cu). That is the shared
// expert gate of qwen3next/qwen35moe, added into the routed output and the residual. The kernel
// promises the exact result of the separate SIGMOID, MUL and ADD kernels. Each case here is built
// twice on the GPU: once as the model builds it, and once with the sigmoid marked as an output,
// which keeps the backend from fusing it. The two must match bit for bit, and both must be close
// to the CPU backend. The cases cover the chain with and without adds, the gated product on
// either side of an add, a chain longer than the kernel takes, several rows, and a sigmoid that
// is read twice (must not be fused away). Without a CUDA/HIP device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int64_t N = 2048;

struct gcase {
    const char * name;
    int64_t      rows;
    int          n_add;
    bool         gated_first; // the gated product is src0 of the adds, not src1
    bool         shared;      // the sigmoid is also read by a second, separate MUL
};

struct net {
    ggml_context * ctx;
    std::vector<ggml_tensor *> inputs;
    ggml_tensor  * out;
    ggml_tensor  * out2;
    ggml_cgraph  * gf;
};

static net build(const gcase & c, bool block_fusion) {
    net n;
    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    n.ctx = ggml_init(ip);
    ggml_context * ctx = n.ctx;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.rows);
    ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.rows);
    n.inputs = { x, g };

    ggml_tensor * s = ggml_sigmoid(ctx, g);
    if (block_fusion) {
        ggml_set_output(s);
    }
    ggml_tensor * cur = ggml_mul(ctx, x, s);
    for (int j = 0; j < c.n_add; ++j) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.rows);
        n.inputs.push_back(a);
        // the first add is ggml_add(moe_out, shexp) in the model, the next ones chain through src0
        cur = (j == 0) != c.gated_first ? ggml_add(ctx, a, cur) : ggml_add(ctx, cur, a);
    }
    n.out  = cur;
    n.out2 = nullptr;
    if (c.shared) {
        ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, c.rows);
        n.inputs.push_back(y);
        n.out2 = ggml_mul(ctx, y, s);
    }

    n.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(n.gf, n.out);
    if (n.out2) {
        ggml_build_forward_expand(n.gf, n.out2);
    }
    return n;
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
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

    const gcase cases[] = {
        { "mul only",               1, 0, false, false },
        { "one add",                1, 1, false, false },
        { "routed + residual",      1, 2, false, false },
        { "gated product as src0",  1, 2, true,  false },
        { "three rows",             3, 2, false, false },
        { "chain longer than four", 2, 6, false, false },
        { "sigmoid read twice",     2, 2, false, true  },
    };

    uint32_t seed = 12345;
    auto rnd = [&seed]() {
        seed = seed*1664525u + 1013904223u;
        return (float) (seed >> 8) / (float) (1u << 24) * 8.0f - 4.0f;
    };

    int n_fail = 0;
    for (const gcase & c : cases) {
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
        for (int k = 0; k < 3; ++k) {
            if (ggml_backend_graph_compute(bes[k], nets[k].gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed\n", c.name);
                return 1;
            }
        }

        for (int o = 0; o < (c.shared ? 2 : 1); ++o) {
            const std::vector<float> fused   = get(o ? nets[0].out2 : nets[0].out);
            const std::vector<float> unfused = get(o ? nets[1].out2 : nets[1].out);
            const std::vector<float> cpu     = get(o ? nets[2].out2 : nets[2].out);
            double err = 0.0, ref = 0.0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                err += (fused[i] - cpu[i]) * (fused[i] - cpu[i]);
                ref += cpu[i] * cpu[i];
            }
            const bool same = memcmp(fused.data(), unfused.data(), fused.size()*sizeof(float)) == 0;
            if (!same || !(err / ref < 1e-10)) {
                fprintf(stderr, "%s, output %d: %s, NMSE vs CPU %g\n", c.name, o,
                        same ? "same bits as unfused" : "DIFFERS from unfused", err / ref);
                n_fail++;
            }
        }

        for (int k = 0; k < 3; ++k) {
            ggml_backend_buffer_free(bufs[k]);
            ggml_free(nets[k].ctx);
        }
    }

    if (n_fail == 0) {
        fprintf(stderr, "%s: %zu sigmoid gate cases match the unfused kernels bit for bit\n",
                ggml_backend_name(be_gpu), sizeof(cases)/sizeof(cases[0]));
    }
    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return n_fail == 0 ? 0 : 1;
}
