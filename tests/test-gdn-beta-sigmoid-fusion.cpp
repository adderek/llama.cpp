// Fork regression test: GATED_DELTA_NET applying the sigmoid of beta itself must give the same bits.
//
// On CUDA/HIP the fork runs SIGMOID(beta) -> GATED_DELTA_NET as one kernel: the GDN is launched at
// the sigmoid's place and applies the sigmoid while reading beta (see the fusion in
// ggml_cuda_try_fuse and beta_sigmoid in gated_delta_net.cu). That is the beta of every
// linear-attention layer of qwen3next/qwen35. Each case is built twice on the GPU, as written and
// with the sigmoid marked as an output so it cannot be fused, and the results must match bit for
// bit; both must be close to the CPU backend. The cases cover one token (decode), several tokens
// and sequences, the GDN whose state snapshots are copied into a cache (that copy is fused into
// the GDN as well, and must still be), and a reshape between the sigmoid and the GDN (not fused,
// must still be right). Without a CUDA/HIP device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int64_t S = 64; // head size
static const int64_t H = 4;  // heads

struct gcase {
    const char * name;
    int64_t      n_tokens;
    int64_t      n_seqs;
    int64_t      K;          // state snapshots
    bool         cache_cpy;  // snapshots copied into a cache tensor right after the GDN
    bool         reshaped;   // beta is sigmoided as [H, n_tokens*n_seqs] and reshaped for the GDN
};

struct net {
    ggml_context * ctx;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outs;
    ggml_cgraph  * gf;
};

static net build(const gcase & c, bool block_fusion) {
    net n;
    ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    n.ctx = ggml_init(ip);
    ggml_context * ctx = n.ctx;

    const int64_t T = c.n_tokens;
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, c.n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, c.n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, c.n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, c.n_seqs);
    ggml_tensor * bl    = c.reshaped ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T*c.n_seqs)
                                     : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, c.n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, c.n_seqs);
    n.inputs = { q, k, v, g, bl, state };

    q = ggml_l2_norm(ctx, q, 1e-6f);
    k = ggml_l2_norm(ctx, k, 1e-6f);
    ggml_tensor * beta = ggml_sigmoid(ctx, bl);
    if (block_fusion) {
        ggml_set_output(beta);
    }
    if (c.reshaped) {
        beta = ggml_reshape_4d(ctx, beta, 1, H, T, c.n_seqs);
    }
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, c.K);

    n.gf = ggml_new_graph(ctx);
    if (!c.cache_cpy) {
        ggml_build_forward_expand(n.gf, out);
        n.outs = { out };
        return n;
    }

    // as the model does: the snapshot tail [D, n_seqs, n_written] goes into a cache slot range
    const int64_t D         = S*S*H;
    const int64_t n_written = T < c.K ? T : c.K;
    const size_t  tail      = ggml_row_size(GGML_TYPE_F32, S*H*T*c.n_seqs);
    ggml_tensor * snap  = ggml_view_3d(ctx, out, D, c.n_seqs, n_written,
            ggml_row_size(GGML_TYPE_F32, D), ggml_row_size(GGML_TYPE_F32, D*c.n_seqs), tail);
    ggml_tensor * cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.n_seqs, c.K);
    n.inputs.push_back(cache);
    ggml_tensor * dst = ggml_view_3d(ctx, cache, D, c.n_seqs, n_written, cache->nb[1], cache->nb[2], 0);
    ggml_build_forward_expand(n.gf, ggml_cpy(ctx, snap, dst));
    ggml_tensor * attn = ggml_cont(ctx, ggml_view_4d(ctx, out, S, H, T, c.n_seqs,
            ggml_row_size(GGML_TYPE_F32, S), ggml_row_size(GGML_TYPE_F32, S*H), ggml_row_size(GGML_TYPE_F32, S*H*T), 0));
    ggml_build_forward_expand(n.gf, attn);
    n.outs = { attn, cache };
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
        { "one token",                  1, 1, 1, false, false },
        { "tokens and sequences",       5, 2, 1, false, false },
        { "snapshots into cache",       3, 1, 2, true,  false },
        { "decode, snapshot into cache", 1, 1, 1, true,  false },
        { "reshape in between",         4, 1, 1, false, true  },
    };

    uint32_t seed = 777;
    auto rnd = [&seed](float lo, float hi) {
        seed = seed*1664525u + 1013904223u;
        return lo + (float) (seed >> 8) / (float) (1u << 24) * (hi - lo);
    };

    int n_fail = 0;
    for (const gcase & c : cases) {
        net nets[3] = { build(c, false), build(c, true), build(c, false) };
        ggml_backend_t bes[3] = { be_gpu, be_gpu, be_cpu };
        ggml_backend_buffer_t bufs[3];
        for (int b = 0; b < 3; ++b) {
            bufs[b] = ggml_backend_alloc_ctx_tensors(nets[b].ctx, bes[b]);
        }
        for (size_t t = 0; t < nets[0].inputs.size(); ++t) {
            // g is a log decay (<= 0), beta a logit, the rest centred
            const float lo = t == 3 ? -1.0f : t == 4 ? -4.0f : -0.5f;
            const float hi = t == 3 ?  0.0f : t == 4 ?  4.0f :  0.5f;
            std::vector<float> v(ggml_nelements(nets[0].inputs[t]));
            for (float & f : v) {
                f = rnd(lo, hi);
            }
            for (int b = 0; b < 3; ++b) {
                ggml_backend_tensor_set(nets[b].inputs[t], v.data(), 0, v.size()*sizeof(float));
            }
        }
        for (int b = 0; b < 3; ++b) {
            if (ggml_backend_graph_compute(bes[b], nets[b].gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed\n", c.name);
                return 1;
            }
        }

        for (size_t o = 0; o < nets[0].outs.size(); ++o) {
            const std::vector<float> fused   = get(nets[0].outs[o]);
            const std::vector<float> unfused = get(nets[1].outs[o]);
            const std::vector<float> cpu     = get(nets[2].outs[o]);
            double err = 0.0, ref = 0.0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                err += (fused[i] - cpu[i]) * (fused[i] - cpu[i]);
                ref += cpu[i] * cpu[i];
            }
            const bool same = memcmp(fused.data(), unfused.data(), fused.size()*sizeof(float)) == 0;
            if (!same || !(err / ref < 1e-8)) {
                fprintf(stderr, "%s, output %zu: %s, NMSE vs CPU %g\n", c.name, o,
                        same ? "same bits as unfused" : "DIFFERS from unfused", err / ref);
                n_fail++;
            }
        }

        for (int b = 0; b < 3; ++b) {
            ggml_backend_buffer_free(bufs[b]);
            ggml_free(nets[b].ctx);
        }
    }

    if (n_fail == 0) {
        fprintf(stderr, "%s: %zu sigmoid(beta) -> gated_delta_net cases match the unfused kernels bit for bit\n",
                ggml_backend_name(be_gpu), sizeof(cases)/sizeof(cases[0]));
    }
    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return n_fail == 0 ? 0 : 1;
}
