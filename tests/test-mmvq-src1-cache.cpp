// Fork regression test: the MMVQ src1 quantization cache must never serve a stale q8_1 copy.
//
// On CUDA/HIP the fork keeps the last two q8_1 quantizations of mat-vec inputs and reuses one
// when a later mat-vec in the same graph evaluation reads the same tensor (see mmvq.cu,
// GGML_CUDA_MMVQ_SRC1_CACHE). This graph mixes the cases the cache key has to tell apart: one
// input read by several weights of different types (hit), a derived input in between, the first
// input again afterwards (hit in the second slot), three inputs cycling through the two slots
// (evict, re-quantize), MUL_MAT_ID sharing an input, one input read both reshaped and as is
// (hit), a reshape of an in-place result over the same bytes (miss), a leaf overwritten by the
// graph and read again through a reshape (miss), and the same input last in one graph and first
// in the next. The input changes on every compute, so a copy kept across graph
// evaluations or baked into a captured CUDA graph shows up as a mismatch against the CPU
// backend. Without a CUDA/HIP device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static const int64_t K        = 4096;
static const int64_t N        = 512;
static const int     N_EXPERT = 4;

struct net {
    ggml_context * ctx;
    ggml_tensor  * x;
    ggml_tensor  * ids;
    ggml_tensor  * st;
    std::vector<ggml_tensor *> weights;
    std::vector<ggml_tensor *> outs;
    ggml_cgraph  * gf;
};

static net build() {
    net n;
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    n.ctx = ggml_init(ip);
    ggml_context * ctx = n.ctx;

    n.x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    n.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    n.st  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    ggml_tensor * w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    ggml_tensor * e1 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, K, N, N_EXPERT);
    ggml_tensor * e2 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, K, N, N_EXPERT);
    n.weights = { w1, w2, w3, e1, e2 };

    ggml_tensor * x2 = ggml_scale(ctx, n.x, -0.5f);
    ggml_tensor * x3 = ggml_reshape_3d(ctx, n.x, K, 1, 1);

    n.outs.push_back(ggml_mul_mat(ctx, w1, n.x));         // quantize x
    n.outs.push_back(ggml_mul_mat(ctx, w2, n.x));         // same x, other weight type: reuse
    n.outs.push_back(ggml_mul_mat(ctx, w3, x2));          // other input: quantize x2
    n.outs.push_back(ggml_mul_mat(ctx, w2, n.x));         // x again after x2: second slot still has it
    ggml_tensor * x4 = ggml_scale(ctx, n.x, 2.0f);
    n.outs.push_back(ggml_mul_mat(ctx, w1, x4));          // third input evicts x2 (least recently used)
    n.outs.push_back(ggml_mul_mat(ctx, w3, x2));          // x2 again: evicted, must be re-quantized
    n.outs.push_back(ggml_mul_mat_id(ctx, e1, x3, n.ids)); // MoE: gate/up style pair on one input
    n.outs.push_back(ggml_mul_mat_id(ctx, e2, x3, n.ids));
    ggml_tensor * h  = ggml_scale(ctx, n.x, 0.75f);        // a computed input (a leaf is never unwrapped)
    n.outs.push_back(ggml_mul_mat_id(ctx, e1, ggml_reshape_3d(ctx, h, K, 1, 1), n.ids)); // routed experts read it reshaped
    n.outs.push_back(ggml_mul_mat(ctx, w2, h));           // shared expert reads it as is: reuse
    ggml_tensor * hb = ggml_scale_inplace(ctx, h, -3.0f); // rewrites h's bytes
    n.outs.push_back(ggml_mul_mat(ctx, w1, ggml_reshape_2d(ctx, hb, K, 1))); // same bytes, new values: quantize
    n.outs.push_back(ggml_mul_mat(ctx, w2, n.st));        // a leaf, like a cache or a recurrent state...
    n.outs.push_back(ggml_cpy(ctx, x2, n.st));            // ...that the graph overwrites...
    n.outs.push_back(ggml_mul_mat(ctx, w3, ggml_reshape_2d(ctx, n.st, K, 1))); // ...and reads again: quantize
    n.outs.push_back(ggml_mul_mat(ctx, w1, n.x));         // last read is x, so the next compute's first
                                                           // mat-vec would hit if the key ignored the graph

    n.gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : n.outs) {
        ggml_build_forward_expand(n.gf, t);
    }
    return n;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        err += (a[i] - b[i]) * (a[i] - b[i]);
        ref += b[i] * b[i];
    }
    return err / ref;
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

    net g = build();
    net c = build();
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(g.ctx, be_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(c.ctx, be_cpu);

    uint32_t seed = 12345;
    auto rnd = [&seed]() {
        seed = seed*1664525u + 1013904223u;
        return (float) (seed >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
    };

    for (size_t i = 0; i < g.weights.size(); ++i) {
        ggml_tensor * w = g.weights[i];
        const int64_t n_per_row = w->ne[0];
        const int64_t nrows     = ggml_nrows(w);
        std::vector<float> f(n_per_row*nrows);
        for (float & v : f) {
            v = rnd();
        }
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(w->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
        ggml_backend_tensor_set(w,            q.data(), 0, q.size());
        ggml_backend_tensor_set(c.weights[i], q.data(), 0, q.size());
    }

    const double max_nmse = 5e-4;
    for (int it = 0; it < 8; ++it) {
        std::vector<float> x(K);
        for (float & v : x) {
            v = rnd();
        }
        const int32_t ids[2] = { it % N_EXPERT, (it + 1 + it/N_EXPERT) % N_EXPERT };
        ggml_backend_tensor_set(g.x,   x.data(), 0, ggml_nbytes(g.x));
        ggml_backend_tensor_set(c.x,   x.data(), 0, ggml_nbytes(c.x));
        ggml_backend_tensor_set(g.st,  x.data(), 0, ggml_nbytes(g.st)); // the graph overwrote it last time
        ggml_backend_tensor_set(c.st,  x.data(), 0, ggml_nbytes(c.st));
        ggml_backend_tensor_set(g.ids, ids,      0, sizeof(ids));
        ggml_backend_tensor_set(c.ids, ids,      0, sizeof(ids));

        if (ggml_backend_graph_compute(be_gpu, g.gf) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(be_cpu, c.gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed at iteration %d\n", it);
            return 1;
        }

        for (size_t o = 0; o < g.outs.size(); ++o) {
            std::vector<float> a(ggml_nelements(g.outs[o]));
            std::vector<float> b(ggml_nelements(c.outs[o]));
            ggml_backend_tensor_get(g.outs[o], a.data(), 0, ggml_nbytes(g.outs[o]));
            ggml_backend_tensor_get(c.outs[o], b.data(), 0, ggml_nbytes(c.outs[o]));
            const double e = nmse(a, b);
            if (!(e < max_nmse)) {
                fprintf(stderr, "iteration %d, output %zu: NMSE %g > %g\n", it, o, e, max_nmse);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s: %zu mat-vecs sharing inputs matched CPU over 8 computes\n", ggml_backend_name(be_gpu), g.outs.size());
    ggml_backend_buffer_free(buf_gpu);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(g.ctx);
    ggml_free(c.ctx);
    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return 0;
}
