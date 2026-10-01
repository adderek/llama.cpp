// Fork regression test: a large-batch mat-mul through hipBLAS/cuBLAS must not overflow f16.
//
// On RDNA3 some quant types (Q6_K, Q2_K, IQ2_XS, ...) leave MMQ above 128 rows and are
// dequantized to f16 for a hipBLAS GEMM. Its output used to be f16 too, so any result past
// 65504 became inf, then NaN downstream: Ornith-1.5-397B answered only "/" with a 512-token
// ubatch. Here the activations are large enough that every output is around 1e5, and the
// result must match the CPU.
//
// Q4_1 and Q5_1 had a second overflow, in the mat-vec path (one row): the dot product
// multiplied the block minimum of the weights by the block sum of the activations in half2.
// Those cases use weights in [-100, 100], so that product reaches ~1e5. Without a CUDA/HIP
// device the test passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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
    ggml_backend_t bes[2] = { ggml_backend_dev_init(gpu, nullptr), ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };

    const int64_t K = 1024, N = 256;
    const ggml_type types[] = { GGML_TYPE_Q6_K, GGML_TYPE_Q2_K, GGML_TYPE_F16, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1 };
    const int64_t   rows[]  = { 1, 64, 512 }; // mat-vec, below and above the MMQ cut-off

    int n_fail = 0;
    for (ggml_type type : types) {
        for (int64_t M : rows) {
            std::vector<float> res[2];
            for (int b = 0; b < 2; ++b) {
                ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, N);
                ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
                ggml_tensor * y = ggml_mul_mat(ctx, w, x);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, y);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, bes[b]);

                // weights ~0.5 with one sign, activations ~200: each output is ~K * 0.5 * 200 = 1e5
                uint32_t seed = 3;
                auto rnd = [&seed]() {
                    seed = seed*1664525u + 1013904223u;
                    return (float) (seed >> 8) / (float) (1u << 24);
                };
                std::vector<float> wf(ggml_nelements(w)), xf(ggml_nelements(x));
                const bool with_min = type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1;
                for (float & v : wf) { v = with_min ? 200.0f*rnd() - 100.0f : 0.25f + 0.5f*rnd(); }
                for (float & v : xf) { v = 100.0f + 200.0f*rnd(); }
                std::vector<uint8_t> wq(ggml_nbytes(w));
                ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, nullptr);
                ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
                ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));

                ggml_backend_graph_compute(bes[b], gf);
                res[b].resize(ggml_nelements(y));
                ggml_backend_tensor_get(y, res[b].data(), 0, ggml_nbytes(y));
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
            double err = 0.0, ref = 0.0; int nonfinite = 0; float max_ref = 0.0f;
            for (size_t i = 0; i < res[1].size(); ++i) {
                if (!std::isfinite(res[0][i])) { nonfinite++; continue; }
                err += (res[0][i] - res[1][i]) * (res[0][i] - res[1][i]);
                ref += res[1][i] * res[1][i];
                max_ref = std::max(max_ref, std::fabs(res[1][i]));
            }
            const bool ok = nonfinite == 0 && err / ref < 1e-4;
            fprintf(stderr, "%-5s rows=%3lld: max |ref| %.0f, non-finite %d, NMSE %g %s\n", ggml_type_name(type),
                    (long long) M, max_ref, nonfinite, err / ref, ok ? "OK" : "FAIL");
            n_fail += !ok;
        }
    }

    ggml_backend_free(bes[0]);
    ggml_backend_free(bes[1]);
    return n_fail == 0 ? 0 : 1;
}
