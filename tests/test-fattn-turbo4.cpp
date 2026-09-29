// Fork regression test: FLASH_ATTN_EXT with a turbo4 K/V cache, GPU against CPU.
//
// Decode with a turbo4 cache takes a TILE kernel that dequantizes turbo4 blocks straight
// into its SRAM tile (fattn-tile-turbo4.cu), so that one tile serves every Q head of a GQA
// group. This checks it op by op against the CPU backend over head sizes, GQA ratios, query
// row counts and KV lengths, with part of the KV masked. Without a CUDA/HIP device it passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct fa_case {
    int64_t D, n_q, n_head_kv, gqa, n_kv;
};

// returns the output of one FLASH_ATTN_EXT on the given backend
static std::vector<float> run(ggml_backend_t backend, const fa_case & c,
        const std::vector<float> & q_data, const std::vector<uint8_t> & k_data, const std::vector<uint8_t> & v_data,
        const std::vector<ggml_fp16_t> & m_data) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,      c.D, c.n_q,  c.n_head_kv*c.gqa, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO4_0, c.D, c.n_kv, c.n_head_kv,       1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO4_0, c.D, c.n_kv, c.n_head_kv,       1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,      c.n_kv, c.n_q, 1,               1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(m, m_data.data(), 0, ggml_nbytes(m));

    std::vector<float> res;
    if (!ggml_backend_supports_op(backend, out)) {
        // e.g. head 512 without GQA: in a real graph the scheduler would run it elsewhere
        res.assign(1, NAN);
    } else if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
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
    ggml_backend_t be_cpu = ggml_backend_cpu_init();

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<fa_case> cases;
    for (int64_t D : {128, 256, 512}) {
        for (int64_t gqa : {1, 2, 4, 8}) {
            for (int64_t n_q : {1, 2, 3, 8, 32}) {
                for (int64_t n_kv : {256, 1024}) {
                    cases.push_back({D, n_q, 2, gqa, n_kv});
                }
            }
        }
    }

    int n_fail = 0;
    int n_skip = 0;
    for (const fa_case & c : cases) {
        const int64_t n_head = c.n_head_kv*c.gqa;

        std::vector<float> q_data(c.D*c.n_q*n_head);
        for (float & x : q_data) {
            x = nd(rng);
        }

        // K and V: random rows quantized to turbo4 on the CPU
        std::vector<float> kv_f(c.D*c.n_kv*c.n_head_kv);
        const size_t nbytes_kv = ggml_row_size(GGML_TYPE_TURBO4_0, c.D)*c.n_kv*c.n_head_kv;
        std::vector<uint8_t> k_data(nbytes_kv), v_data(nbytes_kv);
        for (float & x : kv_f) { x = nd(rng); }
        ggml_quantize_chunk(GGML_TYPE_TURBO4_0, kv_f.data(), k_data.data(), 0, c.n_kv*c.n_head_kv, c.D, nullptr);
        for (float & x : kv_f) { x = nd(rng); }
        ggml_quantize_chunk(GGML_TYPE_TURBO4_0, kv_f.data(), v_data.data(), 0, c.n_kv*c.n_head_kv, c.D, nullptr);

        // mask: causal-looking tail, last 17 KV positions hidden from the first query row
        std::vector<ggml_fp16_t> m_data(c.n_kv*c.n_q, ggml_fp32_to_fp16(0.0f));
        for (int64_t j = c.n_kv - 17; j < c.n_kv; ++j) {
            m_data[j] = ggml_fp32_to_fp16(-INFINITY);
        }

        const std::vector<float> ref = run(be_cpu, c, q_data, k_data, v_data, m_data);
        const std::vector<float> got = run(be_gpu, c, q_data, k_data, v_data, m_data);
        if (got.size() == 1 && std::isnan(got[0])) {
            n_skip++;
            if (getenv("FATTN_TURBO4_VERBOSE")) {
                fprintf(stderr, "D=%3lld gqa=%lld n_q=%lld n_kv=%5lld: not supported on this device, skipped\n",
                    (long long) c.D, (long long) c.gqa, (long long) c.n_q, (long long) c.n_kv);
            }
            continue;
        }

        double num = 0.0, den = 0.0;
        bool finite = got.size() == ref.size() && !ref.empty();
        for (size_t i = 0; finite && i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) {
                finite = false;
                break;
            }
            num += (got[i] - ref[i])*(got[i] - ref[i]);
            den += ref[i]*ref[i];
        }
        const double err = finite && den > 0.0 ? num/den : INFINITY;
        const bool ok = err < 1e-4;
        if (!ok || getenv("FATTN_TURBO4_VERBOSE")) {
            fprintf(stderr, "D=%3lld gqa=%lld n_q=%lld n_kv=%5lld: nmse %.3e %s\n", (long long) c.D, (long long) c.gqa,
                (long long) c.n_q, (long long) c.n_kv, err, ok ? "OK" : "FAIL");
        }
        n_fail += !ok;
    }

    fprintf(stderr, "%s: %d/%zu cases passed, %d not supported by the device\n", ggml_backend_name(be_gpu),
        (int) cases.size() - n_fail - n_skip, cases.size() - n_skip, n_skip);
    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return n_fail == 0 ? 0 : 1;
}
