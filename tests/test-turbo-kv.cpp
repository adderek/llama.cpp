// Fork regression test: a TurboQuant KV cache must give the same attention as an f16 one,
// up to quantization noise.
//
// TurboQuant stores K (and V) WHT-rotated, so every attention path has to rotate Q into
// the same basis and un-rotate the output. A path that forgets does not fail loudly:
// shapes agree, nothing aborts, and the result is merely wrong (qwen4exp QSA answered
// with a filler word instead of the planted needle, see BUG_TURBOQUANT_QSA_Q_ROTATION.md).
//
// The test decodes the same prompt with an f16 cache and with each turbo type, captures
// the attention output of every layer, and compares it with the f16 one. q4_0 is the
// yardstick: a plain block quant with no rotation, so its distance from f16 is what honest
// quantization noise looks like for this model. Measured on the generated models, a
// correct path puts turbo4 at 1-3x the q4_0 error, a missing Q rotation at 34-51x.
//
// The generated models have small random weights, so their attention is close to uniform,
// and a uniform attention reads the same average whichever basis the scores were computed
// in. TURBO_KV_SCALE multiplies Q to sharpen it; see tests/CMakeLists.txt for the value
// each model needs.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-model.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

// Scale Q by exactly one linear factor per layer, applied after the last normalization of Q:
// the Q norm weight if there is one, else the MLA up-projection, else the Q projection.
static int sharpen_attention(llama_model * model, float scale) {
    std::map<std::string, ggml_tensor *> by_name(model->tensors_by_name.begin(), model->tensors_by_name.end());

    std::vector<ggml_tensor *> targets;
    for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
        const std::string blk = "blk." + std::to_string(il) + ".";
        for (const auto & group : std::vector<std::vector<const char *>>{
                { "attn_q_norm.weight" }, { "attn_q_b.weight" }, { "attn_q.weight", "attn_q.bias" } }) {
            bool found = false;
            for (const char * suffix : group) {
                auto it = by_name.find(blk + suffix);
                if (it != by_name.end()) {
                    targets.push_back(it->second);
                    found = true;
                }
            }
            if (found) {
                break;
            }
        }
    }

    int n = 0;
    for (ggml_tensor * t : targets) {
        const int64_t ne = ggml_nelements(t);
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> buf(ne);
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
            for (float & x : buf) {
                x *= scale;
            }
            ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf(ne);
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
            for (ggml_fp16_t & x : buf) {
                x = ggml_fp32_to_fp16(ggml_fp16_to_fp32(x)*scale);
            }
            ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
        } else {
            continue;
        }
        n++;
    }
    return n;
}

// attention outputs of every layer, concatenated over the ubatches of the prompt
using attn_outputs = std::map<std::string, std::vector<float>>;

static bool collect_attn(ggml_tensor * t, bool ask, void * user_data) {
    // kqv_out is the attention result; some archs rename that tensor, so also take attn_output
    const bool match = strncmp(t->name, "kqv_out", 7) == 0 || strncmp(t->name, "attn_output", 11) == 0;
    if (!match || t->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ask) {
        auto & v = (*(attn_outputs *) user_data)[t->name];
        const size_t n = v.size();
        v.resize(n + ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data() + n, 0, ggml_nbytes(t));
    }
    return true;
}

static attn_outputs run(llama_model * model, const common_params & params, ggml_type type_kv) {
    constexpr uint32_t n_prompt = 96;

    attn_outputs out;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx             = 256;
    cparams.n_batch           = n_prompt;
    cparams.n_ubatch          = 32; // several ubatches, so later tokens attend to K read back from the cache
    cparams.n_seq_max         = 1;
    cparams.type_k            = type_kv;
    cparams.type_v            = type_kv;
    cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.cb_eval           = collect_attn;
    cparams.cb_eval_user_data = &out;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return {};
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        common_batch_add(batch, (llama_token) ((7*pos + 1) % (uint32_t) n_vocab), pos, { 0 }, pos + 1 == n_prompt);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    llama_free(ctx);

    return ok ? out : attn_outputs{};
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & x) {
    if (ref.size() != x.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(x[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double d = (double) x[i] - ref[i];
        num += d*d;
        den += (double) ref[i]*ref[i];
    }
    return den == 0.0 ? 0.0 : num/den;
}

// worst layer
static double nmse(const attn_outputs & ref, const attn_outputs & x) {
    double worst = 0.0;
    for (const auto & [name, r] : ref) {
        auto it = x.find(name);
        worst = std::max(worst, it == x.end() ? std::numeric_limits<double>::infinity() : nmse(r, it->second));
    }
    return worst;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    // the weights are modified in place below, which a read-only mmap would not allow
    params.load_mode = LLAMA_LOAD_MODE_NONE;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    // The generated models set ALiBi (max bias 8) and clamp Q/K/V to [-1, 1]. The first turns
    // attention into a near-fixed recency pattern, the second undoes the Q scaling.
    model->hparams.f_max_alibi_bias = 0.0f;
    model->hparams.f_clamp_kqv      = 0.0f;

    const char * scale_env = getenv("TURBO_KV_SCALE");
    const float  scale     = scale_env ? (float) atof(scale_env) : 1.0f;
    if (scale != 1.0f) {
        fprintf(stderr, "%s : scaled Q in %d tensors by %g\n", __func__, sharpen_attention(model, scale), scale);
    }

    const attn_outputs ref = run(model, params, GGML_TYPE_F16);
    if (ref.empty()) {
        fprintf(stderr, "%s : f16 reference decode failed\n", __func__);
        return 1;
    }

    const attn_outputs yardstick = run(model, params, GGML_TYPE_Q4_0);
    if (yardstick.empty()) {
        fprintf(stderr, "%s : q4_0 decode failed\n", __func__);
        return 1;
    }
    const double noise = nmse(ref, yardstick);
    fprintf(stderr, "%s : q4_0     nmse vs f16 = %.3e (yardstick)\n", __func__, noise);

    // allowed error as a multiple of the q4_0 error; 0 = only require a finite result
    const struct { ggml_type type; double max_ratio; } cases[] = {
        { GGML_TYPE_TURBO4_0, 6.0 },
        { GGML_TYPE_TURBO3_0, 8.0 },
        { GGML_TYPE_TURBO2_0, 0.0 },
    };

    bool ok = true;
    for (const auto & c : cases) {
        const attn_outputs x = run(model, params, c.type);
        if (x.empty()) {
            fprintf(stderr, "%s : %-8s FAIL: decode failed\n", __func__, ggml_type_name(c.type));
            ok = false;
            continue;
        }
        const double err   = nmse(ref, x);
        const double ratio = noise > 0.0 ? err/noise : std::numeric_limits<double>::infinity();
        const bool   pass  = std::isfinite(err) && (c.max_ratio == 0.0 || ratio <= c.max_ratio);
        fprintf(stderr, "%s : %-8s nmse vs f16 = %.3e (%.1fx q4_0, limit %s) %s\n", __func__,
                ggml_type_name(c.type), err, ratio,
                c.max_ratio == 0.0 ? "finite" : (std::to_string((int) c.max_ratio) + "x").c_str(),
                pass ? "OK" : "FAIL");
        ok = ok && pass;
    }

    return ok ? 0 : 1;
}
