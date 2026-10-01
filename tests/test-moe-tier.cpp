// Regression test for the MoE expert tiering of this fork (LLAMA_MOE_HOT, LLAMA_MOE_DIRECT, tools/moe-tier).
//
// A tiny qwen3moe model is written to a file, then every tiering configuration must give the logits of the
// untiered model: bit-identical on the CPU, where each routed row is computed once and the other parts add zeros,
// and within a tight NMSE on other devices. The LLAMA_MOE_* variables are read once per process, so each
// configuration runs in a child process: `test-moe-tier child MODEL OUT DEVICE N_UBATCH`.
//
// The model file must be on a file system with O_DIRECT (not tmpfs on old kernels): the test fails when the
// arena is not used, instead of passing on the mmap fallback.

#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "common.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

static const uint32_t N_EXPERT      = 16;
static const uint32_t N_EXPERT_USED = 4;
static const uint32_t N_TOKENS      = 100; // more than one ubatch of 64
static const size_t   SEED          = 1234;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    // large expert weights, so that a wrong or stale expert moves the logits far past the tolerance
    const bool exps = strstr(tensor->name, "_exps") != nullptr || strstr(tensor->name, "gate_inp") != nullptr;
    std::normal_distribution<float> dis(0.0f, exps ? 0.2f : 0.02f);

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ggml_nelements(tensor));
    for (auto & x : tmp) {
        x = dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

static bool write_model(const std::string & path) {
    gguf_context_ptr gguf_ctx(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN3MOE, gguf_ctx.get());

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(LLM_ARCH_QWEN3MOE));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                   uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,               uint32_t(256));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             uint32_t(128));
    ms.add_kv(LLM_KV_BLOCK_COUNT,                  uint32_t(2));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,         uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,      uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1e-5f);
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                 N_EXPERT);
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,            N_EXPERT_USED);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,              "no_vocab");

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devs[] = { nullptr };
    mp.devices = no_devs;

    size_t seed = SEED;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, mp));
    if (!model) {
        fprintf(stderr, "failed to create the test model\n");
        return false;
    }
    llama_model_save_to_file(model.get(), path.c_str());
    return true;
}

// ---- child: load the model with the environment it was given, write the logits of N_TOKENS tokens

static bool g_arena_seen = false;
static long g_streamed   = 0;

static int run_child(const std::string & model_path, const std::string & out_path, const std::string & dev_name, int n_ubatch) {
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (strstr(text, "MoE arena:")) {
            g_arena_seen = true;
        }
        const char * s = strstr(text, "hits, ");
        if (s) {
            g_streamed = atol(s + strlen("hits, "));
        }
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);

    // DEVICE is one name or a comma-separated list, which splits the layers over the devices
    std::vector<ggml_backend_dev_t> devs;
    bool cpu = false;
    for (const auto & name : string_split<std::string>(dev_name, ',')) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) {
            fprintf(stderr, "no device %s\n", name.c_str());
            return 1;
        }
        cpu = cpu || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        devs.push_back(dev);
    }
    // the CPU runs get no device at all, so that nothing is offloaded behind the test's back
    if (cpu) {
        devs.clear();
    }
    devs.push_back(nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = [](float, void *) { return true; };
    mp.devices      = devs.data();
    mp.n_gpu_layers = cpu ? 0 : 99;

    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mp));
    if (!model) {
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 256;
    cp.n_ubatch        = n_ubatch;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        return 1;
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 gen(SEED);
    llama_batch batch = llama_batch_init(N_TOKENS, 0, 1);
    for (uint32_t i = 0; i < N_TOKENS; ++i) {
        common_batch_add(batch, gen() % n_vocab, i, {0}, true);
    }
    const int rc = llama_decode(ctx.get(), batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "decode failed: %d\n", rc);
        return 1;
    }

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        return 1;
    }
    for (uint32_t i = 0; i < N_TOKENS; ++i) {
        fwrite(llama_get_logits_ith(ctx.get(), i), sizeof(float), n_vocab, f);
    }
    fclose(f);

    const char * direct = getenv("LLAMA_MOE_DIRECT");
    if (direct && atoi(direct) != 0) {
        if (!g_arena_seen) {
            fprintf(stderr, "LLAMA_MOE_DIRECT is set but the arena is not in use (no O_DIRECT on this file system?)\n");
            return 3;
        }
        if (g_streamed == 0) {
            fprintf(stderr, "the arena paged in no expert, the test does not exercise it\n");
            return 3;
        }
    }
    return 0;
}

// ---- parent

static std::string g_self;

static bool run(const std::string & env, const std::string & model, const std::string & out, const std::string & dev, int n_ubatch) {
    const std::string cmd = "env LLAMA_MOE_STATS=1 LLAMA_MOE_TRIM_EVERY=1 " + env + " '" + g_self + "' child '" + model + "' '" + out + "' '" + dev + "' " + std::to_string(n_ubatch);
    remove(out.c_str()); // a child that dies must not leave the previous run's logits to compare
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "FAIL: child exited with %d: %s\n", rc, cmd.c_str());
        return false;
    }
    return true;
}

static std::vector<float> read_floats(const std::string & path) {
    std::vector<float> v;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return v;
    }
    float x;
    while (fread(&x, sizeof(x), 1, f) == 1) {
        v.push_back(x);
    }
    fclose(f);
    return v;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0.0, a0 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) (a[i] - b[i])*(a[i] - b[i]);
        a0 += (double) a[i]*a[i];
    }
    return ab/a0;
}

// exact: logits must be bit-identical; otherwise within the NMSE tolerance
static bool compare(const char * what, const std::string & ref, const std::string & got, bool exact) {
    const auto a = read_floats(ref);
    const auto b = read_floats(got);
    if (a.empty() || a.size() != b.size()) {
        printf("  %-44s FAIL (no logits)\n", what);
        return false;
    }
    const bool same = memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
    const double e = nmse(a, b);
    const bool ok = exact ? same : e < 1e-12; // GPUs are bit-identical here too, save for ~1e-17
    printf("  %-44s %s  nmse %.2e%s\n", what, ok ? "OK  " : "FAIL", e, same ? " (bit-identical)" : "");
    return ok;
}

int main(int argc, char ** argv) {
#if !defined(__linux__)
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    printf("test-moe-tier: O_DIRECT arena is Linux only, skipped\n");
    return 0;
#else
    ggml_backend_load_all();

    if (argc == 6 && strcmp(argv[1], "child") == 0) {
        return run_child(argv[2], argv[3], argv[4], atoi(argv[5]));
    }

    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    GGML_ASSERT(n > 0);
    self[n] = '\0';
    g_self = self;

    const std::string dir   = argc > 1 ? argv[1] : ".";
    const std::string model = dir + "/test-moe-tier.gguf";
    auto out = [&](const std::string & tag) { return dir + "/test-moe-tier-" + tag + ".bin"; };

    if (!write_model(model)) {
        return 1;
    }

    // 16 experts, 4 used: 4 hot, 12 cold. The arena keeps 2 cold experts and pages the rest through 8 or 4 frames.
    // With 8 frames a ubatch of 2 tokens (8 routed slots) takes the decode path, with 4 frames a ubatch of 64
    // takes the windowed path: windows [0,6) [6,10) [10,12). With LLAMA_MOE_OVERLAP the windows are half as wide and
// alternate between the two halves of the frames, one paged in while the other computes: [0,4) [4,6) [6,8) [8,10) [10,12).
    const std::string hot     = "LLAMA_MOE_HOT=4";
    const std::string decode  = hot + " LLAMA_MOE_DIRECT=1 LLAMA_MOE_ARENA=2 LLAMA_MOE_FRAMES=8";
    const std::string windows = hot + " LLAMA_MOE_DIRECT=1 LLAMA_MOE_ARENA=2 LLAMA_MOE_FRAMES=4";
    const std::string overlap = windows + " LLAMA_MOE_OVERLAP=1";

    std::vector<std::string> devs;
    std::string gpus;
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            devs.push_back(ggml_backend_dev_name(d));
        }
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus += (n_gpu++ ? "," : "") + std::string(ggml_backend_dev_name(d));
        }
    }
    // layers split over several GPUs, the way the large models run
    if (n_gpu > 1) {
        devs.push_back(gpus);
    }

    bool ok = true;
    for (const auto & dev : devs) {
        const bool cpu = dev == "CPU";
        printf("%s:\n", dev.c_str());

        const std::string d = dev + "-";
        ok &= run("",      model, out(d + "ref-ub64"),     dev, 64);
        ok &= run("",      model, out(d + "ref-ub2"),      dev, 2);
        ok &= run(hot,     model, out(d + "hot-ub64"),     dev, 64);
        ok &= run(hot,     model, out(d + "hot-ub2"),      dev, 2);
        ok &= run(decode,  model, out(d + "decode-ub2"),   dev, 2);
        ok &= run(windows, model, out(d + "windows-ub64"), dev, 64);
        ok &= run(overlap, model, out(d + "overlap-ub64"), dev, 64);

        // the split into hot and cold is exact on the CPU; on a GPU, cold experts run on the host or are offloaded
        ok &= compare("hot/cold split, ubatch 64",               out(d + "ref-ub64"), out(d + "hot-ub64"),     cpu);
        ok &= compare("hot/cold split, ubatch 2",                out(d + "ref-ub2"),  out(d + "hot-ub2"),      cpu);
        ok &= compare("arena, decode path (frames round-robin)", out(d + "hot-ub2"),  out(d + "decode-ub2"),   cpu);
        ok &= compare("arena, windowed path",                    out(d + "hot-ub64"), out(d + "windows-ub64"), cpu);
        ok &= compare("arena, windowed path, read overlapped",   out(d + "hot-ub64"), out(d + "overlap-ub64"), cpu);
    }

    // tools/moe-tier permute: reordering the experts must not change the model
    const char * permute = getenv("TEST_MOE_TIER_PERMUTE");
    if (permute && *permute) {
        const std::string permuted = dir + "/test-moe-tier-perm.gguf";
        const std::string cmd = std::string(permute) + " '" + model + "' '" + permuted + "'";
        printf("permute:\n");
        if (system(cmd.c_str()) != 0) {
            printf("  %-44s FAIL: %s\n", "moe-tier.py permute", cmd.c_str());
            ok = false;
        } else {
            ok &= run("", permuted, out("CPU-perm-ub64"), "CPU", 64);
            ok &= compare("permuted experts, CPU", out("CPU-ref-ub64"), out("CPU-perm-ub64"), true);
        }
    } else {
        printf("permute: skipped, set TEST_MOE_TIER_PERMUTE (see tests/test-moe-tier-permute.sh)\n");
    }

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
#endif
}
