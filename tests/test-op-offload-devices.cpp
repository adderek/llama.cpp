// Fork regression test: GGML_CUDA_OP_OFFLOAD_DEVICES limits op offload to the listed devices.
//
// The scheduler offloads a large-batch op whose weights sit in host RAM to the FIRST device
// that accepts it. On a box whose device 0 is on a narrow chipset link that sends every MoE
// expert weight over the slow bus; the variable lets the fast card take the work instead.
// Without CUDA/HIP devices there is nothing to check and the test passes.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    // must be set before the CUDA backend registers its devices
    setenv("GGML_CUDA_OP_OFFLOAD_DEVICES", "1", 1);

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("CUDA", 0) == 0 || name.rfind("ROCm", 0) == 0) {
            gpus.push_back(dev);
        }
    }
    if (gpus.empty()) {
        fprintf(stderr, "no CUDA/HIP devices, skipping\n");
        return 0;
    }

    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 512); // batch 512, above the default 32
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    bool ok = true;
    for (size_t i = 0; i < gpus.size(); ++i) {
        const bool want = i == 1;
        const bool got  = ggml_backend_dev_offload_op(gpus[i], y);
        fprintf(stderr, "%s: offload %s (want %s)\n", ggml_backend_dev_name(gpus[i]), got ? "yes" : "no", want ? "yes" : "no");
        ok = ok && got == want;
    }

    ggml_free(ctx);
    return ok ? 0 : 1;
}
