// Fork regression test: TOP_K over many long rows must survive CUDA/HIP graph capture.
//
// The fork used to build HIP with hipcub. Under a graph capture, argsort/top-k then took
// rocPRIM's segmented radix sort, which ROCm rejects while a stream is capturing ("operation
// not permitted when stream is capturing") and ggml aborted. It hit qwen4exp decode: the
// QSA indexer takes top-k over the whole KV length for every head. The same graph is
// computed repeatedly here because the CUDA backend only captures once it has seen the graph
// warm up. Without a CUDA/HIP device the test passes.
//
// Default shape 256 x 4112, k 512: with hipcub it aborted at >= 256 rows and > 1024 columns
// (swept 2048..65536 columns, 1..1024 rows). TK_COLS / TK_ROWS / TK_K override it.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
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
    ggml_backend_t backend = ggml_backend_dev_init(gpu, nullptr);

    const int64_t n_cols = getenv("TK_COLS") ? atoll(getenv("TK_COLS")) : 4112;
    const int64_t n_rows = getenv("TK_ROWS") ? atoll(getenv("TK_ROWS")) : 256; // rocPRIM changes algorithm at 256 segments; that one aborts
    const int     k      = getenv("TK_K")    ? atoi(getenv("TK_K"))     : 512;

    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows);
    ggml_tensor * y = ggml_top_k(ctx, x, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> data(n_cols*n_rows);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (float) ((i*2654435761u) % 100003) / 100003.0f;
    }
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));

    for (int it = 0; it < 8; ++it) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed at iteration %d\n", it);
            return 1;
        }
    }

    // the result must be the k largest of each row
    std::vector<int32_t> idx(k*n_rows);
    ggml_backend_tensor_get(y, idx.data(), 0, ggml_nbytes(y));
    for (int64_t r = 0; r < n_rows; ++r) {
        const float * row = data.data() + r*n_cols;
        float min_top = 1e30f;
        for (int j = 0; j < k; ++j) {
            min_top = std::min(min_top, row[idx[r*k + j]]);
        }
        int64_t n_above = 0;
        for (int64_t c = 0; c < n_cols; ++c) {
            n_above += row[c] > min_top;
        }
        if (n_above > k) {
            fprintf(stderr, "row %lld: %lld values exceed the smallest of the top %d\n", (long long) r, (long long) n_above, k);
            return 1;
        }
    }

    fprintf(stderr, "%s: top-k of %lldx%lld survived 8 graph computes\n", ggml_backend_name(backend), (long long) n_rows, (long long) n_cols);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
