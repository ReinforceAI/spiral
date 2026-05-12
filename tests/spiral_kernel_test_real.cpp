// spiral_kernel_test_real.cpp
// ============================
//
// Run kernel on REAL production bytes:
//   - rotated input from a spiral_dump file
//   - weight bytes pre-extracted by extract_weight.py
//
// Args:
//   1. dump_bin       — input from spiral_dump (has 128-byte header)
//   2. weight_bin     — raw weight bytes (no header, packed SPIRAL_3BIT)
//   3. k              — input dim (= weight in_features)
//   4. m              — output dim (= weight out_features)
//   5. output_path    — where to write y_kernel.bin (raw F32)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        exit(1);
    }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read((char *)buf.data(), sz);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <dump_bin> <weight_bin> <k> <m> <output_path>\n", argv[0]);
        return 1;
    }

    std::string dump_path = argv[1];
    std::string weight_path = argv[2];
    int64_t k = std::stoll(argv[3]);
    int64_t m = std::stoll(argv[4]);
    std::string output_path = argv[5];

    // ── 1. Read input dump (has 128-byte header) ───────────────────────
    auto dump_buf = read_file(dump_path);
    if (dump_buf.size() < 128) {
        fprintf(stderr, "ERROR: dump file too small (%zu bytes)\n", dump_buf.size());
        return 1;
    }

    int64_t ne0_dump, ne1_dump;
    memcpy(&ne0_dump, dump_buf.data() + 48, 8);
    memcpy(&ne1_dump, dump_buf.data() + 56, 8);

    if (ne0_dump != k) {
        fprintf(stderr, "ERROR: dump dim ne0=%lld != provided k=%lld\n",
                (long long)ne0_dump, (long long)k);
        return 1;
    }

    int64_t n = ne1_dump;
    fprintf(stderr, "Dump: k=%lld n=%lld (n_tokens)\n", (long long)k, (long long)n);

    const float * x_data = (const float *)(dump_buf.data() + 128);
    int64_t x_bytes = (int64_t)dump_buf.size() - 128;
    int64_t expected_x_bytes = k * n * 4;
    if (x_bytes != expected_x_bytes) {
        fprintf(stderr, "ERROR: input size mismatch %lld != %lld\n",
                (long long)x_bytes, (long long)expected_x_bytes);
        return 1;
    }

    // ── 2. Read weight (raw, no header) ────────────────────────────────
    auto weight_bytes = read_file(weight_path);
    int64_t expected_w_bytes = m * (k / 128) * 50;
    if ((int64_t)weight_bytes.size() != expected_w_bytes) {
        fprintf(stderr, "ERROR: weight size %zu != expected %lld (m=%lld, k=%lld)\n",
                weight_bytes.size(), (long long)expected_w_bytes,
                (long long)m, (long long)k);
        return 1;
    }
    fprintf(stderr, "Weight: m=%lld, %zu bytes\n", (long long)m, weight_bytes.size());

    // ── 3. Init GPU backend (CUDA on Linux, Metal on Mac) ──────────────
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "ERROR: no GPU backend device available\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "ERROR: failed to init GPU backend\n");
        return 1;
    }
    fprintf(stderr, "GPU backend initialized: %s\n", ggml_backend_dev_name(dev));

    // ── 4. Build graph ─────────────────────────────────────────────────
    struct ggml_init_params iparams = { 1ull * 1024 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(iparams);

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_SPIRAL_3BIT, k, m);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_name(w, "w");
    ggml_set_name(x, "x");

    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(y, "y");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "ERROR: failed to allocate backend buffers\n");
        return 1;
    }

    size_t w_expected = ggml_nbytes(w);
    size_t x_expected = ggml_nbytes(x);

    if (w_expected != weight_bytes.size()) {
        fprintf(stderr, "WARNING: ggml expects w=%zu bytes, got %zu\n",
                w_expected, weight_bytes.size());
    }

    ggml_backend_tensor_set(w, weight_bytes.data(), 0,
                            std::min(w_expected, weight_bytes.size()));
    ggml_backend_tensor_set(x, x_data, 0,
                            std::min(x_expected, (size_t)x_bytes));

    fprintf(stderr, "Uploaded inputs\n");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: compute failed (%d)\n", (int)status);
        return 1;
    }
    fprintf(stderr, "Compute done\n");

    size_t y_bytes = ggml_nbytes(y);
    std::vector<uint8_t> y_data(y_bytes);
    ggml_backend_tensor_get(y, y_data.data(), 0, y_bytes);

    std::ofstream out(output_path, std::ios::binary);
    out.write((const char *)y_data.data(), y_data.size());

    fprintf(stderr, "Wrote %s (%zu bytes), shape (n=%lld, m=%lld)\n",
            output_path.c_str(), y_data.size(), (long long)n, (long long)m);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
