// spiral_kernel_test_moe.cpp
// ===========================
//
// Standalone tester for kernel_mul_mv_id_spiral_3bit_f32 (the MoE variant).
//
// Build: cmake --build build --target spiral_kernel_test_moe
//
// Run:
//   python tools/spiral_test_gen_moe.py --m 512 --k 2048 --n_tokens 1 \
//       --n_experts 256 --n_active 8 --out /tmp/fixture_moe/
//   ./build/bin/spiral_kernel_test_moe /tmp/fixture_moe/
//   python tools/verify_spiral_kernel.py /tmp/fixture_moe/

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

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

static int64_t read_meta_int(const std::string & meta_path, const std::string & key) {
    std::ifstream f(meta_path);
    std::string line;
    std::string needle = "\"" + key + "\":";
    while (std::getline(f, line)) {
        size_t p = line.find(needle);
        if (p == std::string::npos) continue;
        p = line.find_first_of("0123456789-", p + needle.size());
        if (p == std::string::npos) continue;
        size_t e = line.find_first_not_of("0123456789-", p);
        return std::stoll(line.substr(p, e - p));
    }
    fprintf(stderr, "ERROR: key '%s' not found\n", key.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <fixture_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];

    int64_t m         = read_meta_int(dir + "/meta.json", "m");
    int64_t k         = read_meta_int(dir + "/meta.json", "k");
    int64_t n_tokens  = read_meta_int(dir + "/meta.json", "n_tokens");
    int64_t n_experts = read_meta_int(dir + "/meta.json", "n_experts");
    int64_t n_active  = read_meta_int(dir + "/meta.json", "n_active");

    fprintf(stderr, "Fixture: m=%lld k=%lld n_tokens=%lld n_experts=%lld n_active=%lld\n",
            (long long)m, (long long)k, (long long)n_tokens,
            (long long)n_experts, (long long)n_active);

    auto weight_buf = read_file(dir + "/weight_packed.bin");
    auto x_buf      = read_file(dir + "/x_rotated.bin");
    auto ids_buf    = read_file(dir + "/ids.bin");

    fprintf(stderr, "Read: weight=%zu, x=%zu, ids=%zu bytes\n",
            weight_buf.size(), x_buf.size(), ids_buf.size());

    // Init Metal
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "ERROR: Metal init failed\n"); return 1; }

    struct ggml_init_params iparams = { 1ull * 1024 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(iparams);

    // Tensors:
    //   w:   [k, m, n_experts]    SPIRAL_3BIT
    //   x:   [k, 1, n_tokens]     F32  (3D for mul_mat_id)
    //   ids: [n_active, n_tokens] I32
    //
    // mul_mat_id signature:
    //   ggml_mul_mat_id(ctx, w, x, ids)
    //   -> output: [m, n_active, n_tokens]

    struct ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_SPIRAL_3BIT, k, m, n_experts);
    struct ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_active, n_tokens);
    ggml_set_name(w,   "w");
    ggml_set_name(x,   "x");
    ggml_set_name(ids, "ids");

    struct ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_set_name(y, "y");

    fprintf(stderr, "Output shape: [%lld, %lld, %lld]\n",
            (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2]);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "ERROR: alloc failed\n"); return 1; }

    size_t w_expected   = ggml_nbytes(w);
    size_t x_expected   = ggml_nbytes(x);
    size_t ids_expected = ggml_nbytes(ids);

    if (w_expected != weight_buf.size())
        fprintf(stderr, "WARN: w expected %zu got %zu\n", w_expected, weight_buf.size());
    if (x_expected != x_buf.size())
        fprintf(stderr, "WARN: x expected %zu got %zu\n", x_expected, x_buf.size());
    if (ids_expected != ids_buf.size())
        fprintf(stderr, "WARN: ids expected %zu got %zu\n", ids_expected, ids_buf.size());

    ggml_backend_tensor_set(w,   weight_buf.data(), 0, std::min(w_expected, weight_buf.size()));
    ggml_backend_tensor_set(x,   x_buf.data(),      0, std::min(x_expected, x_buf.size()));
    ggml_backend_tensor_set(ids, ids_buf.data(),    0, std::min(ids_expected, ids_buf.size()));
    fprintf(stderr, "Inputs uploaded\n");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: compute failed (%d)\n", (int)status);
        return 1;
    }
    fprintf(stderr, "Compute done\n");

    // ggml_mul_mat_id output is [m, n_active, n_tokens]
    // The reference y_reference has shape [n_tokens, m] (already summed over experts)
    // We need to: sum y_kernel over n_active dim before comparing.
    //
    // For now, write the raw kernel output. Verification script handles it.

    size_t y_bytes = ggml_nbytes(y);
    std::vector<uint8_t> y_data(y_bytes);
    ggml_backend_tensor_get(y, y_data.data(), 0, y_bytes);

    std::ofstream out(dir + "/y_kernel_raw.bin", std::ios::binary);
    out.write((const char *)y_data.data(), y_data.size());
    out.close();

    fprintf(stderr, "Wrote y_kernel_raw.bin: %zu bytes, shape [%lld, %lld, %lld]\n",
            y_data.size(),
            (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2]);

    // Sum over n_active and write final y_kernel.bin
    // Layout of y_data: F32, [n_tokens][n_active][m] in row-major
    // (since shape is [m, n_active, n_tokens], with m innermost)
    const float * raw = (const float *)y_data.data();
    std::vector<float> summed((size_t)n_tokens * m, 0.0f);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t a = 0; a < n_active; ++a) {
            for (int64_t mi = 0; mi < m; ++mi) {
                size_t src_idx = (size_t)t * n_active * m + (size_t)a * m + (size_t)mi;
                size_t dst_idx = (size_t)t * m + (size_t)mi;
                summed[dst_idx] += raw[src_idx];
            }
        }
    }
    std::ofstream out2(dir + "/y_kernel.bin", std::ios::binary);
    out2.write((const char *)summed.data(), summed.size() * sizeof(float));
    fprintf(stderr, "Wrote y_kernel.bin: %zu bytes (summed over %lld active experts)\n",
            summed.size() * sizeof(float), (long long)n_active);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
