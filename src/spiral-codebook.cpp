// Spiral PQ codebook loader — reads .spiralcb binary file
// and creates GPU tensors for use by PQ encode/decode kernels.

#include "spiral-codebook.h"
#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <memory>

size_t spiral_codebook_data::total_bytes() const {
    if (!loaded) return 0;
    size_t total = 0;
    // Rotation matrices: 2 × 128 × 128 × 4 = 128 KB
    total += 2 * head_dim * head_dim * sizeof(float);
    // Per-layer: K codebooks + V codebooks + K means
    size_t per_layer_cb = n_kv_heads * n_blocks * n_codewords * pq_block_size * sizeof(float);
    size_t per_layer_means = n_kv_heads * head_dim * sizeof(float);
    total += n_layers * (2 * per_layer_cb + per_layer_means);
    return total;
}

bool spiral_codebook_load(
    const std::string & path,
    spiral_codebook_data & data,
    ggml_context * ctx,
    ggml_backend_buffer_type_t buft) {

    fprintf(stderr, "SPIRAL_CB: opening %s\n", path.c_str());

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open codebook file: %s\n", __func__, path.c_str());
        return false;
    }

    // Read header (struct is smaller than 64-byte file header due to padding)
    spiral_codebook_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        LLAMA_LOG_ERROR("%s: failed to read header from %s\n", __func__, path.c_str());
        fclose(f);
        return false;
    }
    // Skip remaining padding bytes to reach 64-byte file header boundary
    if (sizeof(hdr) < 64) {
        fseek(f, 64 - sizeof(hdr), SEEK_CUR);
    }

    fprintf(stderr, "SPIRAL_CB: header read, magic=%.8s version=%u\n", hdr.magic, hdr.version);

    // Validate magic
    if (memcmp(hdr.magic, "SPIRALCB", 8) != 0) {
        LLAMA_LOG_ERROR("%s: invalid magic in %s (expected SPIRALCB)\n", __func__, path.c_str());
        fclose(f);
        return false;
    }

    if (hdr.version != 1 && hdr.version != 2) {
        LLAMA_LOG_ERROR("%s: unsupported version %u in %s\n", __func__, hdr.version, path.c_str());
        fclose(f);
        return false;
    }

    data.n_layers      = hdr.n_layers;
    data.n_kv_heads    = hdr.n_kv_heads;
    data.head_dim      = hdr.head_dim;
    data.n_blocks      = hdr.n_blocks;
    data.n_codewords   = hdr.n_codewords;
    data.pq_block_size = hdr.pq_block_size;

    fprintf(stderr, "SPIRAL_CB: %u layers, %u heads, %u dim, %u blocks, %u codewords\n",
        data.n_layers, data.n_kv_heads, data.head_dim, data.n_blocks, data.n_codewords);

    LLAMA_LOG_INFO("%s: loading codebooks from %s\n", __func__, path.c_str());
    LLAMA_LOG_INFO("%s:   %u layers, %u KV heads, %u head_dim\n", __func__,
        data.n_layers, data.n_kv_heads, data.head_dim);
    LLAMA_LOG_INFO("%s:   PQ: %u blocks × %u codewords × %u dims\n", __func__,
        data.n_blocks, data.n_codewords, data.pq_block_size);

    // Sizes
    const size_t rot_size = data.head_dim * data.head_dim * sizeof(float);
    const size_t cb_size  = data.n_kv_heads * data.n_blocks * data.n_codewords * data.pq_block_size * sizeof(float);
    const size_t mean_size = data.n_kv_heads * data.head_dim * sizeof(float);

    // Allocate temp buffer for reading
    const size_t max_chunk = std::max({rot_size, cb_size, mean_size});
    float * tmp_raw = (float *)malloc(max_chunk);
    if (!tmp_raw) {
        LLAMA_LOG_ERROR("%s: failed to allocate temp buffer (%zu bytes)\n", __func__, max_chunk);
        fclose(f);
        return false;
    }

    // --- Create rotation tensors ---
    fprintf(stderr, "SPIRAL_CB: creating rotation tensors\n");
    data.R_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.head_dim, data.head_dim);
    ggml_format_name(data.R_kv, "spiral_cb_R_kv");

    data.R_kv_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.head_dim, data.head_dim);
    ggml_format_name(data.R_kv_inv, "spiral_cb_R_kv_inv");

    // --- Create per-layer tensors ---
    fprintf(stderr, "SPIRAL_CB: creating %u layer tensors\n", data.n_layers);
    data.layers.resize(data.n_layers);
    for (uint32_t li = 0; li < data.n_layers; li++) {
        auto & layer = data.layers[li];

        layer.k_codebooks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
            data.n_kv_heads * data.n_blocks * data.n_codewords * data.pq_block_size);
        ggml_set_name(layer.k_codebooks, "spiral_cb_k");

        layer.v_codebooks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
            data.n_kv_heads * data.n_blocks * data.n_codewords * data.pq_block_size);
        ggml_set_name(layer.v_codebooks, "spiral_cb_v");

        layer.k_means = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
            data.n_kv_heads * data.head_dim);
        ggml_set_name(layer.k_means, "spiral_cb_km");
    }

    // --- Allocate buffer for all tensors ---
    fprintf(stderr, "SPIRAL_CB: allocating buffer\n");
    data.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!data.buf) {
        LLAMA_LOG_ERROR("%s: failed to allocate buffer for codebook tensors\n", __func__);
        fclose(f);
        return false;
    }

    LLAMA_LOG_INFO("%s: codebook buffer size = %.2f MiB\n", __func__,
        ggml_backend_buffer_get_size(data.buf) / 1024.0 / 1024.0);

    fprintf(stderr, "SPIRAL_CB: uploading rotation matrices\n");
    fprintf(stderr, "SPIRAL_CB: R_kv tensor: data=%p buffer=%p nbytes=%zu rot_size=%zu\n",
        (void*)data.R_kv->data, (void*)data.R_kv->buffer, ggml_nbytes(data.R_kv), rot_size);

    // --- Read and upload rotation matrices ---
    size_t bytes_read = fread(tmp_raw, 1, rot_size, f);
    fprintf(stderr, "SPIRAL_CB: fread R_kv: %zu / %zu bytes\n", bytes_read, rot_size);
    if (bytes_read != rot_size) {
        LLAMA_LOG_ERROR("%s: failed to read R_kv\n", __func__);
        fclose(f);
        return false;
    }
    fprintf(stderr, "SPIRAL_CB: calling ggml_backend_tensor_set for R_kv\n");
    ggml_backend_tensor_set(data.R_kv, tmp_raw, 0, rot_size);
    fprintf(stderr, "SPIRAL_CB: R_kv uploaded OK\n");

    fprintf(stderr, "SPIRAL_CB: reading R_kv_inv from file\n");
    if (fread(tmp_raw, 1, rot_size, f) != rot_size) {
        LLAMA_LOG_ERROR("%s: failed to read R_kv_inv\n", __func__);
        fclose(f);
        return false;
    }
    fprintf(stderr, "SPIRAL_CB: R_kv_inv tensor: data=%p buffer=%p nbytes=%zu\n",
        (void*)data.R_kv_inv->data, (void*)data.R_kv_inv->buffer, ggml_nbytes(data.R_kv_inv));
    ggml_backend_tensor_set(data.R_kv_inv, tmp_raw, 0, rot_size);
    fprintf(stderr, "SPIRAL_CB: R_kv_inv uploaded OK\n");

    // --- Read and upload per-layer data ---
    fprintf(stderr, "SPIRAL_CB: uploading per-layer data (cb_size=%zu mean_size=%zu)\n", cb_size, mean_size);
    for (uint32_t li = 0; li < data.n_layers; li++) {
        auto & layer = data.layers[li];

        fprintf(stderr, "SPIRAL_CB: layer %u: k_cb data=%p buffer=%p nbytes=%zu\n",
            li, (void*)layer.k_codebooks->data, (void*)layer.k_codebooks->buffer,
            ggml_nbytes(layer.k_codebooks));

        // K codebooks
        size_t r = fread(tmp_raw, 1, cb_size, f);
        fprintf(stderr, "SPIRAL_CB: layer %u: fread K cb: %zu / %zu\n", li, r, cb_size);
        if (r != cb_size) {
            LLAMA_LOG_ERROR("%s: failed to read K codebooks for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        fprintf(stderr, "SPIRAL_CB: layer %u: tensor_set K cb\n", li);
        ggml_backend_tensor_set(layer.k_codebooks, tmp_raw, 0, cb_size);
        fprintf(stderr, "SPIRAL_CB: layer %u: K cb OK\n", li);

        // V codebooks
        fprintf(stderr, "SPIRAL_CB: layer %u: V cb\n", li);
        if (fread(tmp_raw, 1, cb_size, f) != cb_size) {
            LLAMA_LOG_ERROR("%s: failed to read V codebooks for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(layer.v_codebooks, tmp_raw, 0, cb_size);
        fprintf(stderr, "SPIRAL_CB: layer %u: V cb OK\n", li);

        // K mean carriers
        fprintf(stderr, "SPIRAL_CB: layer %u: means\n", li);
        if (fread(tmp_raw, 1, mean_size, f) != mean_size) {
            LLAMA_LOG_ERROR("%s: failed to read K means for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(layer.k_means, tmp_raw, 0, mean_size);
        fprintf(stderr, "SPIRAL_CB: layer %u: means OK\n", li);
    }

    // --- Try to read weight rotation extension ---
    // Extension starts with "SPIRALRT" magic. If not present, skip (backward compat).
    {
        char rot_magic[8] = {0};
        size_t n_read = fread(rot_magic, 1, 8, f);
        if (n_read == 8 && memcmp(rot_magic, "SPIRALRT", 8) == 0) {
            uint32_t n_dims = 0;
            fread(&n_dims, sizeof(uint32_t), 1, f);
            fseek(f, 28, SEEK_CUR);  // skip 28 bytes padding (40-byte total header)

            fprintf(stderr, "SPIRAL_CB: reading weight rotation extension (%u dims)\n", n_dims);

            // Pass 1: read just dim/n_passes headers, skip data to learn layout
            long data_body_pos = ftell(f);
            data.weight_rotations.resize(n_dims);

            for (uint32_t di = 0; di < n_dims; di++) {
                uint32_t dim = 0, n_passes = 0;
                fread(&dim, sizeof(uint32_t), 1, f);
                fread(&n_passes, sizeof(uint32_t), 1, f);

                data.weight_rotations[di].dim = (int32_t)dim;
                data.weight_rotations[di].n_passes = (int32_t)n_passes;

                fprintf(stderr, "SPIRAL_CB: rot dim=%u n_passes=%u\n", dim, n_passes);

                // Skip past signs + perms + inv_perms data
                size_t signs_total = (size_t)n_passes * dim * sizeof(float);
                size_t perms_total = (size_t)(n_passes - 1) * dim * sizeof(int64_t) * 2; // perms are int64 in file
                fseek(f, (long)(signs_total + perms_total), SEEK_CUR);
            }

            // Count tensors needed
            size_t rot_n_tensors = 0;
            for (uint32_t di = 0; di < n_dims; di++) {
                uint32_t np = data.weight_rotations[di].n_passes;
                rot_n_tensors += np;           // signs
                rot_n_tensors += (np - 1) * 2; // perms + inv_perms
            }

            size_t rot_ctx_size = rot_n_tensors * ggml_tensor_overhead() + 1024;
            struct ggml_init_params rot_ctx_params = {
                /*.mem_size   =*/ rot_ctx_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * rot_ctx = ggml_init(rot_ctx_params);

            // Create all tensors
            for (uint32_t di = 0; di < n_dims; di++) {
                auto & rot = data.weight_rotations[di];
                uint32_t dim = rot.dim;
                uint32_t np  = rot.n_passes;

                rot.signs1 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_F32, dim);
                ggml_format_name(rot.signs1, "spiral_rot_s1_d%u", dim);
                rot.signs2 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_F32, dim);
                ggml_format_name(rot.signs2, "spiral_rot_s2_d%u", dim);
                if (np >= 3) {
                    rot.signs3 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_F32, dim);
                    ggml_format_name(rot.signs3, "spiral_rot_s3_d%u", dim);
                }
                rot.perm1 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_I32, dim);
                ggml_format_name(rot.perm1, "spiral_rot_p1_d%u", dim);
                rot.inv_perm1 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_I32, dim);
                ggml_format_name(rot.inv_perm1, "spiral_rot_ip1_d%u", dim);
                if (np >= 3) {
                    rot.perm2 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_I32, dim);
                    ggml_format_name(rot.perm2, "spiral_rot_p2_d%u", dim);
                    rot.inv_perm2 = ggml_new_tensor_1d(rot_ctx, GGML_TYPE_I32, dim);
                    ggml_format_name(rot.inv_perm2, "spiral_rot_ip2_d%u", dim);
                }
            }

            // Allocate GPU buffer
            data.rot_buf = ggml_backend_alloc_ctx_tensors_from_buft(rot_ctx, buft);
            data.rot_ctx = rot_ctx;
            if (!data.rot_buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate rotation tensor buffer\n", __func__);
                ggml_free(rot_ctx);
                data.rot_ctx = nullptr;
            } else {
                // Pass 2: seek back and read + upload data
                fseek(f, data_body_pos, SEEK_SET);

                for (uint32_t di = 0; di < n_dims; di++) {
                    auto & rot = data.weight_rotations[di];
                    uint32_t dim = rot.dim;
                    uint32_t np  = rot.n_passes;

                    // Skip dim + n_passes header
                    uint32_t skip_dim, skip_np;
                    fread(&skip_dim, sizeof(uint32_t), 1, f);
                    fread(&skip_np, sizeof(uint32_t), 1, f);

                    size_t sz = (size_t)dim * sizeof(float);

                    // Ensure tmp buffer is large enough
                    if (sz > max_chunk) {
                        tmp_raw = (float *)realloc(tmp_raw, sz);
                    }

                    // Signs
                    fread(tmp_raw, 1, sz, f);
                    ggml_backend_tensor_set(rot.signs1, tmp_raw, 0, sz);
                    fread(tmp_raw, 1, sz, f);
                    ggml_backend_tensor_set(rot.signs2, tmp_raw, 0, sz);
                    if (np >= 3) {
                        fread(tmp_raw, 1, sz, f);
                        ggml_backend_tensor_set(rot.signs3, tmp_raw, 0, sz);
                    }

                    // Perms (stored as int64 in file, need int32 for tensor)
                    size_t sz64 = (size_t)dim * sizeof(int64_t);
                    if (sz64 > max_chunk * sizeof(float)) {
                        tmp_raw = (float *)realloc(tmp_raw, sz64);
                    }
                    {
                        // Read int64 perms, convert to int32
                        int64_t * tmp64 = (int64_t *)tmp_raw;
                        int32_t * tmp32 = (int32_t *)tmp_raw;  // reuse same buffer, write in-place
                        fread(tmp64, 1, sz64, f);
                        for (uint32_t i = 0; i < dim; i++) tmp32[i] = (int32_t)tmp64[i];
                        ggml_backend_tensor_set(rot.perm1, tmp32, 0, sz);
                    }
                    if (np >= 3) {
                        int64_t * tmp64 = (int64_t *)tmp_raw;
                        int32_t * tmp32 = (int32_t *)tmp_raw;
                        fread(tmp64, 1, sz64, f);
                        for (uint32_t i = 0; i < dim; i++) tmp32[i] = (int32_t)tmp64[i];
                        ggml_backend_tensor_set(rot.perm2, tmp32, 0, sz);
                    }

                    // Inv perms (also int64 in file)
                    {
                        int64_t * tmp64 = (int64_t *)tmp_raw;
                        int32_t * tmp32 = (int32_t *)tmp_raw;
                        fread(tmp64, 1, sz64, f);
                        for (uint32_t i = 0; i < dim; i++) tmp32[i] = (int32_t)tmp64[i];
                        ggml_backend_tensor_set(rot.inv_perm1, tmp32, 0, sz);
                    }
                    if (np >= 3) {
                        int64_t * tmp64 = (int64_t *)tmp_raw;
                        int32_t * tmp32 = (int32_t *)tmp_raw;
                        fread(tmp64, 1, sz64, f);
                        for (uint32_t i = 0; i < dim; i++) tmp32[i] = (int32_t)tmp64[i];
                        ggml_backend_tensor_set(rot.inv_perm2, tmp32, 0, sz);
                    }

                    fprintf(stderr, "SPIRAL_CB: weight rotation D=%u (%u passes) loaded\n", dim, np);
                }

                LLAMA_LOG_INFO("%s: weight rotation buffer = %.2f KiB\n", __func__,
                    ggml_backend_buffer_get_size(data.rot_buf) / 1024.0);
            }
        } else {
            fprintf(stderr, "SPIRAL_CB: no weight rotation extension found (optional)\n");
        }
    }

    fprintf(stderr, "SPIRAL_CB: closing file\n");
    fflush(stderr);
    fclose(f);

    data.loaded = true;
    fprintf(stderr, "SPIRAL_CB: load complete\n");
    fflush(stderr);

    free(tmp_raw);

    return true;
}