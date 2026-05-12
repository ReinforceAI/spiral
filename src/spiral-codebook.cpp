// Spiral PQ codebook loader — reads .spiralcb binary file
// and creates GPU tensors for use by PQ encode/decode kernels.

#include "spiral-codebook.h"
#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <cmath>
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


    // Validate magic
    if (memcmp(hdr.magic, "SPIRALCB", 8) != 0) {
        LLAMA_LOG_ERROR("%s: invalid magic in %s (expected SPIRALCB)\n", __func__, path.c_str());
        fclose(f);
        return false;
    }

    if (hdr.version != 1 && hdr.version != 2 && hdr.version != 3 && hdr.version != 4) {
        LLAMA_LOG_ERROR("%s: unsupported version %u in %s\n", __func__, hdr.version, path.c_str());
        fclose(f);
        return false;
    }

    data.version       = hdr.version;
    data.n_layers      = hdr.n_layers;
    data.n_kv_heads    = hdr.n_kv_heads;
    data.head_dim      = hdr.head_dim;
    data.n_blocks      = hdr.n_blocks;
    data.n_codewords   = hdr.n_codewords;
    data.pq_block_size = hdr.pq_block_size;


    LLAMA_LOG_INFO("%s: loading codebooks from %s\n", __func__, path.c_str());
    LLAMA_LOG_INFO("%s:   %u layers, %u KV heads, %u head_dim\n", __func__,
        data.n_layers, data.n_kv_heads, data.head_dim);
    LLAMA_LOG_INFO("%s:   PQ: %u blocks × %u codewords × %u dims\n", __func__,
        data.n_blocks, data.n_codewords, data.pq_block_size);

    // ── v3/v4 hybrid layer support (NO-OP for v1/v2) ─────────────────────
    // v3 introduced hybrid models (Qwen3.5-MoE: only some layers have KV codebooks).
    // v4 inherits the same header layout — n_attention_layers at byte offset 36 of
    // the 64-byte header, followed by the attention_layer_indices table.
    // For v1/v2, that slot is zero in the reserved padding, so we explicitly gate
    // this entire block on version >= 3 and never touch the v1/v2 byte-stream.
    uint32_t n_attention_layers = 0;
    if (hdr.version >= 3) {
        long save_pos = ftell(f);
        if (fseek(f, 36, SEEK_SET) != 0 ||
            fread(&n_attention_layers, sizeof(uint32_t), 1, f) != 1) {
            LLAMA_LOG_ERROR("%s: failed to read v3/v4 n_attention_layers field\n", __func__);
            fclose(f);
            return false;
        }
        // Restore position: we're at byte 64 (end of header) for the next read.
        if (fseek(f, save_pos, SEEK_SET) != 0) {
            LLAMA_LOG_ERROR("%s: failed to restore file position after v3/v4 header read\n", __func__);
            fclose(f);
            return false;
        }
        // Validate
        if (n_attention_layers > hdr.n_layers) {
            LLAMA_LOG_ERROR("%s: v%u header has n_attention_layers=%u > n_layers=%u\n",
                __func__, hdr.version, n_attention_layers, hdr.n_layers);
            fclose(f);
            return false;
        }
        // For hybrid models (n_attention_layers > 0), read the index table.
        if (n_attention_layers > 0) {
            data.attention_layer_indices.resize(n_attention_layers);
            size_t idx_bytes = (size_t)n_attention_layers * sizeof(uint32_t);
            if (fread(data.attention_layer_indices.data(), 1, idx_bytes, f) != idx_bytes) {
                LLAMA_LOG_ERROR("%s: failed to read attention-layer index table\n", __func__);
                fclose(f);
                return false;
            }
            // Validate each index
            for (uint32_t i = 0; i < n_attention_layers; i++) {
                if (data.attention_layer_indices[i] >= hdr.n_layers) {
                    LLAMA_LOG_ERROR("%s: attention_layer_indices[%u]=%u out of range\n",
                        __func__, i, data.attention_layer_indices[i]);
                    fclose(f);
                    return false;
                }
            }
            LLAMA_LOG_INFO("%s:   v%u hybrid: %u of %u layers have KV codebooks\n",
                __func__, hdr.version, n_attention_layers, hdr.n_layers);
        }
    }

    // For uniform models (v1/v2, or v3/v4 with n_attention_layers==0), every layer has codebooks.
    // For hybrid v3/v4 (n_attention_layers > 0), only attention layers do.
    const bool is_hybrid = (hdr.version >= 3) && (n_attention_layers > 0);
    const uint32_t n_codebook_layers = is_hybrid ? n_attention_layers : data.n_layers;

    // Sizes
    const size_t rot_size = data.head_dim * data.head_dim * sizeof(float);
    const size_t cb_size  = data.n_kv_heads * data.n_blocks * data.n_codewords * data.pq_block_size * sizeof(float);
    const size_t mean_size = data.n_kv_heads * data.head_dim * sizeof(float);

    // Allocate temp buffer for reading. Tracks the current allocated size of
    // tmp_raw — may grow via realloc below if a v4 dense R matrix exceeds the
    // initial max{rot_size, cb_size, mean_size} (typical for 35B: dim=2048
    // gives 16 MB R, vs ~2 MB max chunk among the KV pieces).
    size_t max_chunk = std::max({rot_size, cb_size, mean_size});
    float * tmp_raw = (float *)malloc(max_chunk);
    if (!tmp_raw) {
        LLAMA_LOG_ERROR("%s: failed to allocate temp buffer (%zu bytes)\n", __func__, max_chunk);
        fclose(f);
        return false;
    }

    // --- Create rotation tensors ---
    data.R_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.head_dim, data.head_dim);
    ggml_format_name(data.R_kv, "spiral_cb_R_kv");

    data.R_kv_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.head_dim, data.head_dim);
    ggml_format_name(data.R_kv_inv, "spiral_cb_R_kv_inv");

    // --- Create per-layer tensors ---
    // For v1/v2 uniform: layers[i] is for model layer i (n_codebook_layers == n_layers).
    // For v3 hybrid: layers[i] is for the i-th attention layer; the model layer id
    //   is data.attention_layer_indices[i]. Callers translate model layer id →
    //   compact index using llama_kv_cache::map_layer_ids.
    data.layers.resize(n_codebook_layers);
    for (uint32_t li = 0; li < n_codebook_layers; li++) {
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
    data.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!data.buf) {
        LLAMA_LOG_ERROR("%s: failed to allocate buffer for codebook tensors\n", __func__);
        fclose(f);
        return false;
    }

    LLAMA_LOG_INFO("%s: codebook buffer size = %.2f MiB\n", __func__,
        ggml_backend_buffer_get_size(data.buf) / 1024.0 / 1024.0);


    // --- Read and upload rotation matrices ---
    size_t bytes_read = fread(tmp_raw, 1, rot_size, f);
    if (bytes_read != rot_size) {
        LLAMA_LOG_ERROR("%s: failed to read R_kv\n", __func__);
        fclose(f);
        return false;
    }
    ggml_backend_tensor_set(data.R_kv, tmp_raw, 0, rot_size);

    if (fread(tmp_raw, 1, rot_size, f) != rot_size) {
        LLAMA_LOG_ERROR("%s: failed to read R_kv_inv\n", __func__);
        fclose(f);
        return false;
    }
    ggml_backend_tensor_set(data.R_kv_inv, tmp_raw, 0, rot_size);

    // --- Read and upload per-layer data ---
    // For v1/v2: read all n_layers entries. For v3 hybrid: read only n_attention_layers
    // entries, in the order they appear in attention_layer_indices.
    for (uint32_t li = 0; li < n_codebook_layers; li++) {
        auto & layer = data.layers[li];


        // K codebooks
        size_t r = fread(tmp_raw, 1, cb_size, f);
        if (r != cb_size) {
            LLAMA_LOG_ERROR("%s: failed to read K codebooks for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(layer.k_codebooks, tmp_raw, 0, cb_size);

        // V codebooks
        if (fread(tmp_raw, 1, cb_size, f) != cb_size) {
            LLAMA_LOG_ERROR("%s: failed to read V codebooks for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(layer.v_codebooks, tmp_raw, 0, cb_size);

        // K mean carriers
        if (fread(tmp_raw, 1, mean_size, f) != mean_size) {
            LLAMA_LOG_ERROR("%s: failed to read K means for layer %u\n", __func__, li);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(layer.k_means, tmp_raw, 0, mean_size);
    }

    // --- Try to read weight rotation extension ---
    // Two formats handled here:
    //   v1/v2/v3: "SPIRALRT" magic → MPB-WHT params (signs + permutations).
    //   v4:       "SPIRRT4\0" magic → dense d×d float32 R matrices (Bible 13 §18).
    // After the rotation extension, v4 files also have a "SPIRCB4\0" extension
    // shipping the INT4/INT5 weight codebooks for runtime cross-check against
    // the values hardcoded in the Metal kernel.
    // If no extension magic is found (or fread returns short), we skip everything
    // (backward compat for older v1/v2 files that have no extensions).
    {
        char rot_magic[8] = {0};
        size_t n_read = fread(rot_magic, 1, 8, f);
        if (n_read == 8 && memcmp(rot_magic, "SPIRALRT", 8) == 0) {
            // ────────────────────────────────────────────────────────────
            // v1/v2/v3 MPB-WHT rotation extension (UNCHANGED from v3 code)
            // ────────────────────────────────────────────────────────────
            uint32_t n_dims = 0;
            fread(&n_dims, sizeof(uint32_t), 1, f);
            fseek(f, 28, SEEK_CUR);  // skip 28 bytes padding (40-byte total header)


            // Pass 1: read just dim/n_passes headers, skip data to learn layout
            long data_body_pos = ftell(f);
            data.weight_rotations.resize(n_dims);

            for (uint32_t di = 0; di < n_dims; di++) {
                uint32_t dim = 0, n_passes = 0;
                fread(&dim, sizeof(uint32_t), 1, f);
                fread(&n_passes, sizeof(uint32_t), 1, f);

                data.weight_rotations[di].dim = (int32_t)dim;
                data.weight_rotations[di].n_passes = (int32_t)n_passes;


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

                }

                LLAMA_LOG_INFO("%s: weight rotation buffer = %.2f KiB\n", __func__,
                    ggml_backend_buffer_get_size(data.rot_buf) / 1024.0);
            }
        } else if (n_read == 8 && memcmp(rot_magic, "SPIRRT4\0", 8) == 0) {
            // ────────────────────────────────────────────────────────────
            // v4 dense QR rotation extension (Bible 13 §18)
            //   ext_magic:     8 bytes "SPIRRT4\0"  (already consumed)
            //   n_unique_dims: uint32
            //   reserved:      28 bytes
            //   per dim:
            //     dim:    uint32
            //     pad:    4 bytes (align to 8)
            //     R:      dim × dim × float32  (dense rotation matrix, row-major)
            // ────────────────────────────────────────────────────────────
            if (hdr.version != 4) {
                LLAMA_LOG_ERROR("%s: SPIRRT4 extension found but file version is %u (expected 4)\n",
                    __func__, hdr.version);
                fclose(f);
                free(tmp_raw);
                return false;
            }

            uint32_t n_dims = 0;
            if (fread(&n_dims, sizeof(uint32_t), 1, f) != 1) {
                LLAMA_LOG_ERROR("%s: SPIRRT4: failed to read n_unique_dims\n", __func__);
                fclose(f);
                free(tmp_raw);
                return false;
            }
            if (fseek(f, 28, SEEK_CUR) != 0) {  // skip 28 bytes reserved (40-byte total header)
                LLAMA_LOG_ERROR("%s: SPIRRT4: failed to skip header padding\n", __func__);
                fclose(f);
                free(tmp_raw);
                return false;
            }

            LLAMA_LOG_INFO("%s:   v4 SPIRRT4 extension: %u unique dim(s)\n",
                __func__, n_dims);

            // Pass 1: read dim values, skip R bytes, just to size the tensor context
            long body_start = ftell(f);
            data.weight_rotations.resize(n_dims);
            for (uint32_t di = 0; di < n_dims; di++) {
                uint32_t dim = 0, pad = 0;
                if (fread(&dim, sizeof(uint32_t), 1, f) != 1 ||
                    fread(&pad, sizeof(uint32_t), 1, f) != 1) {
                    LLAMA_LOG_ERROR("%s: SPIRRT4: failed to read dim header for entry %u\n",
                        __func__, di);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }
                if (dim == 0 || dim > 65536) {
                    LLAMA_LOG_ERROR("%s: SPIRRT4: implausible dim=%u for entry %u\n",
                        __func__, dim, di);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }
                data.weight_rotations[di].dim = (int32_t)dim;
                data.weight_rotations[di].n_passes = 0;  // 0 == v4 dense path

                // Skip R payload
                size_t R_bytes = (size_t)dim * (size_t)dim * sizeof(float);
                if (fseek(f, (long)R_bytes, SEEK_CUR) != 0) {
                    LLAMA_LOG_ERROR("%s: SPIRRT4: failed to skip R payload for dim=%u\n",
                        __func__, dim);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }
            }

            // Build the rot context (one 2D tensor per dim, transposed at upload)
            size_t rot_ctx_size = (size_t)n_dims * ggml_tensor_overhead() + 1024;
            struct ggml_init_params rot_ctx_params = {
                /*.mem_size   =*/ rot_ctx_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * rot_ctx = ggml_init(rot_ctx_params);

            for (uint32_t di = 0; di < n_dims; di++) {
                auto & rot = data.weight_rotations[di];
                uint32_t dim = rot.dim;
                rot.R_dense_T = ggml_new_tensor_2d(rot_ctx, GGML_TYPE_F32, dim, dim);
                ggml_format_name(rot.R_dense_T, "spiral_rot_R_T_d%u", dim);
            }

            data.rot_buf = ggml_backend_alloc_ctx_tensors_from_buft(rot_ctx, buft);
            data.rot_ctx = rot_ctx;
            if (!data.rot_buf) {
                LLAMA_LOG_ERROR("%s: SPIRRT4: failed to allocate dense rotation buffer\n", __func__);
                ggml_free(rot_ctx);
                data.rot_ctx = nullptr;
                fclose(f);
                free(tmp_raw);
                return false;
            }

            // Pass 2: seek back, read each R, transpose host-side, upload as R_dense_T
            if (fseek(f, body_start, SEEK_SET) != 0) {
                LLAMA_LOG_ERROR("%s: SPIRRT4: failed to seek back to body start\n", __func__);
                fclose(f);
                free(tmp_raw);
                return false;
            }

            // We need two buffers for the transpose (read into one, transpose into another).
            // Reuse tmp_raw as the read buffer; allocate a separate transpose buffer.
            float * R_T_buf = nullptr;
            size_t R_T_buf_bytes = 0;

            for (uint32_t di = 0; di < n_dims; di++) {
                auto & rot = data.weight_rotations[di];
                uint32_t dim = rot.dim;

                // Skip dim + pad header (already read in pass 1)
                uint32_t skip_dim, skip_pad;
                fread(&skip_dim, sizeof(uint32_t), 1, f);
                fread(&skip_pad, sizeof(uint32_t), 1, f);

                size_t R_bytes = (size_t)dim * (size_t)dim * sizeof(float);

                // Ensure read buffer is large enough
                if (R_bytes > max_chunk) {
                    tmp_raw = (float *)realloc(tmp_raw, R_bytes);
                    if (!tmp_raw) {
                        LLAMA_LOG_ERROR("%s: SPIRRT4: failed to realloc read buffer to %zu bytes\n",
                            __func__, R_bytes);
                        free(R_T_buf);
                        fclose(f);
                        return false;
                    }
                    max_chunk = R_bytes;
                }

                // Read R (row-major in file)
                if (fread(tmp_raw, 1, R_bytes, f) != R_bytes) {
                    LLAMA_LOG_ERROR("%s: SPIRRT4: failed to read R for dim=%u\n",
                        __func__, dim);
                    free(R_T_buf);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }

                // Allocate / grow transpose buffer
                if (R_bytes > R_T_buf_bytes) {
                    float * new_buf = (float *)realloc(R_T_buf, R_bytes);
                    if (!new_buf) {
                        LLAMA_LOG_ERROR("%s: SPIRRT4: failed to alloc transpose buffer (%zu bytes)\n",
                            __func__, R_bytes);
                        free(R_T_buf);
                        fclose(f);
                        free(tmp_raw);
                        return false;
                    }
                    R_T_buf = new_buf;
                    R_T_buf_bytes = R_bytes;
                }

                // Transpose: R_T[i,j] = R[j,i]
                // Note: ggml's 2D tensor layout for ggml_new_tensor_2d(ctx, F32, dim, dim) has
                // the FIRST dim (ne0) varying fastest in memory. So if we want R_dense_T to
                // logically equal transpose(R) when accessed as R_T[i,j], the in-memory
                // layout we upload should be: byte stride for j is (dim*sizeof(float)),
                // byte stride for i is (sizeof(float)). That's the same row-major layout
                // we get from R^T computed as out[i*dim + j] = R[j*dim + i].
                for (uint32_t i = 0; i < dim; i++) {
                    for (uint32_t j = 0; j < dim; j++) {
                        R_T_buf[i * dim + j] = tmp_raw[j * dim + i];
                    }
                }

                ggml_backend_tensor_set(rot.R_dense_T, R_T_buf, 0, R_bytes);

                LLAMA_LOG_INFO("%s:   loaded dense R for dim=%u: %.2f MiB (transposed)\n",
                    __func__, dim, R_bytes / 1024.0 / 1024.0);
            }

            free(R_T_buf);

            LLAMA_LOG_INFO("%s: weight rotation buffer = %.2f MiB\n", __func__,
                ggml_backend_buffer_get_size(data.rot_buf) / 1024.0 / 1024.0);

            // ────────────────────────────────────────────────────────────
            // v4 weight codebook extension (SPIRCB4) — runtime cross-check
            //   ext_magic:    8 bytes "SPIRCB4\0"
            //   n_codebooks:  uint32 (= 2: int4 + int5)
            //   reserved:     20 bytes
            //   per codebook:
            //     bits:       uint32 (4 or 5)
            //     n_levels:   uint32 (16 or 32)
            //     centroids:  n_levels × float32
            // ────────────────────────────────────────────────────────────
            char cb_magic[8] = {0};
            size_t cb_n_read = fread(cb_magic, 1, 8, f);
            if (cb_n_read == 8 && memcmp(cb_magic, "SPIRCB4\0", 8) == 0) {
                uint32_t n_codebooks = 0;
                if (fread(&n_codebooks, sizeof(uint32_t), 1, f) != 1) {
                    LLAMA_LOG_ERROR("%s: SPIRCB4: failed to read n_codebooks\n", __func__);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }
                if (fseek(f, 20, SEEK_CUR) != 0) {  // skip 20 bytes reserved (32-byte total header)
                    LLAMA_LOG_ERROR("%s: SPIRCB4: failed to skip header padding\n", __func__);
                    fclose(f);
                    free(tmp_raw);
                    return false;
                }

                // Hardcoded centroids that the Metal kernel and ggml-spiral-quant.c use.
                // These MUST match the file. If they don't, fail loudly — it means the
                // build pipeline drifted from the runtime, and inference would silently
                // produce subtly-wrong output.
                static const float HARDCODED_INT4_CENTROIDS[16] = {
                    -2.7462113f, -2.0840564f, -1.6337705f, -1.2719219f,
                    -0.9567008f, -0.6680261f, -0.3953774f, -0.1310898f,
                     0.1310898f,  0.3953774f,  0.6680261f,  0.9567008f,
                     1.2719219f,  1.6337705f,  2.0840564f,  2.7462113f
                };
                static const float HARDCODED_INT5_CENTROIDS[32] = {
                    -3.3174999f, -2.7550619f, -2.3857274f, -2.0989547f,
                    -1.8585587f, -1.6477458f, -1.4572858f, -1.2809542f,
                    -1.1153967f, -0.9576638f, -0.8049663f, -0.6558650f,
                    -0.5089168f, -0.3634205f, -0.2186373f, -0.0731202f,
                     0.0731202f,  0.2186373f,  0.3634205f,  0.5089168f,
                     0.6558650f,  0.8049663f,  0.9576638f,  1.1153967f,
                     1.2809542f,  1.4572858f,  1.6477458f,  1.8585587f,
                     2.0989547f,  2.3857274f,  2.7550619f,  3.3174999f
                };

                for (uint32_t ci = 0; ci < n_codebooks; ci++) {
                    uint32_t bits = 0, n_levels = 0;
                    if (fread(&bits, sizeof(uint32_t), 1, f) != 1 ||
                        fread(&n_levels, sizeof(uint32_t), 1, f) != 1) {
                        LLAMA_LOG_ERROR("%s: SPIRCB4: failed to read codebook header %u\n",
                            __func__, ci);
                        fclose(f);
                        free(tmp_raw);
                        return false;
                    }

                    // Sanity check: bits/n_levels must match
                    const uint32_t expected_levels = (1u << bits);
                    if (n_levels != expected_levels) {
                        LLAMA_LOG_ERROR("%s: SPIRCB4: codebook %u bits=%u expects %u levels, got %u\n",
                            __func__, ci, bits, expected_levels, n_levels);
                        fclose(f);
                        free(tmp_raw);
                        return false;
                    }

                    std::vector<float> centroids(n_levels);
                    size_t cb_bytes = (size_t)n_levels * sizeof(float);
                    if (fread(centroids.data(), 1, cb_bytes, f) != cb_bytes) {
                        LLAMA_LOG_ERROR("%s: SPIRCB4: failed to read %u-bit centroids (%u levels)\n",
                            __func__, bits, n_levels);
                        fclose(f);
                        free(tmp_raw);
                        return false;
                    }

                    // Cross-check against compiled-in values. We allow a small
                    // absolute tolerance because codebooks.py computes centroids
                    // numerically (Lloyd-Max iteration on a discrete grid) while
                    // the kernel ships hardcoded float literals. fp32 evaluation
                    // order + BLAS differences between H100 (build host) and Mac
                    // (runtime) produce ULP-scale differences (~5e-8). Any real
                    // divergence between the build pipeline and the kernel would
                    // be many orders of magnitude larger than this tolerance.
                    constexpr float CENTROID_TOLERANCE = 1e-4f;
                    const float * expected = nullptr;
                    if (bits == 4 && n_levels == 16) {
                        expected = HARDCODED_INT4_CENTROIDS;
                    } else if (bits == 5 && n_levels == 32) {
                        expected = HARDCODED_INT5_CENTROIDS;
                    } else {
                        LLAMA_LOG_ERROR("%s: SPIRCB4: unsupported codebook bits=%u n_levels=%u\n",
                            __func__, bits, n_levels);
                        fclose(f);
                        free(tmp_raw);
                        return false;
                    }

                    for (uint32_t lvl = 0; lvl < n_levels; lvl++) {
                        const float diff = fabsf(centroids[lvl] - expected[lvl]);
                        if (diff > CENTROID_TOLERANCE) {
                            LLAMA_LOG_ERROR(
                                "%s: SPIRCB4: INT%u centroid[%u] mismatch — file=%.10g, kernel=%.10g (diff=%.2e > tol=%.2e)\n",
                                __func__, bits, lvl,
                                (double)centroids[lvl], (double)expected[lvl],
                                (double)diff, (double)CENTROID_TOLERANCE);
                            LLAMA_LOG_ERROR(
                                "%s:   .spiralcb file disagrees with hardcoded kernel centroids beyond fp32 noise.\n"
                                "%s:   This means the build pipeline (codebooks.py) was updated\n"
                                "%s:   without a matching update to ggml-spiral-quant.c and the\n"
                                "%s:   Metal kernel — refusing to load to avoid silent corruption.\n",
                                __func__, __func__, __func__, __func__);
                            fclose(f);
                            free(tmp_raw);
                            return false;
                        }
                    }

                    if (bits == 4) {
                        data.int4_centroids = std::move(centroids);
                    } else {
                        data.int5_centroids = std::move(centroids);
                    }

                    LLAMA_LOG_INFO("%s:   v4 SPIRCB4: INT%u %u centroids — kernel cross-check PASS\n",
                        __func__, bits, n_levels);
                }
            } else {
                // SPIRCB4 missing isn't fatal — older v4 files might lack it, or the EOF
                // came right after SPIRRT4. We've already verified rotation is loaded.
                LLAMA_LOG_WARN("%s: v4 file has SPIRRT4 but no SPIRCB4 codebook extension — "
                    "kernel cross-check skipped\n", __func__);
            }
        } else if (n_read != 0) {
            // Got 1-7 bytes or 8 bytes of unknown magic. For v4 files we EXPECT SPIRRT4,
            // so emit a warning. For v1/v2 we tolerate it (some old files may have trailing
            // junk). For v3 we expect SPIRALRT — already handled in the first arm.
            if (hdr.version == 4) {
                LLAMA_LOG_ERROR("%s: v4 file is missing SPIRRT4 extension (got '%c%c%c%c%c%c%c%c')\n",
                    __func__,
                    rot_magic[0] ? rot_magic[0] : '.',
                    rot_magic[1] ? rot_magic[1] : '.',
                    rot_magic[2] ? rot_magic[2] : '.',
                    rot_magic[3] ? rot_magic[3] : '.',
                    rot_magic[4] ? rot_magic[4] : '.',
                    rot_magic[5] ? rot_magic[5] : '.',
                    rot_magic[6] ? rot_magic[6] : '.',
                    rot_magic[7] ? rot_magic[7] : '.');
                fclose(f);
                free(tmp_raw);
                return false;
            }
        }
    }

    fflush(stderr);
    fclose(f);

    data.loaded = true;
    fflush(stderr);

    free(tmp_raw);

    return true;
}