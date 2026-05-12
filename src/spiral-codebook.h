// Spiral PQ codebook data — loaded from .spiralcb file
// This structure holds the PQ codebook tensors for all layers.
// Created once during KV cache initialization, lives for the lifetime of the cache.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <cstdint>

// .spiralcb file header (64 bytes)
struct spiral_codebook_header {
    char     magic[8];       // "SPIRALCB"
    uint32_t version;        // 1
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_blocks;       // head_dim / pq_block_size
    uint32_t n_codewords;    // 256
    uint32_t pq_block_size;  // 4
    // 24 bytes padding to reach 64
};

// Per-layer codebook data stored as GPU tensors
struct spiral_codebook_layer {
    ggml_tensor * k_codebooks = nullptr;  // [n_kv_heads, n_blocks, n_codewords, pq_block_size] F32
    ggml_tensor * v_codebooks = nullptr;  // same shape
    ggml_tensor * k_means     = nullptr;  // [n_kv_heads, head_dim] F32
};

// Per-dimension weight rotation data
//
// In v3 .spiralcb files, rotation is MPB-WHT (signs + permutations) and uses
// the signs/perm/inv_perm fields below. n_passes is 2 or 3.
//
// In v4 .spiralcb files (Bible 13 §18), rotation is dense d×d random orthogonal
// QR. The signs/perm/inv_perm fields are unused and n_passes is 0; instead the
// R_dense_T tensor holds the transpose of the rotation matrix, ready to be
// multiplied against activations via stock ggml_mul_mat (no custom op needed).
//
// The graph builder dispatches based on n_passes: > 0 → v3 spiral_rotate op,
// 0 → v4 ggml_mul_mat(R_dense_T, x).
struct spiral_weight_rotation {
    int32_t  dim       = 0;
    int32_t  n_passes  = 0;       // 2 or 3 for v3 MPB; 0 for v4 dense
    // v3 (MPB-WHT) fields — used when n_passes > 0
    ggml_tensor * signs1    = nullptr;  // [dim] F32
    ggml_tensor * signs2    = nullptr;  // [dim] F32
    ggml_tensor * signs3    = nullptr;  // [dim] F32 (only for 3-pass)
    ggml_tensor * perm1     = nullptr;  // [dim] I32
    ggml_tensor * perm2     = nullptr;  // [dim] I32 (only for 3-pass)
    ggml_tensor * inv_perm1 = nullptr;  // [dim] I32
    ggml_tensor * inv_perm2 = nullptr;  // [dim] I32 (only for 3-pass)
    // v4 (dense QR) field — used when n_passes == 0
    // Stored as the TRANSPOSE of R so the graph builder can do
    // x_rotated = ggml_mul_mat(R_dense_T, x) directly without an extra
    // transpose op per layer per token.
    ggml_tensor * R_dense_T = nullptr;  // [dim, dim] F32
};

// All codebook data for the model
struct spiral_codebook_data {
    // Metadata from header
    uint32_t n_layers      = 0;
    uint32_t n_kv_heads    = 0;
    uint32_t head_dim      = 0;
    uint32_t n_blocks      = 0;
    uint32_t n_codewords   = 0;
    uint32_t pq_block_size = 0;

    // Rotation matrices (shared across layers)
    ggml_tensor * R_kv     = nullptr;  // [128, 128] F32 — forward rotation
    ggml_tensor * R_kv_inv = nullptr;  // [128, 128] F32 — inverse rotation (R^T)

    // Per-layer data
    std::vector<spiral_codebook_layer> layers;

    // For v3 hybrid models: model layer ids that have KV codebooks (sorted ascending).
    // Empty for v1/v2 uniform models. When non-empty, layers[i] corresponds to
    // model layer attention_layer_indices[i].
    std::vector<uint32_t> attention_layer_indices;

    // Weight rotation data — per unique dimension
    std::vector<spiral_weight_rotation> weight_rotations;

    // .spiralcb file version (1, 2, 3 = MPB rotation; 4 = dense QR rotation)
    // Used by the graph builder to dispatch the appropriate rotation op.
    uint32_t version = 0;

    // v4 weight codebooks (loaded from SPIRCB4 extension).
    // Empty for v1/v2/v3. For v4, holds 16 fp32 INT4 centroids and/or 32 fp32 INT5 centroids.
    // The Metal kernel hardcodes these same values; we load them here for runtime
    // cross-check — if file values drift from kernel constants, we fail loudly.
    std::vector<float> int4_centroids;  // size 16 if loaded, else empty
    std::vector<float> int5_centroids;  // size 32 if loaded, else empty

    // Get rotation data for a specific dimension (returns nullptr if not found)
    const spiral_weight_rotation * get_weight_rotation(int64_t dim) const {
        for (const auto & rot : weight_rotations) {
            if (rot.dim == dim) return &rot;
        }
        return nullptr;
    }

    // Whether codebooks are loaded
    bool loaded = false;

    // Ownership of the ggml context and backend buffer
    // These must stay alive for the lifetime of the codebook data.
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Rotation tensor context and buffer (separate from KV codebook ctx/buf)
    ggml_context * rot_ctx = nullptr;
    ggml_backend_buffer_t rot_buf = nullptr;

    ~spiral_codebook_data() {
        if (rot_buf) { ggml_backend_buffer_free(rot_buf); }
        if (rot_ctx) { ggml_free(rot_ctx); }
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }

    // Total memory used for codebook tensors
    size_t total_bytes() const;
};

// Load codebook data from a .spiralcb file.
// Creates tensors in the given ggml context and uploads data to the given buffer type.
// Returns true on success, false on failure (with error logged).
bool spiral_codebook_load(
    const std::string & path,
    spiral_codebook_data & data,
    ggml_context * ctx,
    ggml_backend_buffer_type_t buft);