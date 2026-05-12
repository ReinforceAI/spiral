/*
 * Spiral PQ2: Physics-derived KV cache compression
 * Based on: Spiral Research Bibles 1-5 (rotation + PQ codebook quantization)
 *
 * Implements GGML_TYPE_SPIRAL_PQ2 for use as --cache-type-k spiral_pq2
 *
 * Test 1: Minimal implementation — stores L2 norm, zero codes.
 *         Dequantize returns zero vectors.
 *         Purpose: prove the ggml dispatch path works end-to-end.
 *
 * Block layout (36 bytes per 128 values = 2.25 bits/value):
 *   float    d;          // 4 bytes: L2 norm
 *   uint8_t  codes[32];  // 32 bytes: PQ block codes (32 blocks × 8 bits)
 *                        // Each code indexes into a [256, 4] codebook
 *                        // 32 blocks × 4 dims = 128 dims total
 *
 * Future tests will add:
 *   Test 2: Lloyd-Max scalar quantization (our centroids)
 *   Test 3: R_kv rotation before quantization
 *   Test 4: PQ codebook quantization (the real physics)
 *   Test 5: K mean carrier subtraction
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

/* ─── Reference quantize: fp32 → block_spiral_pq2 ─── */

void quantize_row_spiral_pq2_ref(const float * GGML_RESTRICT x,
                                  block_spiral_pq2 * GGML_RESTRICT y,
                                  int64_t k) {
    assert(k % QK_SPIRAL_PQ2 == 0);
    const int nb = k / QK_SPIRAL_PQ2;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK_SPIRAL_PQ2;

        /* Compute L2 norm of this 128-dim vector */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_SPIRAL_PQ2; j++) {
            norm_sq += src[j] * src[j];
        }
        y[i].d = sqrtf(norm_sq);

        /* Test 1: store zero codes (no actual quantization yet) */
        memset(y[i].codes, 0, 32);
    }
}

/* ─── Reference dequantize: block_spiral_pq2 → fp32 ─── */

void dequantize_row_spiral_pq2(const block_spiral_pq2 * GGML_RESTRICT x,
                                float * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_SPIRAL_PQ2 == 0);
    const int nb = k / QK_SPIRAL_PQ2;

    for (int i = 0; i < nb; i++) {
        float * dst = y + i * QK_SPIRAL_PQ2;

        /* Test 1: output zeros (no codebook lookup yet) */
        for (int j = 0; j < QK_SPIRAL_PQ2; j++) {
            dst[j] = 0.0f;
        }
    }
}

/* ─── Batch quantize (called by ggml_quantize_chunk) ─── */

size_t quantize_spiral_pq2(const float * GGML_RESTRICT src,
                            void * GGML_RESTRICT dst,
                            int64_t nrows,
                            int64_t n_per_row,
                            const float * imatrix) {
    (void)imatrix; /* unused for now */

    assert(n_per_row % QK_SPIRAL_PQ2 == 0);
    const int nb_per_row = n_per_row / QK_SPIRAL_PQ2;
    const size_t row_size = nb_per_row * sizeof(block_spiral_pq2);

    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_spiral_pq2_ref(
            src + row * n_per_row,
            (block_spiral_pq2 *)((char *)dst + row * row_size),
            n_per_row
        );
    }

    return nrows * row_size;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Spiral 3-bit: MPB-rotated weight quantization (8-level Lloyd-Max for N(0,1))
 *
 * Block layout (50 bytes per 128 values = 3.125 bits/value):
 *   ggml_half norm;      // 2 bytes: per-row L2 norm
 *   uint8_t   qs[48];    // 48 bytes: 128 × 3-bit codes packed 8-per-3-bytes
 *
 * 3-bit packing (8 codes c0..c7 into 3 bytes):
 *   byte0 = c0 | (c1 << 3) | ((c2 & 0x3) << 6)
 *   byte1 = (c2 >> 2) | (c3 << 1) | (c4 << 4) | ((c5 & 0x1) << 7)
 *   byte2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)
 *
 * Lloyd-Max centroids for N(0,1), 8 levels:
 *   {-2.1556325, -1.3483801, -0.7599415, -0.2466970,
 *     0.2466970,  0.7599415,  1.3483801,  2.1556325}
 *
 * Dequant: value = centroid[code] * float(norm) * inv_scale
 *   where inv_scale = 1.0/sqrt(in_features), but for CPU reference
 *   dequant we omit inv_scale (applied at matmul time).
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float spiral_3bit_centroids[8] = {
    -2.1556325f, -1.3483801f, -0.7599415f, -0.2466970f,
     0.2466970f,  0.7599415f,  1.3483801f,  2.1556325f
};

/* ─── Unpack 3 bytes into 8 codes (each 0..7) ─── */
static inline void spiral_3bit_unpack8(const uint8_t * packed, uint8_t * codes) {
    codes[0] =  packed[0]       & 0x7;
    codes[1] = (packed[0] >> 3) & 0x7;
    codes[2] = (packed[0] >> 6) | ((packed[1] & 0x1) << 2);
    codes[3] = (packed[1] >> 1) & 0x7;
    codes[4] = (packed[1] >> 4) & 0x7;
    codes[5] = (packed[1] >> 7) | ((packed[2] & 0x3) << 1);
    codes[6] = (packed[2] >> 2) & 0x7;
    codes[7] = (packed[2] >> 5) & 0x7;
}

/* ─── Reference quantize: fp32 → block_spiral_3bit ─── */

void quantize_row_spiral_3bit_ref(const float * GGML_RESTRICT x,
                                   block_spiral_3bit * GGML_RESTRICT y,
                                   int64_t k) {
    assert(k % QK_SPIRAL_3BIT == 0);
    const int nb = k / QK_SPIRAL_3BIT;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK_SPIRAL_3BIT;

        /* Compute L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_SPIRAL_3BIT; j++) {
            norm_sq += src[j] * src[j];
        }
        float norm = sqrtf(norm_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        /* Quantize: find nearest centroid for each normalized value */
        float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

        uint8_t codes[QK_SPIRAL_3BIT];
        for (int j = 0; j < QK_SPIRAL_3BIT; j++) {
            float val = src[j] * inv_norm * sqrtf((float)QK_SPIRAL_3BIT);
            /* Find nearest centroid */
            int best = 0;
            float best_dist = fabsf(val - spiral_3bit_centroids[0]);
            for (int c = 1; c < 8; c++) {
                float dist = fabsf(val - spiral_3bit_centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            codes[j] = (uint8_t)best;
        }

        /* Pack 8 codes per 3 bytes */
        for (int g = 0; g < QK_SPIRAL_3BIT / 8; g++) {
            uint8_t * c = codes + g * 8;
            uint8_t * p = y[i].qs + g * 3;
            p[0] =  c[0]       | (c[1] << 3) | ((c[2] & 0x3) << 6);
            p[1] = (c[2] >> 2) | (c[3] << 1) | (c[4] << 4) | ((c[5] & 0x1) << 7);
            p[2] = (c[5] >> 1) | (c[6] << 2) | (c[7] << 5);
        }
    }
}

/* ─── Reference dequantize: block_spiral_3bit → fp32 ─── */

void dequantize_row_spiral_3bit(const block_spiral_3bit * GGML_RESTRICT x,
                                 float * GGML_RESTRICT y,
                                 int64_t k) {
    assert(k % QK_SPIRAL_3BIT == 0);
    const int nb = k / QK_SPIRAL_3BIT;

    for (int i = 0; i < nb; i++) {
        float * dst = y + i * QK_SPIRAL_3BIT;
        const float norm = GGML_FP16_TO_FP32(x[i].norm);
        const float scale = norm / sqrtf((float)QK_SPIRAL_3BIT);

        /* Unpack and dequantize 8 codes at a time */
        for (int g = 0; g < QK_SPIRAL_3BIT / 8; g++) {
            uint8_t codes[8];
            spiral_3bit_unpack8(x[i].qs + g * 3, codes);

            for (int j = 0; j < 8; j++) {
                dst[g * 8 + j] = spiral_3bit_centroids[codes[j]] * scale;
            }
        }
    }
}

/* ─── Batch quantize (called by ggml_quantize_chunk) ─── */

size_t quantize_spiral_3bit(const float * GGML_RESTRICT src,
                             void * GGML_RESTRICT dst,
                             int64_t nrows,
                             int64_t n_per_row,
                             const float * imatrix) {
    (void)imatrix;

    assert(n_per_row % QK_SPIRAL_3BIT == 0);
    const int nb_per_row = n_per_row / QK_SPIRAL_3BIT;
    const size_t row_size = nb_per_row * sizeof(block_spiral_3bit);

    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_spiral_3bit_ref(
            src + row * n_per_row,
            (block_spiral_3bit *)((char *)dst + row * row_size),
            n_per_row
        );
    }

    return nrows * row_size;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Spiral INT4: dense-QR-rotated 4-bit MoE weight quantization (v4 format)
 * Bible 13 §18: locked-in INT4/INT5 mixed precision recipe for Qwen3.6-35B-A3B
 *
 * Used for: MoE expert tensors (gate_up_proj, down_proj) on 32 of 40 layers
 *           (the "INT4 layers" — all layers NOT in the §18 boost set).
 *
 * Block layout (66 bytes per 128 values = 4.125 bits/value):
 *   ggml_half norm;      //  2 bytes: PRE-SCALED row L2 norm (row_norm/sqrt(in_features))
 *   uint8_t   qs[64];    // 64 bytes: 128 × 4-bit codes packed 2-per-byte
 *
 * 4-bit packing (2 codes per byte):
 *   byte[k] = (codes[2k] & 0x0F) | ((codes[2k+1] & 0x0F) << 4)
 *   even-index code in low nibble, odd-index code in high nibble
 *
 * Centroids: 16 fp32 Lloyd-Max-N(0,1) values — deterministic math constants.
 *   Generated by codebooks.lloyd_max_n01(16) in the H100 build pipeline.
 *   Bit-identical to what the build pipeline used to quantize and to what
 *   codebooks.py ships in .spiralcb (verified at SPIRCB4 cross-check time:
 *   max diff 0.00e+00).
 *
 * KEY DIFFERENCES FROM SPIRAL_3BIT (v3):
 *
 *   1. norm field stores PRE-SCALED row_norm (row_norm/sqrt(in_features)).
 *      The H100 converter (spiral_to_gguf.py::repack_quantized_matrix) bakes
 *      the 1/sqrt(in_features) factor into the stored norm. CPU dequant just
 *      multiplies by norm — no runtime sqrt division.
 *      (v3 stored raw row L2 norm and divided by sqrt(QK) at dequant time,
 *       which was technically incorrect for matrices where in_features ≠ QK.)
 *
 *   2. Centroids count: 16 (vs v3's 8).
 *
 *   3. Rotation: dense d×d float32 R matrix loaded at runtime from .spiralcb
 *      SPIRRT4 extension (vs v3's MPB-WHT signs+perm packed factor structure).
 *      The dense rotation IS NOT applied here — the CPU dequant produces
 *      values in ROTATED SPACE. The kernel applies R^T separately, typically
 *      folded into the matmul via x_rotated = R^T @ x precomputed once per
 *      layer per token, then y[r] = norm[r] * sum_i centroid[code[r,i]] * x_rotated[i].
 *
 * QUANTIZATION PATH:
 * Production quantization happens OFFLINE on H100 in build_spiral_artifact.py
 * (with proper dense QR rotation applied first). The quantize_row_spiral_int4_ref
 * below is for testing/symmetry — it does NOT apply rotation (no R available
 * at this layer). Outputs of this reference function are NOT bit-compatible
 * with what the H100 pipeline produces.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* 16-level Lloyd-Max-N(0,1) centroids — the v4 INT4 codebook.
 * Symmetric: pairs (c[i], c[15-i]) with c[15-i] = -c[i]. */
static const float spiral_int4_centroids[16] = {
    -2.7462113f, -2.0840564f, -1.6337705f, -1.2719219f,
    -0.9567008f, -0.6680261f, -0.3953774f, -0.1310898f,
     0.1310898f,  0.3953774f,  0.6680261f,  0.9567008f,
     1.2719219f,  1.6337705f,  2.0840564f,  2.7462113f
};

/* ─── Reference quantize: fp32 → block_spiral_int4 ─── */
/* Reference / testing only. NOT bit-compatible with H100 production output
 * because no dense QR rotation is applied. */

void quantize_row_spiral_int4_ref(const float * GGML_RESTRICT x,
                                    block_spiral_int4 * GGML_RESTRICT y,
                                    int64_t k) {
    assert(k % QK_SPIRAL_INT4 == 0);
    const int nb = k / QK_SPIRAL_INT4;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK_SPIRAL_INT4;

        /* Block L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_SPIRAL_INT4; j++) {
            norm_sq += src[j] * src[j];
        }
        const float norm = sqrtf(norm_sq);

        /* Pre-scale by 1/sqrt(QK_SPIRAL_INT4) for v4 storage convention.
         * Reference function only sees one block at a time — uses QK as the
         * proxy for in_features. */
        const float scaled_norm = norm / sqrtf((float)QK_SPIRAL_INT4);
        y[i].norm = GGML_FP32_TO_FP16(scaled_norm);

        /* Quantize: project to unit sphere * sqrt(QK), then snap to centroid */
        const float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

        uint8_t codes[QK_SPIRAL_INT4];
        for (int j = 0; j < QK_SPIRAL_INT4; j++) {
            const float val = src[j] * inv_norm * sqrtf((float)QK_SPIRAL_INT4);
            int best = 0;
            float best_dist = fabsf(val - spiral_int4_centroids[0]);
            for (int c = 1; c < 16; c++) {
                const float dist = fabsf(val - spiral_int4_centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            codes[j] = (uint8_t)best;
        }

        /* Pack 2 codes per byte: byte[k] = codes[2k] | (codes[2k+1] << 4) */
        for (int b = 0; b < QK_SPIRAL_INT4 / 2; b++) {
            y[i].qs[b] = (uint8_t)((codes[2*b] & 0x0F) | ((codes[2*b + 1] & 0x0F) << 4));
        }
    }
}

/* ─── Reference dequantize: block_spiral_int4 → fp32 (rotated space) ─── */
/* Output is in ROTATED space — kernel applies R^T separately for original-space recovery. */

void dequantize_row_spiral_int4(const block_spiral_int4 * GGML_RESTRICT x,
                                 float * GGML_RESTRICT y,
                                 int64_t k) {
    assert(k % QK_SPIRAL_INT4 == 0);
    const int nb = k / QK_SPIRAL_INT4;

    for (int i = 0; i < nb; i++) {
        float * dst = y + i * QK_SPIRAL_INT4;
        const float scaled_norm = GGML_FP16_TO_FP32(x[i].norm);

        /* Unpack 2 codes per byte. Even-index in low nibble, odd in high. */
        for (int b = 0; b < QK_SPIRAL_INT4 / 2; b++) {
            const uint8_t byte = x[i].qs[b];
            const uint8_t code_lo = byte & 0x0F;
            const uint8_t code_hi = (byte >> 4) & 0x0F;
            dst[2*b]     = spiral_int4_centroids[code_lo] * scaled_norm;
            dst[2*b + 1] = spiral_int4_centroids[code_hi] * scaled_norm;
        }
    }
}

/* ─── Batch quantize (called by ggml_quantize_chunk) ─── */

size_t quantize_spiral_int4(const float * GGML_RESTRICT src,
                             void * GGML_RESTRICT dst,
                             int64_t nrows,
                             int64_t n_per_row,
                             const float * imatrix) {
    (void)imatrix;

    assert(n_per_row % QK_SPIRAL_INT4 == 0);
    const int nb_per_row = n_per_row / QK_SPIRAL_INT4;
    const size_t row_size = nb_per_row * sizeof(block_spiral_int4);

    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_spiral_int4_ref(
            src + row * n_per_row,
            (block_spiral_int4 *)((char *)dst + row * row_size),
            n_per_row
        );
    }

    return nrows * row_size;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Spiral INT5: dense-QR-rotated 5-bit MoE weight quantization (v4 format)
 * Bible 13 §18: boost layers {5, 6, 10, 12, 33, 35, 38, 39}
 *
 * Used for: MoE expert tensors on 8 of 40 layers in Qwen3.6-35B-A3B (the
 *           §18 boost set, where per-layer attribution showed positive
 *           contribution above the +0.0007 nat threshold).
 *
 * Block layout (82 bytes per 128 values = 5.125 bits/value):
 *   ggml_half norm;      //  2 bytes: PRE-SCALED row L2 norm (row_norm/sqrt(in_features))
 *   uint8_t   qs[80];    // 80 bytes: 128 × 5-bit codes packed 8-per-5-bytes
 *
 * 5-bit packing (8 codes c0..c7 → 5 bytes b0..b4):
 *   b0 = (c0 & 0x1F)        | ((c1 & 0x07) << 5)
 *   b1 = ((c1 >> 3) & 0x03) | ((c2 & 0x1F) << 2) | ((c3 & 0x01) << 7)
 *   b2 = ((c3 >> 1) & 0x0F) | ((c4 & 0x0F) << 4)
 *   b3 = ((c4 >> 4) & 0x01) | ((c5 & 0x1F) << 1) | ((c6 & 0x03) << 6)
 *   b4 = ((c6 >> 2) & 0x07) | ((c7 & 0x1F) << 3)
 *
 * Centroids: 32 fp32 Lloyd-Max-N(0,1) values from codebooks.lloyd_max_n01(32).
 *
 * Same conventions as SPIRAL_INT4: pre-scaled norm, dequant in rotated space,
 * dense rotation applied separately by kernel via x_rotated = R^T @ x.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* 32-level Lloyd-Max-N(0,1) centroids — the v4 INT5 codebook.
 * Symmetric: pairs (c[i], c[31-i]) with c[31-i] = -c[i]. */
static const float spiral_int5_centroids[32] = {
    -3.3174999f, -2.7550619f, -2.3857274f, -2.0989547f,
    -1.8585587f, -1.6477458f, -1.4572858f, -1.2809542f,
    -1.1153967f, -0.9576638f, -0.8049663f, -0.6558650f,
    -0.5089168f, -0.3634205f, -0.2186373f, -0.0731202f,
     0.0731202f,  0.2186373f,  0.3634205f,  0.5089168f,
     0.6558650f,  0.8049663f,  0.9576638f,  1.1153967f,
     1.2809542f,  1.4572858f,  1.6477458f,  1.8585587f,
     2.0989547f,  2.3857274f,  2.7550619f,  3.3174999f
};

/* ─── Pack 8 codes (each 0..31) into 5 bytes ─── */
static inline void spiral_int5_pack8(const uint8_t * codes, uint8_t * packed) {
    const uint8_t c0 = codes[0] & 0x1F, c1 = codes[1] & 0x1F;
    const uint8_t c2 = codes[2] & 0x1F, c3 = codes[3] & 0x1F;
    const uint8_t c4 = codes[4] & 0x1F, c5 = codes[5] & 0x1F;
    const uint8_t c6 = codes[6] & 0x1F, c7 = codes[7] & 0x1F;

    packed[0] = (uint8_t)( c0           | ((c1 & 0x07) << 5));
    packed[1] = (uint8_t)((c1 >> 3)     |  (c2          << 2) | ((c3 & 0x01) << 7));
    packed[2] = (uint8_t)((c3 >> 1)     | ((c4 & 0x0F) << 4));
    packed[3] = (uint8_t)((c4 >> 4)     |  (c5          << 1) | ((c6 & 0x03) << 6));
    packed[4] = (uint8_t)((c6 >> 2)     |  (c7          << 3));
}

/* ─── Unpack 5 bytes into 8 codes (each 0..31) ─── */
static inline void spiral_int5_unpack8(const uint8_t * packed, uint8_t * codes) {
    const uint8_t b0 = packed[0], b1 = packed[1];
    const uint8_t b2 = packed[2], b3 = packed[3], b4 = packed[4];

    codes[0] =   b0       & 0x1F;
    codes[1] = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3);
    codes[2] =  (b1 >> 2) & 0x1F;
    codes[3] = ((b1 >> 7) & 0x01) | ((b2 & 0x0F) << 1);
    codes[4] = ((b2 >> 4) & 0x0F) | ((b3 & 0x01) << 4);
    codes[5] =  (b3 >> 1) & 0x1F;
    codes[6] = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2);
    codes[7] =  (b4 >> 3) & 0x1F;
}

/* ─── Reference quantize: fp32 → block_spiral_int5 ─── */

void quantize_row_spiral_int5_ref(const float * GGML_RESTRICT x,
                                    block_spiral_int5 * GGML_RESTRICT y,
                                    int64_t k) {
    assert(k % QK_SPIRAL_INT5 == 0);
    const int nb = k / QK_SPIRAL_INT5;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK_SPIRAL_INT5;

        /* Block L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_SPIRAL_INT5; j++) {
            norm_sq += src[j] * src[j];
        }
        const float norm = sqrtf(norm_sq);

        const float scaled_norm = norm / sqrtf((float)QK_SPIRAL_INT5);
        y[i].norm = GGML_FP32_TO_FP16(scaled_norm);

        const float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

        uint8_t codes[QK_SPIRAL_INT5];
        for (int j = 0; j < QK_SPIRAL_INT5; j++) {
            const float val = src[j] * inv_norm * sqrtf((float)QK_SPIRAL_INT5);
            int best = 0;
            float best_dist = fabsf(val - spiral_int5_centroids[0]);
            for (int c = 1; c < 32; c++) {
                const float dist = fabsf(val - spiral_int5_centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            codes[j] = (uint8_t)best;
        }

        /* Pack 8 codes per 5 bytes (16 groups per block of 128 codes) */
        for (int g = 0; g < QK_SPIRAL_INT5 / 8; g++) {
            spiral_int5_pack8(codes + g * 8, y[i].qs + g * 5);
        }
    }
}

/* ─── Reference dequantize: block_spiral_int5 → fp32 (rotated space) ─── */

void dequantize_row_spiral_int5(const block_spiral_int5 * GGML_RESTRICT x,
                                 float * GGML_RESTRICT y,
                                 int64_t k) {
    assert(k % QK_SPIRAL_INT5 == 0);
    const int nb = k / QK_SPIRAL_INT5;

    for (int i = 0; i < nb; i++) {
        float * dst = y + i * QK_SPIRAL_INT5;
        const float scaled_norm = GGML_FP16_TO_FP32(x[i].norm);

        /* Unpack and dequantize 8 codes at a time (16 groups per block) */
        for (int g = 0; g < QK_SPIRAL_INT5 / 8; g++) {
            uint8_t codes[8];
            spiral_int5_unpack8(x[i].qs + g * 5, codes);

            for (int j = 0; j < 8; j++) {
                dst[g * 8 + j] = spiral_int5_centroids[codes[j]] * scaled_norm;
            }
        }
    }
}

/* ─── Batch quantize (called by ggml_quantize_chunk) ─── */

size_t quantize_spiral_int5(const float * GGML_RESTRICT src,
                             void * GGML_RESTRICT dst,
                             int64_t nrows,
                             int64_t n_per_row,
                             const float * imatrix) {
    (void)imatrix;

    assert(n_per_row % QK_SPIRAL_INT5 == 0);
    const int nb_per_row = n_per_row / QK_SPIRAL_INT5;
    const size_t row_size = nb_per_row * sizeof(block_spiral_int5);

    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_spiral_int5_ref(
            src + row * n_per_row,
            (block_spiral_int5 *)((char *)dst + row * row_size),
            n_per_row
        );
    }

    return nrows * row_size;
}