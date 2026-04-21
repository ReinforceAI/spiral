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