/*
 * Fused mul_mat for SPIRAL_INT4 / SPIRAL_INT5 weight types — pure float.
 *
 * Mirrors the Metal reference kernel at ggml-metal/ggml-metal.metal:5234-5323
 * (SPIRAL_INT4) and :5346-5450 (SPIRAL_INT5).
 *
 * Spiral v4 (Bible 13 §18): dense-QR-rotated 4-bit / 5-bit MoE expert weights.
 * Lloyd-Max-N(0,1) centroids (16 for INT4, 32 for INT5). Per-block fp16 row
 * norm. Block size 128. The rotation R^T @ activation is inserted upstream as
 * GGML_OP_MUL_MAT (see spiral_rotate_activation in llama-graph.cpp); this
 * kernel receives activations already in the rotated basis.
 *
 * Computation — bit-identical to Metal:
 *   For each spiral block:
 *     - Unpack 128 codes (4-bit for INT4, 5-bit for INT5)
 *     - Look up float centroid value (c[code])
 *     - Multiply by raw f32 activation
 *     - Accumulate in float
 *   Final: multiply by per-row fp16 norm, simd reduce, lane-0 stores.
 *
 * Parallelization — mirrors Metal:
 *   32 threads per warp / simdgroup
 *   ix = lane / 8  ∈ {0,1,2,3}   — "block lane": each handles every 4th block
 *   it = lane % 8  ∈ {0,...,7}   — "chunk":      each handles 16 of 128 elements
 *   Final simd_sum (warp shuffle reduce) collapses all 32 partial sums.
 *
 * Multi-column extension (decode + multi-token batch ne11 ∈ [1..8]):
 *   Each thread loads its 16 weight codes ONCE per block, reuses across
 *   ncols_dst columns of activation. This is the key cost-amortization of the
 *   multi-token path. Identical pattern to TurboQuant's multi-token kernel
 *   (mmvq-tq.cu) — same INFRASTRUCTURE, different (correct) COMPUTATION.
 */

#include "mmvq-spiral.cuh"

// ============================================================================
// Lloyd-Max-N(0,1) centroid tables (float, in __constant__ memory)
// ============================================================================
//
// Verbatim from ggml-metal.metal:5269-5274 (INT4) and :5374-5383 (INT5).
// These same values are produced by codebooks.py:lloyd_max_n01(16) / (32)
// — verified bit-identical to 5e-8 (Lloyd-Max iterate precision floor).

__constant__ float SPIRAL_CENTROIDS_INT4_F32[16] = {
    -2.7462113f, -2.0840564f, -1.6337705f, -1.2719219f,
    -0.9567008f, -0.6680261f, -0.3953774f, -0.1310898f,
     0.1310898f,  0.3953774f,  0.6680261f,  0.9567008f,
     1.2719219f,  1.6337705f,  2.0840564f,  2.7462113f
};

__constant__ float SPIRAL_CENTROIDS_INT5_F32[32] = {
    -3.3174999f, -2.7550619f, -2.3857274f, -2.0989547f,
    -1.8585587f, -1.6477458f, -1.4572858f, -1.2809542f,
    -1.1153967f, -0.9576638f, -0.8049663f, -0.6558650f,
    -0.5089168f, -0.3634205f, -0.2186373f, -0.0731202f,
     0.0731202f,  0.2186373f,  0.3634205f,  0.5089168f,
     0.6558650f,  0.8049663f,  0.9576638f,  1.1153967f,
     1.2809542f,  1.4572858f,  1.6477458f,  1.8585587f,
     2.0989547f,  2.3857274f,  2.7550619f,  3.3174999f
};

// ============================================================================
// Block layouts (matches ggml-common.h: block_spiral_int4 / block_spiral_int5)
// ============================================================================

static constexpr int QK_SPIRAL = 128;

// SPIRAL_INT4: 66 bytes per block — 2-byte fp16 norm + 64 bytes of 4-bit codes.
//   qs[k] holds two codes:  low nibble = code 2k,  high nibble = code 2k+1.
struct block_spiral_int4_layout {
    __half  norm;     // 2 bytes — same value in every block of a row
    uint8_t qs[64];   // 64 bytes — 128 4-bit codes
};
static_assert(sizeof(block_spiral_int4_layout) == 66, "block_spiral_int4 must be 66 bytes");

// SPIRAL_INT5: 82 bytes per block — 2-byte fp16 norm + 80 bytes of 5-bit codes.
//   8 codes packed into every 5 bytes (40 bits = 8 * 5).
struct block_spiral_int5_layout {
    __half  norm;     // 2 bytes
    uint8_t qs[80];   // 80 bytes — 128 5-bit codes
};
static_assert(sizeof(block_spiral_int5_layout) == 82, "block_spiral_int5 must be 82 bytes");

// ============================================================================
// Multi-token SPIRAL_INT4 kernel
// ============================================================================
//
// 32 threads/warp, 4 warps/block → 4 rows per CUDA block.
// Grid: ceil(nrows_x / 4) blocks in x dimension.

template <int ncols_dst>
static __global__ void mul_mat_spiral_int4_f32(
        const void  * __restrict__ vx,        // SPIRAL_INT4 weights
        const float * __restrict__ vy,        // raw f32 activations
        float       * __restrict__ dst,
        const int ncols_x,                    // = ne00 (input dim)
        const int nrows_x,                    // = ne01 (output dim)
        const int stride_col_y,               // floats per activation column
        const int stride_col_dst) {           // floats per output column

    const int row_base = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_base >= nrows_x) return;

    const int lane = threadIdx.x;                   // 0..31
    const int ix   = lane >> 3;                     // 0..3, block lane
    const int it   = lane & 7;                      // 0..7, chunk within block
    const int boff = it * 8;                        // byte offset of this chunk's 8 bytes

    const int nb = ncols_x / QK_SPIRAL;
    const block_spiral_int4_layout * x_row =
        ((const block_spiral_int4_layout *) vx) + (int64_t) row_base * nb;

    float sumf[ncols_dst];
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) sumf[j] = 0.f;

    // Iterate over every 4th block (this thread handles blocks ib = ix, ix+4, ix+8, ...).
    for (int ib = ix; ib < nb; ib += 4) {
        const block_spiral_int4_layout * blk = &x_row[ib];

        // Load this chunk's 8 weight bytes (= 16 codes) ONCE per block.
        // memcpy avoids unaligned-access faults: block is 66 bytes (not 4-aligned).
        uint8_t qs[8];
        __builtin_memcpy(qs, blk->qs + boff, 8);

        // Look up 16 centroids from 8 weight bytes.
        // Byte k holds codes 2k and 2k+1 (low nibble + high nibble).
        float c_vals[16];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            c_vals[2*k + 0] = SPIRAL_CENTROIDS_INT4_F32[ qs[k]       & 0xF];
            c_vals[2*k + 1] = SPIRAL_CENTROIDS_INT4_F32[(qs[k] >> 4) & 0xF];
        }

        const float norm = __half2float(blk->norm);

        // For each output column, dot 16 centroids against 16 activations.
        // Activations layout: column-major [ncols_x, ncols_dst]. Stride between
        // columns is stride_col_y floats. Element offset within a column is
        // (block_idx * QK_SPIRAL + chunk_offset).
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float * y_col = vy + j * stride_col_y + ib * QK_SPIRAL + (it * 16);

            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                acc += c_vals[i] * y_col[i];
            }
            sumf[j] += acc * norm;
        }
    }

    // Warp reduction across 32 threads — collapses all (ix, it) partials.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            sumf[j] += __shfl_xor_sync(0xFFFFFFFFu, sumf[j], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            dst[j * stride_col_dst + row_base] = sumf[j];
        }
    }
}

// ============================================================================
// Multi-token SPIRAL_INT5 kernel
// ============================================================================
//
// Same parallelization as INT4. Per chunk reads 10 bytes (16 codes × 5 bits =
// 80 bits) instead of 8 bytes; unpacking is more involved. Verbatim from
// Metal:5408-5432.

template <int ncols_dst>
static __global__ void mul_mat_spiral_int5_f32(
        const void  * __restrict__ vx,
        const float * __restrict__ vy,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    const int row_base = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_base >= nrows_x) return;

    const int lane = threadIdx.x;
    const int ix   = lane >> 3;
    const int it   = lane & 7;
    const int boff = it * 10;                       // 16 codes × 5 bits = 80 bits = 10 bytes

    const int nb = ncols_x / QK_SPIRAL;
    const block_spiral_int5_layout * x_row =
        ((const block_spiral_int5_layout *) vx) + (int64_t) row_base * nb;

    float sumf[ncols_dst];
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) sumf[j] = 0.f;

    for (int ib = ix; ib < nb; ib += 4) {
        const block_spiral_int5_layout * blk = &x_row[ib];

        uint8_t qs[10];
        __builtin_memcpy(qs, blk->qs + boff, 10);

        // Unpack 16 codes from 10 bytes — 2 groups of 8 codes (5 bytes each).
        // Identical to Metal:5410-5423.
        float c_vals[16];
        #pragma unroll
        for (int g = 0; g < 2; g++) {
            const uint8_t b0 = qs[g*5 + 0];
            const uint8_t b1 = qs[g*5 + 1];
            const uint8_t b2 = qs[g*5 + 2];
            const uint8_t b3 = qs[g*5 + 3];
            const uint8_t b4 = qs[g*5 + 4];

            const uint8_t i0 =   b0       & 0x1F;
            const uint8_t i1 = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3);
            const uint8_t i2 =  (b1 >> 2) & 0x1F;
            const uint8_t i3 = ((b1 >> 7) & 0x01) | ((b2 & 0x0F) << 1);
            const uint8_t i4 = ((b2 >> 4) & 0x0F) | ((b3 & 0x01) << 4);
            const uint8_t i5 =  (b3 >> 1) & 0x1F;
            const uint8_t i6 = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2);
            const uint8_t i7 =  (b4 >> 3) & 0x1F;

            c_vals[g*8 + 0] = SPIRAL_CENTROIDS_INT5_F32[i0];
            c_vals[g*8 + 1] = SPIRAL_CENTROIDS_INT5_F32[i1];
            c_vals[g*8 + 2] = SPIRAL_CENTROIDS_INT5_F32[i2];
            c_vals[g*8 + 3] = SPIRAL_CENTROIDS_INT5_F32[i3];
            c_vals[g*8 + 4] = SPIRAL_CENTROIDS_INT5_F32[i4];
            c_vals[g*8 + 5] = SPIRAL_CENTROIDS_INT5_F32[i5];
            c_vals[g*8 + 6] = SPIRAL_CENTROIDS_INT5_F32[i6];
            c_vals[g*8 + 7] = SPIRAL_CENTROIDS_INT5_F32[i7];
        }

        const float norm = __half2float(blk->norm);

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float * y_col = vy + j * stride_col_y + ib * QK_SPIRAL + (it * 16);

            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                acc += c_vals[i] * y_col[i];
            }
            sumf[j] += acc * norm;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            sumf[j] += __shfl_xor_sync(0xFFFFFFFFu, sumf[j], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            dst[j * stride_col_dst + row_base] = sumf[j];
        }
    }
}

// ============================================================================
// Launchers
// ============================================================================

#define MMVQ_SPIRAL_NWARPS 4

template <int ncols_dst>
static void launch_spiral_int4(
        const void * vx, const float * vy, float * dst,
        int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst,
        cudaStream_t stream) {
    const int n_row_blocks = (nrows_x + MMVQ_SPIRAL_NWARPS - 1) / MMVQ_SPIRAL_NWARPS;
    const dim3 grid(n_row_blocks, 1, 1);
    const dim3 block(WARP_SIZE, MMVQ_SPIRAL_NWARPS, 1);
    mul_mat_spiral_int4_f32<ncols_dst><<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_spiral_int5(
        const void * vx, const float * vy, float * dst,
        int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst,
        cudaStream_t stream) {
    const int n_row_blocks = (nrows_x + MMVQ_SPIRAL_NWARPS - 1) / MMVQ_SPIRAL_NWARPS;
    const dim3 grid(n_row_blocks, 1, 1);
    const dim3 block(WARP_SIZE, MMVQ_SPIRAL_NWARPS, 1);
    mul_mat_spiral_int5_f32<ncols_dst><<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

// ============================================================================
// Entry point
// ============================================================================

void ggml_cuda_mul_mat_spiral(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_SPIRAL_INT4 ||
                src0->type == GGML_TYPE_SPIRAL_INT5);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % QK_SPIRAL == 0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 == 1 && ne13 == 1);
    GGML_ASSERT(ne11 >= 1 && ne11 <= 8);

    cudaStream_t stream = ctx.stream();

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *)       dst->data;

    // Stride between activation / output columns, in float elements.
    // ggml stores nb in bytes; convert.
    const int stride_col_y   = (int) (src1->nb[1] / sizeof(float));
    const int stride_col_dst = (int) (dst->nb[1]  / sizeof(float));
    const int ncols_x        = (int) ne00;
    const int nrows_x        = (int) ne01;

    if (src0->type == GGML_TYPE_SPIRAL_INT4) {
        switch ((int) ne11) {
            case 1: launch_spiral_int4<1>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 2: launch_spiral_int4<2>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 3: launch_spiral_int4<3>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 4: launch_spiral_int4<4>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 5: launch_spiral_int4<5>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 6: launch_spiral_int4<6>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 7: launch_spiral_int4<7>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 8: launch_spiral_int4<8>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
        }
    } else {
        switch ((int) ne11) {
            case 1: launch_spiral_int5<1>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 2: launch_spiral_int5<2>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 3: launch_spiral_int5<3>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 4: launch_spiral_int5<4>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 5: launch_spiral_int5<5>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 6: launch_spiral_int5<6>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 7: launch_spiral_int5<7>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 8: launch_spiral_int5<8>(src0_d, src1_d, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
