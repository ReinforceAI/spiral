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

// ============================================================================
// MoE variant — SPIRAL_INT4 mul_mat_id kernel
// ============================================================================
//
// Computes the matmul output for ONE (token, expert_slot, output_row) triple
// per warp. Reads its expert_idx from the `ids` tensor on device (no host sync),
// then dispatches to the same per-row compute as the non-id kernel.
//
// Grid layout:
//   blockIdx.x = row group (0 .. ceil(nrows_x / NWARPS))
//   blockIdx.y = expert slot (0 .. n_expert_used)
//   blockIdx.z = token index (0 .. n_tokens)
//
// Output indexing (mul_mat_id semantics): dst is [ne0, n_expert_used, n_tokens]
// in column-major. We write dst[token][expert_slot][output_row]:
//   addr = dst + token*stride_tok_dst + expert_slot*stride_slot_dst + output_row.

static __global__ void mul_mat_spiral_int4_id_f32(
        const void    * __restrict__ vx,         // weights [n_experts, nrows_x, ncols_x/QK*66]
        const float   * __restrict__ vy,         // activations [ncols_x, n_tokens]
        const int32_t * __restrict__ ids,        // [n_expert_used, n_tokens]
        float         * __restrict__ dst,        // [nrows_x, n_expert_used, n_tokens]
        const int ncols_x,
        const int nrows_x,
        const int n_expert_used,
        const int64_t nb02_w,                    // bytes per expert in weight tensor
        const int stride_tok_y,                  // floats between activation tokens
        const int ids_stride_slot,               // elements between ids slots (= ids->nb[0] / sizeof int32)
        const int ids_stride_tok,                // elements between ids tokens (= ids->nb[1] / sizeof int32)
        const int stride_slot_dst,               // floats between dst expert slots
        const int stride_tok_dst) {              // floats between dst tokens

    const int row_base    = blockIdx.x * blockDim.y + threadIdx.y;
    const int expert_slot = blockIdx.y;
    const int token       = blockIdx.z;
    if (row_base >= nrows_x) return;

    // On-device expert lookup — this is THE key to graph compatibility.
    const int32_t expert_idx = ids[expert_slot * ids_stride_slot + token * ids_stride_tok];

    // Offset weight pointer to the selected expert's slice.
    const void * vx_expert = (const char *) vx + (int64_t) expert_idx * nb02_w;

    const int lane = threadIdx.x;
    const int ix   = lane >> 3;
    const int it   = lane & 7;
    const int boff = it * 8;

    const int nb = ncols_x / QK_SPIRAL;
    const block_spiral_int4_layout * x_row =
        ((const block_spiral_int4_layout *) vx_expert) + (int64_t) row_base * nb;

    // Activation pointer for this token.
    const float * vy_tok = vy + token * stride_tok_y;

    float sumf = 0.f;
    for (int ib = ix; ib < nb; ib += 4) {
        const block_spiral_int4_layout * blk = &x_row[ib];

        uint8_t qs[8];
        __builtin_memcpy(qs, blk->qs + boff, 8);

        float c_vals[16];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            c_vals[2*k + 0] = SPIRAL_CENTROIDS_INT4_F32[ qs[k]       & 0xF];
            c_vals[2*k + 1] = SPIRAL_CENTROIDS_INT4_F32[(qs[k] >> 4) & 0xF];
        }

        const float norm = __half2float(blk->norm);
        const float * y_chunk = vy_tok + ib * QK_SPIRAL + (it * 16);

        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += c_vals[i] * y_chunk[i];
        }
        sumf += acc * norm;
    }

    // Warp reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xFFFFFFFFu, sumf, offset);
    }

    if (lane == 0) {
        dst[token * stride_tok_dst + expert_slot * stride_slot_dst + row_base] = sumf;
    }
}

// ============================================================================
// MoE variant — SPIRAL_INT5 mul_mat_id kernel
// ============================================================================

static __global__ void mul_mat_spiral_int5_id_f32(
        const void    * __restrict__ vx,
        const float   * __restrict__ vy,
        const int32_t * __restrict__ ids,
        float         * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int n_expert_used,
        const int64_t nb02_w,
        const int stride_tok_y,
        const int ids_stride_slot,
        const int ids_stride_tok,
        const int stride_slot_dst,
        const int stride_tok_dst) {

    const int row_base    = blockIdx.x * blockDim.y + threadIdx.y;
    const int expert_slot = blockIdx.y;
    const int token       = blockIdx.z;
    if (row_base >= nrows_x) return;

    const int32_t expert_idx = ids[expert_slot * ids_stride_slot + token * ids_stride_tok];
    const void * vx_expert = (const char *) vx + (int64_t) expert_idx * nb02_w;

    const int lane = threadIdx.x;
    const int ix   = lane >> 3;
    const int it   = lane & 7;
    const int boff = it * 10;

    const int nb = ncols_x / QK_SPIRAL;
    const block_spiral_int5_layout * x_row =
        ((const block_spiral_int5_layout *) vx_expert) + (int64_t) row_base * nb;

    const float * vy_tok = vy + token * stride_tok_y;

    float sumf = 0.f;
    for (int ib = ix; ib < nb; ib += 4) {
        const block_spiral_int5_layout * blk = &x_row[ib];

        uint8_t qs[10];
        __builtin_memcpy(qs, blk->qs + boff, 10);

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
        const float * y_chunk = vy_tok + ib * QK_SPIRAL + (it * 16);

        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            acc += c_vals[i] * y_chunk[i];
        }
        sumf += acc * norm;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xFFFFFFFFu, sumf, offset);
    }

    if (lane == 0) {
        dst[token * stride_tok_dst + expert_slot * stride_slot_dst + row_base] = sumf;
    }
}

// ============================================================================
// MoE entry point
// ============================================================================
//
// Graph-capture-compatible: no host synchronization, no host-readback memcpy.
// Reads the `ids` tensor on device inside the kernel. Replaces the host-sync
// fallback in upstream `ggml_cuda_mul_mat_id` for Spiral weight types.
//
// Called from the early branch in ggml_cuda_mul_mat_id (added in ggml-cuda.cu).

void ggml_cuda_mul_mat_spiral_id(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_SPIRAL_INT4 ||
                src0->type == GGML_TYPE_SPIRAL_INT5);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src0->ne[0] % QK_SPIRAL == 0);

    // src0: [ne00=ncols_x, ne01=nrows_x, ne02=n_experts]
    // src1: [ne10=ncols_x, ne11=?, ne12=n_tokens]  (ne11 is per-token batch dim, usually 1)
    // ids:  [ne0_ids=n_expert_used, ne1_ids=n_tokens]
    // dst:  [ne0=nrows_x, ne1=n_expert_used, ne2=n_tokens]
    const int64_t ncols_x       = src0->ne[0];
    const int64_t nrows_x       = src0->ne[1];
    const int64_t n_tokens      = src1->ne[2];
    const int64_t n_expert_used = ids->ne[0];

    GGML_ASSERT(src1->ne[0] == ncols_x);
    GGML_ASSERT(src1->ne[1] == 1);   // standard MoE convention: 1 row per token
    GGML_ASSERT(ids->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == nrows_x);
    GGML_ASSERT(dst->ne[1] == n_expert_used);
    GGML_ASSERT(dst->ne[2] == n_tokens);

    // One-time diagnostic: print actual dimensions to stderr so we can verify our
    // assumptions about Qwen3.6 MoE shape match what's actually being passed.
    // Particularly: is `n_expert_used` truly 8 (Qwen36 active experts), or do we
    // see other values (e.g., shared experts coming through this path)?
    static bool printed_once = false;
    if (!printed_once) {
        printed_once = true;
        fprintf(stderr, "=== spiral_id first call: ncols_x=%lld nrows_x=%lld "
                        "n_expert_used=%lld n_tokens=%lld type=%s ===\n",
                (long long) ncols_x, (long long) nrows_x,
                (long long) n_expert_used, (long long) n_tokens,
                ggml_type_name(src0->type));
    }

    // CUDA grid dimension limits: gridDim.y and gridDim.z are bounded by 65535 each.
    // n_expert_used maps to gridDim.y, n_tokens maps to gridDim.z. Either dim of 65535+
    // would silently fail on launch.
    GGML_ASSERT(n_expert_used <= 65535 && n_tokens <= 65535);

    cudaStream_t stream = ctx.stream();

    const void    * src0_d = src0->data;
    const float   * src1_d = (const float *)   src1->data;
    const int32_t * ids_d  = (const int32_t *) ids->data;
    float         * dst_d  = (float *)         dst->data;

    // Stride conversions: ggml stores nb in bytes; we need element counts.
    const int stride_tok_y       = (int) (src1->nb[2]   / sizeof(float));
    const int ids_stride_slot    = (int) (ids->nb[0]    / sizeof(int32_t));
    const int ids_stride_tok     = (int) (ids->nb[1]    / sizeof(int32_t));
    const int stride_slot_dst    = (int) (dst->nb[1]    / sizeof(float));
    const int stride_tok_dst     = (int) (dst->nb[2]    / sizeof(float));
    const int64_t nb02_w         = (int64_t) src0->nb[2];

    const int n_row_blocks = (int) ((nrows_x + MMVQ_SPIRAL_NWARPS - 1) / MMVQ_SPIRAL_NWARPS);
    const dim3 grid(n_row_blocks, (unsigned) n_expert_used, (unsigned) n_tokens);
    const dim3 block(WARP_SIZE, MMVQ_SPIRAL_NWARPS, 1);

    if (src0->type == GGML_TYPE_SPIRAL_INT4) {
        mul_mat_spiral_int4_id_f32<<<grid, block, 0, stream>>>(
            src0_d, src1_d, ids_d, dst_d,
            (int) ncols_x, (int) nrows_x, (int) n_expert_used,
            nb02_w, stride_tok_y,
            ids_stride_slot, ids_stride_tok,
            stride_slot_dst, stride_tok_dst);
    } else {
        mul_mat_spiral_int5_id_f32<<<grid, block, 0, stream>>>(
            src0_d, src1_d, ids_d, dst_d,
            (int) ncols_x, (int) nrows_x, (int) n_expert_used,
            nb02_w, stride_tok_y,
            ids_stride_slot, ids_stride_tok,
            stride_slot_dst, stride_tok_dst);
    }
    CUDA_CHECK(cudaGetLastError());
}
