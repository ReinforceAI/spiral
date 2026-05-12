/*
 * Fused mul_mat for SPIRAL_INT4 / SPIRAL_INT5 weight types.
 *
 * Spiral v4 (Bible 13 §18): dense-QR-rotated 4-bit / 5-bit MoE expert weights.
 * Lloyd-Max-N(0,1) centroids (16 levels for INT4, 32 for INT5).
 * Per-block fp16 row norm. Block size 128.
 *
 * The rotation R^T @ activation is inserted upstream as a GGML_OP_MUL_MAT in
 * the graph (see spiral_rotate_activation in llama-graph.cpp). This kernel
 * receives activations already in the rotated basis — it does NOT rotate.
 *
 * Algorithm:
 *   1. Pre-quantize the activation to block_q8_1 (32-element sub-blocks).
 *   2. Dispatch to type-specific kernel.
 *   3. Per spiral block: unpack codes, look up int8 centroids, __dp4a against int8 activation.
 *   4. Accumulate per-row across blocks, apply row_norm * centroid_scale at end.
 *   5. Warp reduction, lane-0 stores result.
 *
 * Step 2f: SPIRAL_INT4 + INT5 with ncols_dst ∈ [1..8] (decode + multi-token).
 *   - Templated kernels over ncols_dst; centroid lookup reused across columns
 *     within each weight block (mirrors TurboQuant's pattern at mmvq-tq.cu:142-220)
 *   - 16 kernel instantiations total (2 types × 8 ncols_dst), dispatched at runtime
 *   - Caller (ggml_cuda_mul_mat dispatch) guarantees ne11 ≤ MMVQ_MAX_BATCH_SIZE
 */

#include "mmvq-spiral.cuh"
#include "convert.cuh"   // for block_q8_1
#include "quantize.cuh"  // for quantize_row_q8_1_cuda

#include <mutex>

// ============================================================================
// Lloyd-Max-N(0,1) centroid tables (float source-of-truth)
// ============================================================================
//
// These tables are copied verbatim from ggml/src/ggml-metal/ggml-metal.metal,
// specifically:
//   spiral_centroids_int4   — lines 1556-1561
//   spiral_centroids_int5   — lines 1564-1573
//
// DO NOT modify these values independently. The build pipeline (codebooks.py,
// build_spiral_artifact.py) generates weight codes against these specific
// Lloyd-Max iterates. Any drift breaks bit-identity between platforms.

static constexpr float SPIRAL_CENTROIDS_INT4_F32[16] = {
    -2.7462113f, -2.0840564f, -1.6337705f, -1.2719219f,
    -0.9567008f, -0.6680261f, -0.3953774f, -0.1310898f,
     0.1310898f,  0.3953774f,  0.6680261f,  0.9567008f,
     1.2719219f,  1.6337705f,  2.0840564f,  2.7462113f
};

static constexpr float SPIRAL_CENTROIDS_INT5_F32[32] = {
    -3.3174999f, -2.7550619f, -2.3857274f, -2.0989547f,
    -1.8585587f, -1.6477458f, -1.4572858f, -1.2809542f,
    -1.1153967f, -0.9576638f, -0.8049663f, -0.6558650f,
    -0.5089168f, -0.3634205f, -0.2186373f, -0.0731202f,
     0.0731202f,  0.2186373f,  0.3634205f,  0.5089168f,
     0.6558650f,  0.8049663f,  0.9576638f,  1.1153967f,
     1.2809542f,  1.4572858f,  1.6477458f,  1.8585587f,
     2.0989547f,  2.3857274f,  2.7550619f,  3.3174999f
};

static_assert(sizeof(SPIRAL_CENTROIDS_INT4_F32) / sizeof(float) == 16,
              "SPIRAL_INT4 must have exactly 16 centroids (4-bit codes)");
static_assert(sizeof(SPIRAL_CENTROIDS_INT5_F32) / sizeof(float) == 32,
              "SPIRAL_INT5 must have exactly 32 centroids (5-bit codes)");

// ============================================================================
// int8-quantized centroid tables in __constant__ memory
// ============================================================================
//
// Pre-quantized to int8: scale by 127/max|centroid| so max|centroid| → ±127.
// Reverse scale (max|centroid|/127) applied at end of dot product.
//
//   max|SPIRAL_INT4| = 2.7462113  →  to_i8 = 127/2.7462113 = 46.2455
//   max|SPIRAL_INT5| = 3.3174999  →  to_i8 = 127/3.3174999 = 38.2818
//
// These __constant__ tables are kept for validation / fallback paths.
// The hot decode path uses packed compile-time constants below.

static __constant__ int8_t SPIRAL_CENTROIDS_INT4_Q[16];
static __constant__ int8_t SPIRAL_CENTROIDS_INT5_Q[32];

static float SPIRAL_INT4_RECOVERY_SCALE = 0.0f;
static float SPIRAL_INT5_RECOVERY_SCALE = 0.0f;

// ============================================================================
// SPIRAL_INT4 packed centroid constants for register-based dp4a lookup
// ============================================================================
//
// The 16 int8 centroids packed into 4 uint32s, little-endian byte order.
// Used by spiral_int4_cents8_reg() for parallel lookup via __byte_perm,
// avoiding any memory access in the inner loop.
//
// Pack layout (each uint32 = 4 centroids, byte[0]=LSB):
//   SP_I4_CR03 = [-127, -96, -76, -59]                    = 0xC5B4A081
//   SP_I4_CR47 = [-44, -31, -18, -6]                      = 0xFAEEE1D4
//   SP_I4_CR8B = [6, 18, 31, 44]                          = 0x2C1F1206
//   SP_I4_CRCF = [59, 76, 96, 127]                        = 0x7F604C3B
//
// Differs from TurboQuant TQ4_1S only in centroids[2]/[13] (76 vs 75).

static constexpr uint32_t SP_I4_CR03 = 0xC5B4A081u;
static constexpr uint32_t SP_I4_CR47 = 0xFAEEE1D4u;
static constexpr uint32_t SP_I4_CR8B = 0x2C1F1206u;
static constexpr uint32_t SP_I4_CRCF = 0x7F604C3Bu;

// ============================================================================
// One-time init for __constant__ centroid tables
// ============================================================================
//
// Thread-safe via std::call_once. Computes int8 tables on host from the
// float source-of-truth, verifies they match the packed compile-time
// constants, and uploads via cudaMemcpyToSymbol. Also stores recovery scales.

static std::once_flag SPIRAL_CENTROID_INIT_FLAG;

static void spiral_centroid_init() {
    // ---- INT4: 16 centroids ----
    {
        float max_abs = 0.0f;
        for (int i = 0; i < 16; i++) {
            const float a = fabsf(SPIRAL_CENTROIDS_INT4_F32[i]);
            if (a > max_abs) max_abs = a;
        }
        const float to_i8_scale = 127.0f / max_abs;
        SPIRAL_INT4_RECOVERY_SCALE = max_abs / 127.0f;

        int8_t host_table[16];
        for (int i = 0; i < 16; i++) {
            const float scaled = SPIRAL_CENTROIDS_INT4_F32[i] * to_i8_scale;
            int rounded = (int) roundf(scaled);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            host_table[i] = (int8_t) rounded;
        }

        // Verify the compile-time packed constants match the runtime table.
        const int8_t expected[16] = {
            (int8_t)((SP_I4_CR03 >>  0) & 0xFF), (int8_t)((SP_I4_CR03 >>  8) & 0xFF),
            (int8_t)((SP_I4_CR03 >> 16) & 0xFF), (int8_t)((SP_I4_CR03 >> 24) & 0xFF),
            (int8_t)((SP_I4_CR47 >>  0) & 0xFF), (int8_t)((SP_I4_CR47 >>  8) & 0xFF),
            (int8_t)((SP_I4_CR47 >> 16) & 0xFF), (int8_t)((SP_I4_CR47 >> 24) & 0xFF),
            (int8_t)((SP_I4_CR8B >>  0) & 0xFF), (int8_t)((SP_I4_CR8B >>  8) & 0xFF),
            (int8_t)((SP_I4_CR8B >> 16) & 0xFF), (int8_t)((SP_I4_CR8B >> 24) & 0xFF),
            (int8_t)((SP_I4_CRCF >>  0) & 0xFF), (int8_t)((SP_I4_CRCF >>  8) & 0xFF),
            (int8_t)((SP_I4_CRCF >> 16) & 0xFF), (int8_t)((SP_I4_CRCF >> 24) & 0xFF),
        };
        for (int i = 0; i < 16; i++) {
            GGML_ASSERT(host_table[i] == expected[i] &&
                        "SPIRAL_INT4 packed constants out of sync with float centroids");
        }

        CUDA_CHECK(cudaMemcpyToSymbol(SPIRAL_CENTROIDS_INT4_Q, host_table, sizeof(host_table)));
    }

    // ---- INT5: 32 centroids ----
    {
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float a = fabsf(SPIRAL_CENTROIDS_INT5_F32[i]);
            if (a > max_abs) max_abs = a;
        }
        const float to_i8_scale = 127.0f / max_abs;
        SPIRAL_INT5_RECOVERY_SCALE = max_abs / 127.0f;

        int8_t host_table[32];
        for (int i = 0; i < 32; i++) {
            const float scaled = SPIRAL_CENTROIDS_INT5_F32[i] * to_i8_scale;
            int rounded = (int) roundf(scaled);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            host_table[i] = (int8_t) rounded;
        }
        CUDA_CHECK(cudaMemcpyToSymbol(SPIRAL_CENTROIDS_INT5_Q, host_table, sizeof(host_table)));
    }
}

// ============================================================================
// Register-based centroid lookup for SPIRAL_INT4
// ============================================================================
//
// Takes one uint32 of weight code bytes (4 bytes × 2 nibbles = 8 codes) and
// produces 2 packed uint32s of int8 centroid values (8 centroids total, ready
// for dp4a).
//
// Algorithm (same as TurboQuant's tq4_cents8_reg at mmvq-tq.cu:67-100):
//   1. Extract low and high nibbles of each byte (8 nibbles total).
//   2. Interleave: nibbles 0..3 → sel0, nibbles 4..7 → sel1.
//   3. For each sel, use __byte_perm to look up 4 centroids from
//      the packed constants. The MSB of each nibble selects between
//      [CR03,CR47] (codes 0..7) and [CR8B,CRCF] (codes 8..15).

__device__ __forceinline__ void spiral_int4_cents8_reg(uint32_t four_bytes, int &c0, int &c1) {
    const uint32_t lo = four_bytes & 0x0F0F0F0Fu;
    const uint32_t hi = (four_bytes >> 4) & 0x0F0F0F0Fu;

    const uint32_t sel0 = __byte_perm(lo, hi, 0x5140u);
    const uint32_t sel1 = __byte_perm(lo, hi, 0x7362u);

    {
        const uint32_t flo = __byte_perm(SP_I4_CR03, SP_I4_CR47, sel0);
        const uint32_t fhi = __byte_perm(SP_I4_CR8B, SP_I4_CRCF, sel0);
        const uint32_t msb = (sel0 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c0 = (int)__byte_perm(flo, fhi, psel);
    }

    {
        const uint32_t flo = __byte_perm(SP_I4_CR03, SP_I4_CR47, sel1);
        const uint32_t fhi = __byte_perm(SP_I4_CR8B, SP_I4_CRCF, sel1);
        const uint32_t msb = (sel1 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c1 = (int)__byte_perm(flo, fhi, psel);
    }
}

// ============================================================================
// Per-block dot product for SPIRAL_INT4
// ============================================================================
//
// Computes float partial sum across one spiral block (128 codes) and the
// corresponding 4 q8_1 sub-blocks (4 × 32 = 128 activation values), with
// per-sub-block activation scale applied.
//
// Returns: float partial sum = Σ_qb (d_act_qb × int_dot_qb) for ONE output column
// Caller must still apply: result × SPIRAL_INT4_RECOVERY_SCALE × row_norm
//
// Multi-column variant: produces ncols_dst float partial sums in one pass,
// reusing the weight centroid lookup across columns. This is the optimization
// from TurboQuant mmvq-tq.cu:142-220 — weight loaded once, dp4a'd against all
// ncols_dst activation columns before moving to the next weight block.

template <int ncols_dst>
__device__ __forceinline__ void spiral_int4_dot_block_multi(
        const block_spiral_int4 * __restrict__ blk,
        const block_q8_1 * __restrict__ vy_q8,    // base pointer to all q8_1 blocks
        int ib,                                    // weight block index
        int stride_col_y,                          // q8_1 blocks per activation column
        float (& block_sums)[ncols_dst]) {

    // 64 weight bytes per spiral block, viewed as 16 uint32s.
    //
    // ALIGNMENT NOTE: block_spiral_int4 is 66 bytes (2 + 64). When packed into a
    // tensor, block N starts at byte offset 66*N, with qs[] at +2 inside.
    // For most N, qs is NOT 4-byte aligned (e.g. block 0: qs at byte 2; block 1:
    // qs at byte 68; block 2: qs at byte 134). Casting blk->qs to uint32_t* and
    // dereferencing produces a CUDA "misaligned address" fault on H100. We use
    // __builtin_memcpy to read each uint32 — the compiler emits the appropriate
    // unaligned load (PTX ld.b32 with byte source on Hopper) which doesn't fault.
    const uint8_t * qs_bytes = blk->qs;

    // Unpack all 32 centroid uint32s (16 lo + 16 hi, one pair per weight uint32)
    // once. These get reused across all ncols_dst activation columns.
    int c_lo[16], c_hi[16];
    #pragma unroll
    for (int w = 0; w < 16; w++) {
        uint32_t weight_u32;
        __builtin_memcpy(&weight_u32, qs_bytes + w * 4, sizeof(uint32_t));
        spiral_int4_cents8_reg(weight_u32, c_lo[w], c_hi[w]);
    }

    // For each output column, walk the 4 q8_1 sub-blocks of column j's
    // weight block ib, accumulating scaled int dots.
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) {
        const block_q8_1 * a_subs = vy_q8 + j * stride_col_y + ib * 4;

        float col_sum = 0.0f;
        #pragma unroll
        for (int qb = 0; qb < 4; qb++) {
            const block_q8_1 & a = a_subs[qb];
            const float d_act = __half2float((__half) a.ds.x);
            // block_q8_1 is 36 bytes (4 + 32), qs at offset 4 → always 4-aligned,
            // so this cast is safe.
            const int * a_qs = (const int *) a.qs;

            int sum_int = 0;
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                const int cw_idx = qb * 4 + w;
                sum_int = __dp4a(c_lo[cw_idx], a_qs[w * 2 + 0], sum_int);
                sum_int = __dp4a(c_hi[cw_idx], a_qs[w * 2 + 1], sum_int);
            }

            col_sum += d_act * (float) sum_int;
        }

        block_sums[j] += col_sum;
    }
}

// ============================================================================
// SPIRAL_INT4 mul_mat_vec_q kernel (templated on ncols_dst)
// ============================================================================

#define MMVQ_SPIRAL_NWARPS 4  // 4 warps × 32 lanes = 128 threads per block

template <int ncols_dst>
static __global__ void mul_mat_vec_q_spiral_int4(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,          // input dim (% 128 == 0)
        const int nrows_x,          // output dim
        const int stride_col_y,     // q8_1 blocks per activation column
        const int stride_col_dst,   // floats per output column
        const float recovery_scale) {

    const int row = blockIdx.x * MMVQ_SPIRAL_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / 128;
    const block_spiral_int4 * x_row =
        ((const block_spiral_int4 *) vx) + (int64_t) row * blocks_per_row;

    const float norm = __half2float(x_row[0].norm);

    float sumf[ncols_dst];
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) sumf[j] = 0.0f;

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        spiral_int4_dot_block_multi<ncols_dst>(&x_row[ib], vy_q8, ib, stride_col_y, sumf);
    }

    // Warp reduction (32 lanes) for each column
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
        }
    }

    if (lane == 0) {
        const float scale = recovery_scale * norm;
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            dst[j * stride_col_dst + row] = sumf[j] * scale;
        }
    }
}

// ============================================================================
// Launch wrapper (INT4) — instantiates the kernel template for given ncols_dst
// ============================================================================

template <int ncols_dst>
static void launch_spiral_int4(
        const void * src0_d,
        const block_q8_1 * q8_buf,
        float * dst_d,
        int ncols_x,
        int nrows_x,
        int stride_col_y,
        int stride_col_dst,
        float recovery_scale,
        cudaStream_t stream) {

    const dim3 block(WARP_SIZE, MMVQ_SPIRAL_NWARPS);
    const dim3 grid((nrows_x + MMVQ_SPIRAL_NWARPS - 1) / MMVQ_SPIRAL_NWARPS);
    mul_mat_vec_q_spiral_int4<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, recovery_scale);
}

// ============================================================================
// SPIRAL_INT5: 5-byte group unpacker + centroid lookup
// ============================================================================
//
// Takes 5 bytes of packed codes and returns 2 packed int32s containing 8 int8
// centroid values (4 per int32), ready for dp4a.
//
// Unpacking is verbatim from ggml-metal.metal:1955-1965 (spiral_int5_unpack_group).
// Each code is 5 bits, range 0..31; codes straddle byte boundaries per
// ggml-common.h:415-420.
//
// Centroid lookup is from __constant__ memory (SPIRAL_CENTROIDS_INT5_Q,
// populated by spiral_centroid_init). Unlike INT4 which uses register-packed
// __byte_perm lookup, INT5 with 32 entries doesn't fit cleanly in 4 registers,
// so we use __constant__ indexing. The __constant__ cache makes this fast for
// our access pattern (broadcast within warp typically; the 5 cache lines for
// 32 bytes fit easily).

__device__ __forceinline__ void spiral_int5_group_to_centroids(
        const uint8_t * __restrict__ group_bytes, int & c_lo, int & c_hi) {
    const uint8_t b0 = group_bytes[0];
    const uint8_t b1 = group_bytes[1];
    const uint8_t b2 = group_bytes[2];
    const uint8_t b3 = group_bytes[3];
    const uint8_t b4 = group_bytes[4];

    // Unpack 8 codes — verbatim copy of Metal's spiral_int5_unpack_group
    const uint8_t idx0 =   b0       & 0x1F;
    const uint8_t idx1 = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3);
    const uint8_t idx2 =  (b1 >> 2) & 0x1F;
    const uint8_t idx3 = ((b1 >> 7) & 0x01) | ((b2 & 0x0F) << 1);
    const uint8_t idx4 = ((b2 >> 4) & 0x0F) | ((b3 & 0x01) << 4);
    const uint8_t idx5 =  (b3 >> 1) & 0x1F;
    const uint8_t idx6 = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2);
    const uint8_t idx7 =  (b4 >> 3) & 0x1F;

    // Look up int8 centroids from __constant__ memory and pack into int32s.
    // Cast through uint8_t to preserve two's-complement bit pattern when widening.
    c_lo = ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx0])       |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx1] <<  8) |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx2] << 16) |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx3] << 24);
    c_hi = ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx4])       |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx5] <<  8) |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx6] << 16) |
           ((uint32_t)(uint8_t) SPIRAL_CENTROIDS_INT5_Q[idx7] << 24);
}

// ============================================================================
// Per-block dot product for SPIRAL_INT5 (multi-column)
// ============================================================================
//
// Structure mirrors spiral_int4_dot_block_multi:
//   - 1 spiral block = 128 codes = 80 bytes = 16 groups of 5 bytes
//   - 4 q8_1 sub-blocks per spiral block, sub-block qb spans groups [qb*4 .. qb*4+3]
//   - For each output column j, accumulate scaled int dots over 4 sub-blocks
//
// Centroid lookup happens once per group (16 groups) and is reused across all
// ncols_dst columns — same optimization as INT4.

template <int ncols_dst>
__device__ __forceinline__ void spiral_int5_dot_block_multi(
        const block_spiral_int5 * __restrict__ blk,
        const block_q8_1 * __restrict__ vy_q8,
        int ib,
        int stride_col_y,
        float (& block_sums)[ncols_dst]) {

    const uint8_t * qs = blk->qs;  // 80 bytes = 16 groups of 5

    // Unpack all 16 groups' centroids once. 16 × 2 = 32 packed int32s.
    int c_lo[16], c_hi[16];
    #pragma unroll
    for (int g = 0; g < 16; g++) {
        spiral_int5_group_to_centroids(qs + g * 5, c_lo[g], c_hi[g]);
    }

    // For each output column, walk the 4 q8_1 sub-blocks accumulating scaled int dots.
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) {
        const block_q8_1 * a_subs = vy_q8 + j * stride_col_y + ib * 4;

        float col_sum = 0.0f;
        #pragma unroll
        for (int qb = 0; qb < 4; qb++) {
            const block_q8_1 & a = a_subs[qb];
            const float d_act = __half2float((__half) a.ds.x);
            const int * a_qs = (const int *) a.qs;

            int sum_int = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const int g_idx = qb * 4 + g;
                sum_int = __dp4a(c_lo[g_idx], a_qs[g * 2 + 0], sum_int);
                sum_int = __dp4a(c_hi[g_idx], a_qs[g * 2 + 1], sum_int);
            }

            col_sum += d_act * (float) sum_int;
        }

        block_sums[j] += col_sum;
    }
}

// ============================================================================
// SPIRAL_INT5 mul_mat_vec_q kernel (templated on ncols_dst)
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_vec_q_spiral_int5(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst,
        const float recovery_scale) {

    const int row = blockIdx.x * MMVQ_SPIRAL_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / 128;
    const block_spiral_int5 * x_row =
        ((const block_spiral_int5 *) vx) + (int64_t) row * blocks_per_row;

    const float norm = __half2float(x_row[0].norm);

    float sumf[ncols_dst];
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++) sumf[j] = 0.0f;

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        spiral_int5_dot_block_multi<ncols_dst>(&x_row[ib], vy_q8, ib, stride_col_y, sumf);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
        }
    }

    if (lane == 0) {
        const float scale = recovery_scale * norm;
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            dst[j * stride_col_dst + row] = sumf[j] * scale;
        }
    }
}

// ============================================================================
// Launch wrapper (INT5) — instantiates the kernel template for given ncols_dst
// ============================================================================

template <int ncols_dst>
static void launch_spiral_int5(
        const void * src0_d,
        const block_q8_1 * q8_buf,
        float * dst_d,
        int ncols_x,
        int nrows_x,
        int stride_col_y,
        int stride_col_dst,
        float recovery_scale,
        cudaStream_t stream) {

    const dim3 block(WARP_SIZE, MMVQ_SPIRAL_NWARPS);
    const dim3 grid((nrows_x + MMVQ_SPIRAL_NWARPS - 1) / MMVQ_SPIRAL_NWARPS);
    mul_mat_vec_q_spiral_int5<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, recovery_scale);
}

// ============================================================================
// Entry point
// ============================================================================

void ggml_cuda_mul_mat_spiral(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    // Contract assertions
    GGML_ASSERT(src0->type == GGML_TYPE_SPIRAL_INT4 ||
                src0->type == GGML_TYPE_SPIRAL_INT5);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % 128 == 0);

    const int64_t ne00 = src0->ne[0];   // input dim (= ncols_x)
    const int64_t ne01 = src0->ne[1];   // output dim (= nrows_x)
    const int64_t ne10 = src1->ne[0];   // input dim of activation
    const int64_t ne11 = src1->ne[1];   // ncols_dst
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne00 == ne10);          // input dims must match
    GGML_ASSERT(ne12 == 1 && ne13 == 1);// batch handling not yet supported

    // Step 2f: INT4 and INT5 with ncols_dst ∈ [1..8] are implemented.
    // The caller (ggml_cuda_mul_mat dispatch) guarantees ne11 <= MMVQ_MAX_BATCH_SIZE.
    GGML_ASSERT(ne11 >= 1 && ne11 <= 8);

    // Thread-safe one-time init of __constant__ centroid tables.
    std::call_once(SPIRAL_CENTROID_INIT_FLAG, spiral_centroid_init);

    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;

    // Activation quantization: produce padded block_q8_1 buffer.
    // The standard llama.cpp mmvq path pads ne10 to MATRIX_ROW_PADDING.
    // Our Spiral block size (128) divides MATRIX_ROW_PADDING (≥ 256), so any
    // padding produces complete extra spiral blocks that we ignore (kernel
    // only loops over ne00/128 real blocks).
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t n_q8_blocks_padded = ne10_padded * ne11 * ne12 * ne13 / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> q8_buf(ctx.pool(id), n_q8_blocks_padded);

    // Strides for src1: standard llama.cpp convention is nbXX / sizeof(float).
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    quantize_row_q8_1_cuda(
        src1_d, /*ids=*/nullptr, q8_buf.get(),
        src0->type,
        /*ne00=*/ne10,
        /*s01=*/nb11 / sizeof(float),
        /*s02=*/nb12 / sizeof(float),
        /*s03=*/nb13 / sizeof(float),
        /*ne0 =*/ne10_padded,
        /*ne1 =*/ne11,
        /*ne2 =*/ne12,
        /*ne3 =*/ne13,
        stream);
    CUDA_CHECK(cudaGetLastError());

    // Strides for kernel:
    //   stride_col_y: q8_1 blocks per activation column = ne10_padded / QK8_1
    //   stride_col_dst: floats per output column = nrows_x (column-major dst layout)
    const int stride_col_y   = (int) (ne10_padded / QK8_1);
    const int stride_col_dst = (int) ne01;
    const int ncols_x        = (int) ne00;
    const int nrows_x        = (int) ne01;

    // Dispatch via ncols_dst (1..8) and type. The compiler instantiates 16 kernels
    // total (2 types × 8 ncols_dst); each is selected by this runtime switch.
    if (src0->type == GGML_TYPE_SPIRAL_INT4) {
        const float scale = SPIRAL_INT4_RECOVERY_SCALE;
        switch ((int) ne11) {
            case 1: launch_spiral_int4<1>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 2: launch_spiral_int4<2>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 3: launch_spiral_int4<3>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 4: launch_spiral_int4<4>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 5: launch_spiral_int4<5>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 6: launch_spiral_int4<6>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 7: launch_spiral_int4<7>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 8: launch_spiral_int4<8>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
        }
    } else {
        const float scale = SPIRAL_INT5_RECOVERY_SCALE;
        switch ((int) ne11) {
            case 1: launch_spiral_int5<1>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 2: launch_spiral_int5<2>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 3: launch_spiral_int5<3>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 4: launch_spiral_int5<4>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 5: launch_spiral_int5<5>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 6: launch_spiral_int5<6>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 7: launch_spiral_int5<7>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
            case 8: launch_spiral_int5<8>(src0_d, q8_buf.get(), dst_d,
                        ncols_x, nrows_x, stride_col_y, stride_col_dst, scale, stream); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
