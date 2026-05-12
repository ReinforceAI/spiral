/*
 * Fused mul_mat for SPIRAL_INT4 / SPIRAL_INT5 weight types.
 *
 * Spiral v4 (Bible 13 §18): dense-QR-rotated 4-bit / 5-bit MoE expert weights.
 * Lloyd-Max-N(0,1) centroids (16 levels for INT4, 32 for INT5). Per-block fp16
 * row norm. Block size 128.
 *
 * The rotation R^T @ activation is inserted upstream as a GGML_OP_MUL_MAT in
 * the graph (see spiral_rotate_activation in llama-graph.cpp). This kernel
 * receives activations already in the rotated basis — it does NOT rotate.
 *
 * Implementation: pure-float arithmetic, mirroring the Metal reference kernel
 * (ggml-metal.metal:5234-5450). 32-thread warp per row, 4 block lanes × 8
 * chunks, raw f32 activation × f32 centroid lookup × float accumulation, then
 * per-row fp16 norm. Same INFRASTRUCTURE as TurboQuant (entry shape, multi-
 * column templating) but Spiral-native COMPUTATION — no q8_1 quantization,
 * no int8 centroids, no dp4a, because those are TurboQuant-specific
 * optimizations that don't match Spiral's float-arithmetic contract.
 *
 * ne[1] = 1 (decode):    pure-float kernel, 1 col
 * ne[1] ≤ 8 (multi-tok): same kernel templated on ncols_dst, weight loads
 *                        amortized across columns
 * ne[1] > 8 (prefill):   handled separately (cuBLAS dequant path — TBD)
 */

#pragma once

#include "common.cuh"

// Spiral mul_mat entry point.
//
// Contract:
//   src0->type ∈ {GGML_TYPE_SPIRAL_INT4, GGML_TYPE_SPIRAL_INT5}
//   src1->type == GGML_TYPE_F32  (activation, already in rotated basis)
//   dst->type  == GGML_TYPE_F32
//   src0->ne[0] must be % 128 == 0 (QK_SPIRAL_INT4 = QK_SPIRAL_INT5 = 128)
//   src1->ne[1] <= MMVQ_MAX_BATCH_SIZE = 8 (caller-enforced)
//
// Algorithm:
//   1. For each output row, a 32-thread warp computes the dot product.
//   2. 4 thread-lanes (ix = lane/8) iterate every 4th weight block.
//   3. Each lane handles 8 chunks (it = lane%8) of 16 elements within a block.
//   4. Unpack codes → look up float centroid → multiply by raw f32 activation
//      → accumulate in float. After loop, multiply by per-row fp16 norm.
//   5. Warp shuffle-reduce all 32 partials, lane 0 stores the result.
void ggml_cuda_mul_mat_spiral(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

// MoE variant — graph-capture-compatible Spiral mul_mat_id.
//
// Contract:
//   src0->type ∈ {GGML_TYPE_SPIRAL_INT4, GGML_TYPE_SPIRAL_INT5}, shape [ncols_x, nrows_x, n_experts]
//   src1->type == GGML_TYPE_F32, shape [ncols_x, 1, n_tokens] (one row per token)
//   ids->type  == GGML_TYPE_I32, shape [n_expert_used, n_tokens]
//   dst->type  == GGML_TYPE_F32, shape [nrows_x, n_expert_used, n_tokens]
//
// Each warp computes one (output_row, expert_slot, token) triple. Reads the
// expert index from `ids` on device — no host synchronization. Replaces the
// host-sync fallback in upstream `ggml_cuda_mul_mat_id` for Spiral types.
void ggml_cuda_mul_mat_spiral_id(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst);
