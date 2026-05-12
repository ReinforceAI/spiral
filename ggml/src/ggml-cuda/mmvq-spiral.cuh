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
 * (Contrast with TurboQuant, which pre-rotates inside the kernel.)
 *
 * ne[1] = 1 (decode):    int8-MMA dp4a path
 * ne[1] ≤ 8 (multi-tok): int8-MMA dp4a path with weight reuse across tokens
 * ne[1] > 8 (prefill):   handled separately (TBD — see Bible XV §8)
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
//   src1->ne[1] <= MMVQ_MAX_BATCH_SIZE (caller-enforced; see ggml_cuda_mul_mat dispatch)
//
// Algorithm:
//   1. Pre-quantize the activation to block_q8_1 (32-element sub-blocks, per-block scale).
//   2. Dispatch to a templated kernel parameterized on (type, ncols_dst).
//   3. Per spiral block: unpack codes, look up int8 centroids, __dp4a against int8 activation.
//   4. Accumulate per-row across blocks, apply row_norm * centroid_scale at the end.
//   5. Warp reduction, lane-0 stores result.
void ggml_cuda_mul_mat_spiral(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);
