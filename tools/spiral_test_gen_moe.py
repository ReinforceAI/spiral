#!/usr/bin/env python
"""
spiral_test_gen_moe.py
=======================

Generate fixtures for testing kernel_mul_mv_id_spiral_3bit_f32 — the MoE
variant of the SPIRAL_3BIT matmul kernel. Used by ffn_gate_exps,
ffn_up_exps, ffn_down_exps in qwen35moe.

The MoE matmul: for each token, select n_active experts (e.g., 8 of 256),
matmul the token's activation against each selected expert's weight, sum.

Usage:
  python spiral_test_gen_moe.py --m 512 --k 2048 --n_tokens 1 \\
                                 --n_experts 256 --n_active 8 \\
                                 --out fixture_moe/

Outputs:
  meta.json
  weight_packed.bin   — [n_experts][m][k_packed/2] SPIRAL_3BIT, contiguous
  x_rotated.bin       — [n_tokens][k] F32 (rotated input, applies to all
                                              expert calls for this token)
  ids.bin             — [n_tokens][n_active] int32 expert indices
  y_reference.bin     — [n_tokens][m] F32 (sum over active experts)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

QK_SPIRAL_3BIT = 128
SPIRAL_3BIT_BLOCK_BYTES = 50

CENTROIDS = np.array([
    -2.1556325, -1.3483801, -0.7599415, -0.2466970,
     0.2466970,  0.7599415,  1.3483801,  2.1556325,
], dtype=np.float32)


def generate_mpb_rotation(seed: int, d: int, block_size: int = 128):
    n_passes = max(2, int(math.ceil(math.log(d) / math.log(block_size))))
    rng = np.random.default_rng(seed + d)
    signs_list = []
    perms_list = []
    for p in range(n_passes):
        signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        signs_list.append(signs)
        if p < n_passes - 1:
            perm = rng.permutation(d).astype(np.int64)
            perms_list.append(perm)
    return signs_list, perms_list


def fwht_blocks(x: np.ndarray, block_size: int) -> np.ndarray:
    out = x.copy()
    n_blocks = out.shape[-1] // block_size
    flat = out.reshape(*out.shape[:-1], n_blocks, block_size)
    h = 1
    while h < block_size:
        for i in range(0, block_size, h * 2):
            a = flat[..., i:i+h].copy()
            b = flat[..., i+h:i+2*h].copy()
            flat[..., i:i+h] = a + b
            flat[..., i+h:i+2*h] = a - b
        h *= 2
    flat = flat / math.sqrt(block_size)
    return flat.reshape(*out.shape)


def mpb_forward(x: np.ndarray, signs_list, perms_list, block_size=128):
    out = x.copy()
    n_passes = len(signs_list)
    for p in range(n_passes):
        out = out * signs_list[p]
        out = fwht_blocks(out, block_size)
        if p < n_passes - 1:
            out = out[..., perms_list[p]]
    return out


def quantize_to_spiral3bit(W_rotated: np.ndarray):
    """W_rotated: [m, k] FP32. Returns row_norms[m] FP16, packed[m, k*3//8]."""
    m, k = W_rotated.shape
    row_norms_full = np.linalg.norm(W_rotated, axis=1)
    row_norms = (row_norms_full / math.sqrt(k)).astype(np.float16)
    row_norms_f32 = row_norms.astype(np.float32) + 1e-12
    W_norm = W_rotated / row_norms_f32[:, None]
    diffs = W_norm[..., None] - CENTROIDS[None, None, :]
    codes = np.argmin(diffs ** 2, axis=-1).astype(np.uint8)
    packed = pack_3bit_codes(codes)
    return row_norms, packed


def pack_3bit_codes(codes):
    m, k = codes.shape
    out = np.zeros((m, k * 3 // 8), dtype=np.uint8)
    for g in range(k // 8):
        c = [codes[:, g*8 + i] for i in range(8)]
        out[:, g*3+0] = (c[0] & 0x7) | ((c[1] & 0x7) << 3) | ((c[2] & 0x3) << 6)
        out[:, g*3+1] = ((c[2] >> 2) & 0x1) | ((c[3] & 0x7) << 1) | ((c[4] & 0x7) << 4) | ((c[5] & 0x1) << 7)
        out[:, g*3+2] = ((c[5] >> 1) & 0x3) | ((c[6] & 0x7) << 2) | ((c[7] & 0x7) << 5)
    return out


def write_block_format(row_norms, packed_codes):
    m, codes_per_row = packed_codes.shape
    cpb = QK_SPIRAL_3BIT * 3 // 8
    nb = codes_per_row // cpb
    out = np.zeros((m, nb, SPIRAL_3BIT_BLOCK_BYTES), dtype=np.uint8)
    norm_arr = np.frombuffer(row_norms.tobytes(), dtype=np.uint8).reshape(m, 2)
    out[:, :, :2] = norm_arr[:, None, :]
    out[:, :, 2:50] = packed_codes.reshape(m, nb, cpb)
    return out.tobytes()


def decode_3bit_row(packed_row, k):
    codes = np.zeros(k, dtype=np.uint8)
    for g in range(k // 8):
        b0, b1, b2 = packed_row[g*3+0], packed_row[g*3+1], packed_row[g*3+2]
        codes[g*8+0] = b0 & 0x7
        codes[g*8+1] = (b0 >> 3) & 0x7
        codes[g*8+2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        codes[g*8+3] = (b1 >> 1) & 0x7
        codes[g*8+4] = (b1 >> 4) & 0x7
        codes[g*8+5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        codes[g*8+6] = (b2 >> 2) & 0x7
        codes[g*8+7] = (b2 >> 5) & 0x7
    return codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m',         type=int, required=True, help='output dim per expert (e.g. 512)')
    p.add_argument('--k',         type=int, required=True, help='input dim (e.g. 2048)')
    p.add_argument('--n_tokens',  type=int, required=True, help='tokens (1 for decode)')
    p.add_argument('--n_experts', type=int, required=True, help='total experts (e.g. 256)')
    p.add_argument('--n_active',  type=int, required=True, help='active per token (e.g. 8)')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--out',       type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Inputs
    x_natural = rng.standard_normal((args.n_tokens, args.k), dtype=np.float32) * 0.5
    print(f"x_natural: {x_natural.shape}, norm={np.linalg.norm(x_natural):.4f}")

    # Generate weights for each expert
    W_natural = rng.standard_normal((args.n_experts, args.m, args.k),
                                    dtype=np.float32) * (1.0 / math.sqrt(args.k))
    print(f"W_natural: {W_natural.shape}")

    # Random expert IDs for each token
    ids = np.zeros((args.n_tokens, args.n_active), dtype=np.int32)
    for t in range(args.n_tokens):
        ids[t] = rng.choice(args.n_experts, size=args.n_active, replace=False)
    print(f"ids[0]: {ids[0].tolist()}")

    # Rotation
    signs_list, perms_list = generate_mpb_rotation(args.seed, args.k)
    x_rotated = mpb_forward(x_natural, signs_list, perms_list).astype(np.float32)
    norm_diff = abs(np.linalg.norm(x_natural) - np.linalg.norm(x_rotated))
    assert norm_diff < 1e-2
    print(f"x_rotated: norm preserved (diff={norm_diff:.2e})")

    # Rotate every expert's weights
    W_rotated_all = np.zeros_like(W_natural)
    for e in range(args.n_experts):
        W_rotated_all[e] = mpb_forward(W_natural[e], signs_list, perms_list).astype(np.float32)
    print(f"All experts rotated.")

    # Quantize each expert's weight to SPIRAL_3BIT
    blob_list = []
    W_decoded_all = np.zeros_like(W_natural)  # for ground-truth computation
    for e in range(args.n_experts):
        row_norms, packed = quantize_to_spiral3bit(W_rotated_all[e])
        # Reconstruct decoded weight to use for ground truth
        rn_f32 = row_norms.astype(np.float32)
        for r in range(args.m):
            codes_r = decode_3bit_row(packed[r], args.k)
            W_decoded_all[e, r] = CENTROIDS[codes_r] * rn_f32[r]
        blob_list.append(write_block_format(row_norms, packed))

    blob = b''.join(blob_list)
    expected_size = args.n_experts * args.m * (args.k // 128) * 50
    print(f"weight_packed.bin: {len(blob)} bytes (expected {expected_size})")
    assert len(blob) == expected_size

    # Ground truth: for each token, sum over active experts
    # y[t, m_idx] = sum_{e in ids[t]} (W_decoded_all[e] @ x_rotated[t]).T
    y_reference = np.zeros((args.n_tokens, args.m), dtype=np.float32)
    for t in range(args.n_tokens):
        for e_idx in ids[t]:
            y_reference[t] += W_decoded_all[e_idx] @ x_rotated[t]
    print(f"y_reference: {y_reference.shape}, norm={np.linalg.norm(y_reference):.4f}")

    # Write outputs
    (args.out / "weight_packed.bin").write_bytes(blob)
    (args.out / "x_rotated.bin").write_bytes(x_rotated.astype(np.float32).tobytes())
    (args.out / "ids.bin").write_bytes(ids.tobytes())
    (args.out / "y_reference.bin").write_bytes(y_reference.tobytes())

    meta = {
        "m": args.m,
        "k": args.k,
        "n_tokens": args.n_tokens,
        "n_experts": args.n_experts,
        "n_active": args.n_active,
        "seed": args.seed,
        "weight_blob_bytes": len(blob),
        "x_rotated_bytes": x_rotated.nbytes,
        "ids_bytes": ids.nbytes,
        "y_reference_bytes": y_reference.nbytes,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote fixtures to {args.out}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
