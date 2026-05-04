#!/usr/bin/env python
"""
spiral_test_gen.py
==================

Generate test fixtures for standalone SPIRAL_3BIT mul_mm kernel testing.

Produces deterministic binary files that the C++ standalone tester reads:

  fixture/
    meta.json         — shape and parameters
    x_rotated.bin     — rotated input activation, F32, shape [n, k]
    weight_packed.bin — SPIRAL_3BIT packed weights, shape [m * (k/128) * 50] bytes
    y_reference.bin   — Python ground truth W_natural @ x_natural, F32, shape [n, m]
    debug_info.txt    — human-readable summary

The C++ tester:
  1. Reads x_rotated.bin and weight_packed.bin
  2. Calls kernel_mul_mm_tq_rotated via Metal API
  3. Writes its output to y_kernel.bin

Then verify_spiral_kernel.py:
  1. Loads y_kernel.bin and y_reference.bin
  2. Computes per-element error
  3. Reports PASS/FAIL with detailed diagnostics

Usage:
  python spiral_test_gen.py --m 2048 --k 2048 --n 13 --out fixture/
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


QK_SPIRAL_3BIT = 128
SPIRAL_3BIT_BLOCK_BYTES = 50  # 2 (norm fp16) + 48 (3-bit codes)


# Centroid values — MUST match writer's lloyd_max_n01(8)
# These are the Lloyd-Max optimal centroids for unit Gaussian, 8 levels
# Verified to match the centroids hardcoded in ggml-metal.metal kernel.
CENTROIDS = np.array([
    -2.1556325, -1.3483801, -0.7599415, -0.2466970,
     0.2466970,  0.7599415,  1.3483801,  2.1556325,
], dtype=np.float32)


# ── MPB rotation generation (deterministic from seed) ──────────────────
def generate_mpb_rotation(seed: int, d: int, block_size: int = 128):
    n_passes = max(2, int(math.ceil(math.log(d) / math.log(block_size))))
    rng = np.random.default_rng(seed + d)
    signs_list = []
    perms_list = []
    inv_perms_list = []
    for p in range(n_passes):
        signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        signs_list.append(signs)
        if p < n_passes - 1:
            perm = rng.permutation(d).astype(np.int64)
            inv_perm = np.argsort(perm).astype(np.int64)
            perms_list.append(perm)
            inv_perms_list.append(inv_perm)
    return n_passes, signs_list, perms_list, inv_perms_list


# ── FWHT block-wise (orthonormal) ──────────────────────────────────────
def fwht_blocks(x: np.ndarray, block_size: int) -> np.ndarray:
    out = x.copy()
    n_blocks = out.shape[-1] // block_size
    flat = out.reshape(*out.shape[:-1], n_blocks, block_size)
    h = 1
    while h < block_size:
        for i in range(0, block_size, h * 2):
            a = flat[..., i:i+h].copy()
            b = flat[..., i+h:i+2*h].copy()
            flat[..., i:i+h]      = a + b
            flat[..., i+h:i+2*h]  = a - b
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


# ── Quantize FP32 weight (already rotated) to SPIRAL_3BIT format ───────
def quantize_to_spiral3bit(W_rotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    W_rotated: shape [m, k], FP32, already in rotated basis.
    Returns:
      row_norms: shape [m], FP16
      packed_codes: shape [m, k * 3 // 8], uint8
    """
    m, k = W_rotated.shape
    assert k % QK_SPIRAL_3BIT == 0, f"k={k} must be divisible by {QK_SPIRAL_3BIT}"

    # Per-row L2 norm. GGUF stores it pre-divided by sqrt(k):
    #   gguf_norm[i] = ||W_rotated[i]|| / sqrt(k)  ≈  std of row i
    # The kernel decodes as:  decoded = centroid[code] * gguf_norm[i]
    # So we want: centroid[code] ≈ W_rotated[i,j] / gguf_norm[i]
    row_norms_full = np.linalg.norm(W_rotated, axis=1)  # ||W[i]||, shape [m]
    row_norms = (row_norms_full / math.sqrt(k)).astype(np.float16)
    row_norms_f32 = row_norms.astype(np.float32)  # [m]

    # Avoid division by zero for any row that's accidentally all zeros
    eps = 1e-12
    scales = row_norms_f32 + eps  # [m] — divide W by this directly (NOT by * sqrt(k))
    W_normalized = W_rotated / scales[:, None]  # [m, k], should be ~unit variance

    # Find nearest centroid for each value
    # CENTROIDS shape [8], W_normalized shape [m, k]
    diffs = W_normalized[..., None] - CENTROIDS[None, None, :]
    codes = np.argmin(diffs ** 2, axis=-1).astype(np.uint8)  # [m, k] in 0..7

    # Sanity check: reconstruction error on row 0
    decoded_row0 = CENTROIDS[codes[0]] * row_norms_f32[0]
    rel_err_row0 = np.linalg.norm(decoded_row0 - W_rotated[0]) / (np.linalg.norm(W_rotated[0]) + 1e-12)
    if rel_err_row0 > 0.5:
        print(f"  WARNING: row 0 quant rel_err = {rel_err_row0:.4f}, expected ~0.19 for Lloyd-Max 3-bit")
        print(f"  This means the quantization formula is wrong.")

    # Pack codes: 8 codes (3 bits each) -> 3 bytes
    packed = pack_3bit_codes(codes)  # [m, k * 3 // 8]
    return row_norms, packed


def pack_3bit_codes(codes: np.ndarray) -> np.ndarray:
    """codes: [m, k] uint8, values 0..7. Returns [m, k*3//8] uint8."""
    m, k = codes.shape
    assert k % 8 == 0
    out = np.zeros((m, k * 3 // 8), dtype=np.uint8)
    for g in range(k // 8):
        # codes[:, g*8 + 0..7], pack into 3 bytes
        c0, c1, c2, c3, c4, c5, c6, c7 = [codes[:, g*8 + i] for i in range(8)]
        b0 = (c0 & 0x7) | ((c1 & 0x7) << 3) | ((c2 & 0x3) << 6)
        b1 = ((c2 >> 2) & 0x1) | ((c3 & 0x7) << 1) | ((c4 & 0x7) << 4) | ((c5 & 0x1) << 7)
        b2 = ((c5 >> 1) & 0x3) | ((c6 & 0x7) << 2) | ((c7 & 0x7) << 5)
        out[:, g*3 + 0] = b0
        out[:, g*3 + 1] = b1
        out[:, g*3 + 2] = b2
    return out


def write_gguf_spiral_block_format(row_norms: np.ndarray, packed_codes: np.ndarray) -> bytes:
    """
    Produce the GGUF SPIRAL_3BIT block layout:
    For each row, n_blocks = k / 128 blocks of 50 bytes each.
    First block of each row has the row_norm in its first 2 bytes.
    Subsequent blocks have garbage in those 2 bytes (kernel ignores).
    Codes occupy bytes 2..50 of each block (48 bytes = 128 codes * 3 / 8).

    row_norms: [m] FP16
    packed_codes: [m, k*3//8] uint8
    Returns: bytes object of size m * (k/128) * 50
    """
    m, codes_per_row = packed_codes.shape
    cpb = QK_SPIRAL_3BIT * 3 // 8  # 48 codes-bytes per block
    nb = codes_per_row // cpb       # blocks per row

    out = np.zeros((m, nb, SPIRAL_3BIT_BLOCK_BYTES), dtype=np.uint8)
    # Place row_norms in block 0 of each row (first 2 bytes as FP16)
    norm_bytes = row_norms.tobytes()  # [m * 2] bytes
    norm_arr = np.frombuffer(norm_bytes, dtype=np.uint8).reshape(m, 2)
    out[:, 0, :2] = norm_arr
    # Place codes bytes 2..50 of each block
    codes_reshaped = packed_codes.reshape(m, nb, cpb)
    out[:, :, 2:50] = codes_reshaped
    return out.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, required=True, help='output dim (rows of weight)')
    p.add_argument('--k', type=int, required=True, help='input dim (cols of weight, must be % 128 == 0)')
    p.add_argument('--n', type=int, required=True, help='number of input columns (tokens)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1. Generate random unrotated input x: shape [n, k] FP32
    x_natural = rng.standard_normal((args.n, args.k), dtype=np.float32) * 0.5
    print(f"x_natural: shape={x_natural.shape}, norm={np.linalg.norm(x_natural):.4f}")

    # 2. Generate random unrotated weight W: shape [m, k] FP32
    W_natural = rng.standard_normal((args.m, args.k), dtype=np.float32) * (1.0 / math.sqrt(args.k))
    print(f"W_natural: shape={W_natural.shape}, norm={np.linalg.norm(W_natural):.4f}")

    # 3. Compute the unquantized "true" output: y_natural = x_natural @ W_natural.T
    #    This is for diagnostic only. The actual y_reference (what kernel must
    #    match) is computed AFTER quantization below.
    y_natural = (x_natural @ W_natural.T).astype(np.float32)
    print(f"y_natural (FP32 truth): norm={np.linalg.norm(y_natural):.4f}, "
          f"range=[{y_natural.min():.4f}, {y_natural.max():.4f}]")

    # 4. Generate rotation params for dim k
    n_passes, signs_list, perms_list, inv_perms_list = generate_mpb_rotation(args.seed, args.k)
    print(f"Rotation: {n_passes} passes, dim={args.k}")

    # 5. Apply forward rotation to x: x_rotated = R(x)
    x_rotated = mpb_forward(x_natural, signs_list, perms_list).astype(np.float32)
    print(f"x_rotated: shape={x_rotated.shape}, norm={np.linalg.norm(x_rotated):.4f}")

    # Verify rotation is orthonormal
    norm_diff = abs(np.linalg.norm(x_natural) - np.linalg.norm(x_rotated))
    assert norm_diff < 1e-2, f"Rotation not orthonormal! norm diff = {norm_diff}"
    print(f"  ✓ Rotation preserves norm (diff = {norm_diff:.6e})")

    # 6. Apply rotation to weight: W_rotated = W @ R^T
    #    (so that W_rotated @ R(x) = W @ R^T @ R @ x = W @ x  [since R is orthonormal])
    #    R^T applied to a row vector means: multiply on right by R.T
    #    For [m, k] W matrix, each row is rotated: W_rotated[i] = R(W[i])
    W_rotated = mpb_forward(W_natural, signs_list, perms_list).astype(np.float32)
    print(f"W_rotated: shape={W_rotated.shape}, norm={np.linalg.norm(W_rotated):.4f}")

    # Sanity check: W_rotated @ x_rotated.T should equal W_natural @ x_natural.T (orthonormal)
    y_check = (x_rotated @ W_rotated.T).astype(np.float32)
    rel_err_sanity = np.linalg.norm(y_check - y_natural) / np.linalg.norm(y_natural)
    print(f"  Sanity check (FP32, no quantization): rel_err = {rel_err_sanity:.6e}")
    assert rel_err_sanity < 1e-3, f"Rotation math broken: rel_err={rel_err_sanity}"
    print(f"  ✓ Rotation math is correct")

    # 7. Quantize W_rotated to SPIRAL_3BIT
    row_norms, packed_codes = quantize_to_spiral3bit(W_rotated)
    print(f"Quantized: row_norms shape={row_norms.shape}, packed shape={packed_codes.shape}")

    # 8. Reconstruct W_rotated from quantized format — this is what the
    #    kernel will produce internally during dequantization. The kernel's
    #    output should match (rotated_x @ W_rotated_decoded.T) to within
    #    floating-point noise, NOT to within Lloyd-Max quantization error.
    W_rotated_decoded = np.zeros_like(W_rotated)
    nb = args.k // QK_SPIRAL_3BIT
    cpb = QK_SPIRAL_3BIT * 3 // 8
    row_norms_f32 = row_norms.astype(np.float32)
    for i in range(args.m):
        codes_i = decode_3bit_row(packed_codes[i], args.k)  # [k]
        W_rotated_decoded[i] = CENTROIDS[codes_i] * row_norms_f32[i]

    # The TRUE reference for the kernel is: rotated_x @ W_rotated_decoded.T
    # because that's exactly what the kernel computes (after dequant)
    y_reference = (x_rotated @ W_rotated_decoded.T).astype(np.float32)
    print(f"y_reference (post-quant): shape={y_reference.shape}, "
          f"norm={np.linalg.norm(y_reference):.4f}, "
          f"range=[{y_reference.min():.4f}, {y_reference.max():.4f}]")

    # Also compute the unquantized true answer for diagnostic purposes
    quant_error = np.linalg.norm(y_reference - y_natural) / np.linalg.norm(y_natural)
    print(f"  (Quantization error vs FP32: {quant_error:.4f}; expected ~0.19 for Lloyd-Max)")

    # 9. Write outputs
    blob = write_gguf_spiral_block_format(row_norms, packed_codes)
    print(f"Weight blob: {len(blob)} bytes (expected {args.m * nb * SPIRAL_3BIT_BLOCK_BYTES})")

    (out_dir / "x_rotated.bin").write_bytes(x_rotated.tobytes())
    (out_dir / "weight_packed.bin").write_bytes(blob)
    (out_dir / "y_reference.bin").write_bytes(y_reference.tobytes())

    meta = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "seed": args.seed,
        "block_size": QK_SPIRAL_3BIT,
        "block_bytes": SPIRAL_3BIT_BLOCK_BYTES,
        "n_blocks_per_row": nb,
        "weight_blob_bytes": len(blob),
        "x_rotated_bytes": x_rotated.nbytes,
        "y_reference_bytes": y_reference.nbytes,
        "x_rotated_layout": "F32, contiguous, shape [n, k] in row-major (n outer, k inner)",
        "y_reference_layout": "F32, contiguous, shape [n, m] in row-major",
        "weight_blob_layout": (
            f"For each row r in 0..{args.m}: {nb} blocks of {SPIRAL_3BIT_BLOCK_BYTES} bytes."
            " Each block: 2 bytes FP16 row_norm (only block 0 used) + 48 bytes packed 3-bit codes."
        ),
        "centroids": CENTROIDS.tolist(),
        "rotation_seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    debug = []
    debug.append(f"shape: m={args.m} k={args.k} n={args.n}")
    debug.append(f"y_reference[0, :8]: {y_reference[0, :8].tolist()}")
    debug.append(f"y_reference[0] norm: {np.linalg.norm(y_reference[0]):.6f}")
    debug.append(f"x_rotated[0, :8]: {x_rotated[0, :8].tolist()}")
    debug.append(f"row_norms[0]: {float(row_norms[0]):.6f}")
    debug.append(f"first 16 packed bytes of row 0: {packed_codes[0, :16].tolist()}")
    (out_dir / "debug_info.txt").write_text("\n".join(debug))

    print()
    print(f"Wrote fixtures to {out_dir}/")
    print(f"  meta.json")
    print(f"  x_rotated.bin       ({x_rotated.nbytes} bytes)")
    print(f"  weight_packed.bin   ({len(blob)} bytes)")
    print(f"  y_reference.bin     ({y_reference.nbytes} bytes)")
    print(f"  debug_info.txt")


def decode_3bit_row(packed_row: np.ndarray, k: int) -> np.ndarray:
    """Decode one row of packed 3-bit codes back to [k] uint8."""
    codes = np.zeros(k, dtype=np.uint8)
    for g in range(k // 8):
        b0, b1, b2 = packed_row[g*3 + 0], packed_row[g*3 + 1], packed_row[g*3 + 2]
        codes[g*8 + 0] = b0 & 0x7
        codes[g*8 + 1] = (b0 >> 3) & 0x7
        codes[g*8 + 2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        codes[g*8 + 3] = (b1 >> 1) & 0x7
        codes[g*8 + 4] = (b1 >> 4) & 0x7
        codes[g*8 + 5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        codes[g*8 + 6] = (b2 >> 2) & 0x7
        codes[g*8 + 7] = (b2 >> 5) & 0x7
    return codes


if __name__ == "__main__":
    sys.exit(main() or 0)
