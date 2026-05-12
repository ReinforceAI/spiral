#!/usr/bin/env python
"""
spiral_test_gen_int4.py
========================

Generate fixtures for testing the SPIRAL_INT4 CUDA mul_mat kernel.

Mirrors spiral_test_gen.py (which is for the v3 SPIRAL_3BIT path) but adapted
for v4 INT4: 16 Lloyd-Max centroids, 66-byte blocks (2 norm + 64 packed),
2 codes per byte (low nibble | high nibble).

For initial kernel correctness validation, we generate WITHOUT rotation —
both x and W are passed unrotated, so the kernel's output equals
(x @ W_decoded.T) directly. This isolates kernel math from rotation handling.

Usage:
  python spiral_test_gen_int4.py --m 2048 --k 2048 --n 1 --out fixture_int4/

Output layout matches block_spiral_int4 from ggml-common.h:
  per block: 2-byte fp16 norm + 64 bytes packed 4-bit codes = 66 bytes
  per row: (k / 128) blocks
  total weight: m * (k / 128) * 66 bytes

Then run on H100:
  ./build/bin/spiral_kernel_test fixture_int4/      # writes y_kernel.bin
  python verify_spiral_kernel.py fixture_int4/      # compares
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


QK_SPIRAL_INT4 = 128
SPIRAL_INT4_BLOCK_BYTES = 66  # 2-byte fp16 norm + 64 bytes of 4-bit codes


# Lloyd-Max N(0,1) centroids — verbatim from codebooks.py:lloyd_max_n01(16).
# Verified bit-identical to mmvq-spiral.cu hardcoded values to 5e-8.
def lloyd_max_n01(n_levels: int, n_iter: int = 200) -> np.ndarray:
    """Identical to codebooks.py lloyd_max_n01. Returns float32 centroids sorted."""
    grid = np.linspace(-6, 6, 8192)
    pdf = np.exp(-0.5 * grid ** 2) / math.sqrt(2 * math.pi)
    w = pdf * (grid[1] - grid[0])
    if n_levels % 2 == 0:
        half = np.linspace(0.5, 2.5, n_levels // 2)
        c = np.concatenate([-half[::-1], half])
    else:
        half = np.linspace(0.5, 2.5, n_levels // 2)
        c = np.concatenate([-half[::-1], [0.0], half])
    c = np.sort(c)
    for _ in range(n_iter):
        b = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(b, grid)
        nc = c.copy()
        for k in range(n_levels):
            m = (idx == k)
            mass = w[m].sum()
            if mass > 1e-15:
                nc[k] = (grid[m] * w[m]).sum() / mass
        if np.max(np.abs(nc - c)) < 1e-10:
            break
        c = nc
    return np.sort(c).astype(np.float32)


CENTROIDS_INT4 = lloyd_max_n01(16)


def quantize_to_int4(W: np.ndarray):
    """W: [m, k] FP32. Returns row_norms[m] FP16, packed[m, k//2] uint8.

    Per-row L2 norm scaled as: gguf_norm[i] = ||W[i]|| / sqrt(k)
    Codes are nearest-Lloyd-Max-centroid indices in 0..15.
    """
    m, k = W.shape
    assert k % QK_SPIRAL_INT4 == 0, f"k={k} must be multiple of {QK_SPIRAL_INT4}"

    # Per-row scaling — matches build_spiral_artifact.py convention:
    #   normalized = W[i] / (||W[i]|| / sqrt(k))
    # which has unit variance under N(0,1) row assumption.
    row_norms_full = np.linalg.norm(W, axis=1)
    row_norms = (row_norms_full / math.sqrt(k)).astype(np.float16)
    row_norms_f32 = row_norms.astype(np.float32) + 1e-12
    W_normalized = W / row_norms_f32[:, None]

    # Nearest centroid
    diffs = W_normalized[..., None] - CENTROIDS_INT4[None, None, :]
    codes = np.argmin(diffs ** 2, axis=-1).astype(np.uint8)
    return row_norms, codes


def pack_int4_codes(codes: np.ndarray) -> np.ndarray:
    """codes: [m, k] uint8 in 0..15. Returns [m, k//2] uint8.
    Two codes per byte: low nibble = even, high nibble = odd.
    Matches block_spiral_int4 layout (ggml-common.h).
    """
    m, k = codes.shape
    assert k % 2 == 0
    out = np.zeros((m, k // 2), dtype=np.uint8)
    out[:, :] = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    return out


def decode_int4_row(packed_row: np.ndarray, k: int) -> np.ndarray:
    """Inverse of pack_int4_codes for one row. Returns codes [k] uint8."""
    codes = np.zeros(k, dtype=np.uint8)
    codes[0::2] = packed_row & 0xF
    codes[1::2] = (packed_row >> 4) & 0xF
    return codes


def write_block_format(row_norms: np.ndarray, packed_codes: np.ndarray) -> bytes:
    """Produce the GGUF SPIRAL_INT4 block layout:
       For each row, n_blocks blocks of 66 bytes:
         - bytes [0:2] = fp16 row_norm (same value in every block of the row)
         - bytes [2:66] = 64 bytes of packed 4-bit codes (128 codes)
    """
    m, codes_per_row_bytes = packed_codes.shape
    cpb = QK_SPIRAL_INT4 // 2  # 64 bytes of codes per block
    nb = codes_per_row_bytes // cpb

    out = np.zeros((m, nb, SPIRAL_INT4_BLOCK_BYTES), dtype=np.uint8)
    # Place row_norms in EVERY block of each row (first 2 bytes as fp16)
    norm_bytes = row_norms.tobytes()
    norm_arr = np.frombuffer(norm_bytes, dtype=np.uint8).reshape(m, 2)
    out[:, :, :2] = norm_arr[:, None, :]
    # Place codes
    codes_reshaped = packed_codes.reshape(m, nb, cpb)
    out[:, :, 2:66] = codes_reshaped
    return out.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, required=True,
                   help='output dim (rows of W)')
    p.add_argument('--k', type=int, required=True,
                   help='input dim (cols of W, must be multiple of 128)')
    p.add_argument('--n', type=int, required=True,
                   help='number of input columns (tokens); use 1 for decode')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Random unrotated input and weight ──────────────────────────────
    # No rotation here — we test kernel math in isolation. If kernel passes,
    # rotation handling is the only remaining suspect for the garbage output.
    x = rng.standard_normal((args.n, args.k), dtype=np.float32) * 0.5
    W = rng.standard_normal((args.m, args.k), dtype=np.float32) * (1.0 / math.sqrt(args.k))
    print(f"x: shape={x.shape}, norm={np.linalg.norm(x):.4f}")
    print(f"W: shape={W.shape}, norm={np.linalg.norm(W):.4f}")

    # ── Quantize W to SPIRAL_INT4 ──────────────────────────────────────
    row_norms, codes = quantize_to_int4(W)
    packed = pack_int4_codes(codes)
    print(f"Quantized: row_norms={row_norms.shape}, packed={packed.shape}")

    # ── Reconstruct W from quantized form (this is what kernel will see) ─
    W_decoded = np.zeros_like(W)
    row_norms_f32 = row_norms.astype(np.float32)
    for i in range(args.m):
        codes_i = decode_int4_row(packed[i], args.k)
        W_decoded[i] = CENTROIDS_INT4[codes_i] * row_norms_f32[i]

    # ── Compute reference: y = x @ W_decoded.T ─────────────────────────
    # This is what the kernel should produce, modulo int8 quantization noise
    # from our q8_1 activation pre-quantization step.
    y_reference = (x @ W_decoded.T).astype(np.float32)
    print(f"y_reference: shape={y_reference.shape}, norm={np.linalg.norm(y_reference):.4f}")

    # Diagnostic: error vs unquantized true answer
    y_unquant = (x @ W.T).astype(np.float32)
    quant_err = np.linalg.norm(y_reference - y_unquant) / np.linalg.norm(y_unquant)
    print(f"  (Quantization error vs FP32 truth: {quant_err:.4f}; expected ~0.10 for Lloyd-Max INT4)")

    # ── Write outputs ───────────────────────────────────────────────────
    blob = write_block_format(row_norms, packed)
    expected_blob_bytes = args.m * (args.k // QK_SPIRAL_INT4) * SPIRAL_INT4_BLOCK_BYTES
    assert len(blob) == expected_blob_bytes, \
        f"blob size mismatch: got {len(blob)}, expected {expected_blob_bytes}"
    print(f"Weight blob: {len(blob)} bytes ({len(blob) / 1024 / 1024:.2f} MiB)")

    (out_dir / "x_rotated.bin").write_bytes(x.tobytes())  # Named for compat with C++ tester
    (out_dir / "weight_packed.bin").write_bytes(blob)
    (out_dir / "y_reference.bin").write_bytes(y_reference.tobytes())

    nb = args.k // QK_SPIRAL_INT4
    meta = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "seed": args.seed,
        "block_size": QK_SPIRAL_INT4,
        "block_bytes": SPIRAL_INT4_BLOCK_BYTES,
        "n_blocks_per_row": nb,
        "weight_blob_bytes": len(blob),
        "x_rotated_bytes": x.nbytes,
        "y_reference_bytes": y_reference.nbytes,
        "x_rotated_layout": "F32, contiguous, shape [n, k] row-major",
        "y_reference_layout": "F32, contiguous, shape [n, m] row-major",
        "weight_blob_layout": (
            f"For each row r in 0..{args.m}: {nb} blocks of {SPIRAL_INT4_BLOCK_BYTES} bytes."
            " Each block: 2 bytes fp16 row_norm (repeated) + 64 bytes packed 4-bit codes."
        ),
        "centroids": CENTROIDS_INT4.tolist(),
        "rotation_applied": False,
        "quantization_type": "SPIRAL_INT4",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    debug = []
    debug.append(f"shape: m={args.m} k={args.k} n={args.n}")
    debug.append(f"y_reference[0, :8]: {y_reference[0, :8].tolist()}")
    debug.append(f"y_reference[0] norm: {np.linalg.norm(y_reference[0]):.6f}")
    debug.append(f"x[0, :8]: {x[0, :8].tolist()}")
    debug.append(f"row_norms[0]: {float(row_norms[0]):.6f}")
    debug.append(f"first 16 packed bytes of row 0: {packed[0, :16].tolist()}")
    (out_dir / "debug_info.txt").write_text("\n".join(debug))

    print()
    print(f"Wrote fixtures to {out_dir}/")
    print(f"  meta.json")
    print(f"  x_rotated.bin       ({x.nbytes} bytes)")
    print(f"  weight_packed.bin   ({len(blob)} bytes)")
    print(f"  y_reference.bin     ({y_reference.nbytes} bytes)")
    print(f"  debug_info.txt")


if __name__ == "__main__":
    sys.exit(main() or 0)
