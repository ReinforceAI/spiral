#!/usr/bin/env python
"""
compare_kernel_runs.py
======================

Compare three things:
  1. y_isolated_real.bin — output from our isolated kernel test, fed
     production's actual rotated input bytes and weight bytes
  2. y_production.bin    — production's wqkv_output dump
                            (from /tmp/spiral_dump/000006_wqkv_output.bin)
  3. y_python.bin (computed inline) — Python ground truth using the
     same weight and same input via numpy

If isolated == production but != python: kernel matches itself but
deviates from ground truth in a consistent way (i.e., kernel has a bug
that's deterministic on the real bytes).

If isolated == python but != production: the isolated kernel run gets
the right answer, meaning production has SOMETHING extra going wrong
(buffer aliasing, scheduler, additional ops in the graph).

If isolated != production != python: all three differ — different bug.

Usage:
  python compare_kernel_runs.py \
      /tmp/y_isolated_real.bin \
      /tmp/spiral_dump/000006_wqkv_output.bin \
      /tmp/spiral_dump/000005_wqkv_input_rotated.bin \
      runs/qwen36-35b-a3b.gguf \
      blk.0.attn_qkv.weight
"""
from __future__ import annotations
import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np


SPIRAL_3BIT_TYPE_ID = 48
QK = 128
BLOCK_BYTES = 50

CENTROIDS = np.array([
    -2.1556325, -1.3483801, -0.7599415, -0.2466970,
     0.2466970,  0.7599415,  1.3483801,  2.1556325,
], dtype=np.float32)


def read_dump(path):
    """Read spiral_dump format file. Returns (header_dict, np.float32 array)."""
    raw = Path(path).read_bytes()
    if len(raw) < 128:
        raise ValueError(f"File too small: {path}")
    ne = struct.unpack_from("<4q", raw, 48)
    hdr = {
        'tag': raw[8:40].rstrip(b'\x00').decode('ascii', errors='replace'),
        'ne': ne,
        'dtype': struct.unpack_from("<i", raw, 112)[0],
    }
    arr = np.frombuffer(raw[128:], dtype=np.float32)
    return hdr, arr


def read_raw_f32(path, n_elements):
    """Read raw F32 file (no header)."""
    raw = Path(path).read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != n_elements:
        print(f"WARNING: {path} has {arr.size} elements, expected {n_elements}")
    return arr


def unpack_3bit_codes(packed, in_features):
    out = packed.shape[0]
    codes = np.zeros((out, in_features), dtype=np.uint8)
    for g in range(in_features // 8):
        bi = g*3
        b0, b1, b2 = packed[:, bi], packed[:, bi+1], packed[:, bi+2]
        codes[:, g*8+0] = b0 & 0x7
        codes[:, g*8+1] = (b0 >> 3) & 0x7
        codes[:, g*8+2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        codes[:, g*8+3] = (b1 >> 1) & 0x7
        codes[:, g*8+4] = (b1 >> 4) & 0x7
        codes[:, g*8+5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        codes[:, g*8+6] = (b2 >> 2) & 0x7
        codes[:, g*8+7] = (b2 >> 5) & 0x7
    return codes


def decode_weight_from_gguf(gguf_path, tensor_name):
    """Decode a SPIRAL_3BIT weight from GGUF. Returns W_decoded [m, k] FP32."""
    from gguf import GGMLQuantizationType
    from gguf.quants import GGML_QUANT_SIZES
    if SPIRAL_3BIT_TYPE_ID not in GGMLQuantizationType._value2member_map_:
        m = int.__new__(GGMLQuantizationType, SPIRAL_3BIT_TYPE_ID)
        m._name_ = 'SPIRAL_3BIT'
        m._value_ = SPIRAL_3BIT_TYPE_ID
        GGMLQuantizationType._value2member_map_[SPIRAL_3BIT_TYPE_ID] = m
        GGMLQuantizationType._member_map_['SPIRAL_3BIT'] = m
        GGML_QUANT_SIZES[GGMLQuantizationType(SPIRAL_3BIT_TYPE_ID)] = (QK, BLOCK_BYTES)
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    t = next(x for x in r.tensors if x.name == tensor_name)

    # GGUF tensor shape: (in_features, out_features) in ggml convention
    # but exposed differently — check
    shape = tuple(int(s) for s in t.shape)
    print(f"  GGUF tensor '{tensor_name}' shape={shape} type={t.tensor_type.name}")

    # For SPIRAL_3BIT, GGUF stores rows of weight: out_features = shape[1], in_features = shape[0]
    in_features = shape[0]
    out_features = shape[1]

    raw = np.asarray(t.data).reshape(-1).view(np.uint8)
    nb = in_features // QK
    arr = raw.reshape(out_features, nb, BLOCK_BYTES)
    norm_bytes = arr[:, 0, :2].copy()
    row_norms = np.frombuffer(norm_bytes.tobytes(), dtype=np.float16).astype(np.float32)

    code_blocks = arr[:, :, 2:BLOCK_BYTES].reshape(out_features, nb * (QK * 3 // 8)).copy()
    codes = unpack_3bit_codes(code_blocks, in_features)

    W = (CENTROIDS[codes] * row_norms[:, None]).astype(np.float32)
    return W, in_features, out_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('y_isolated', help='output from spiral_kernel_test_real (raw F32)')
    p.add_argument('y_production_dump', help='spiral_dump wqkv_output bin (with header)')
    p.add_argument('x_production_dump', help='spiral_dump wqkv_input_rotated bin (with header)')
    p.add_argument('gguf_path', help='runs/qwen36-35b-a3b.gguf')
    p.add_argument('weight_name', default='blk.0.attn_qkv.weight', nargs='?')
    args = p.parse_args()

    print("=== Loading production output ===")
    prod_hdr, y_prod_flat = read_dump(args.y_production_dump)
    ne = prod_hdr['ne']
    m_prod = ne[0]  # output dim
    n_prod = ne[1]  # tokens
    print(f"  ne={ne}, n={n_prod}, m={m_prod}")
    y_prod = y_prod_flat.reshape(n_prod, m_prod)
    print(f"  norm={np.linalg.norm(y_prod):.4f}  range=[{y_prod.min():.4f}, {y_prod.max():.4f}]")

    print()
    print("=== Loading production rotated input ===")
    inp_hdr, x_flat = read_dump(args.x_production_dump)
    inp_ne = inp_hdr['ne']
    k = inp_ne[0]
    n_in = inp_ne[1]
    print(f"  ne={inp_ne}, k={k}, n={n_in}")
    if n_in != n_prod:
        print(f"  WARNING: input n={n_in} != output n={n_prod}")
    x = x_flat.reshape(n_in, k)
    print(f"  norm={np.linalg.norm(x):.4f}")

    print()
    print("=== Decoding weight from GGUF ===")
    W, in_features, out_features = decode_weight_from_gguf(args.gguf_path, args.weight_name)
    print(f"  W shape=({out_features}, {in_features}) norm={np.linalg.norm(W):.4f}")

    if k != in_features:
        print(f"WARNING: input k={k} != weight in_features={in_features}")
    if m_prod != out_features:
        print(f"WARNING: output m={m_prod} != weight out_features={out_features}")

    print()
    print("=== Computing Python ground truth ===")
    y_python = (x @ W.T).astype(np.float32)
    print(f"  norm={np.linalg.norm(y_python):.4f}  range=[{y_python.min():.4f}, {y_python.max():.4f}]")

    print()
    print("=== Loading isolated kernel output ===")
    y_iso_flat = read_raw_f32(args.y_isolated, n_prod * m_prod)
    y_iso = y_iso_flat.reshape(n_prod, m_prod)
    print(f"  norm={np.linalg.norm(y_iso):.4f}  range=[{y_iso.min():.4f}, {y_iso.max():.4f}]")

    print()
    print("=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)

    def compare(name, a, b):
        diff = a - b
        rel = np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-12)
        return rel

    iso_vs_python = compare("iso vs python", y_iso, y_python)
    iso_vs_prod   = compare("iso vs prod  ", y_iso, y_prod)
    prod_vs_python = compare("prod vs python", y_prod, y_python)

    print(f"  isolated vs python : rel_err = {iso_vs_python:.4f}")
    print(f"  isolated vs production : rel_err = {iso_vs_prod:.4f}")
    print(f"  production vs python : rel_err = {prod_vs_python:.4f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if iso_vs_python < 0.01:
        print("✓ Isolated kernel agrees with Python ground truth.")
        if prod_vs_python > 0.5:
            print("✗ Production output diverges from Python ground truth.")
            print()
            print("CONCLUSION: The kernel produces correct output when called in")
            print("isolation with production's actual input/weight bytes.")
            print("Production's broken output is therefore caused by something")
            print("OTHER than the kernel — likely buffer aliasing, scheduler,")
            print("or a wrong tensor being passed at runtime that we never see")
            print("in our isolated test.")
    elif iso_vs_prod < 0.01:
        print("✓ Isolated kernel matches production output exactly.")
        print()
        print("CONCLUSION: The kernel reproduces production's broken output")
        print("on the same byte inputs. The bug is in the kernel's deterministic")
        print("behavior on these specific inputs — possibly a numerical edge case")
        print("not triggered by random data.")
    else:
        print("? All three outputs differ from each other.")
        print("Need closer inspection of which positions differ.")


if __name__ == "__main__":
    sys.exit(main() or 0)
