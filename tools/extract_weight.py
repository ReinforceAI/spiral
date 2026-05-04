#!/usr/bin/env python
"""
extract_weight.py
=================

Extract raw bytes of one tensor from a GGUF file.

Usage:
  python extract_weight.py <gguf_path> <tensor_name> <output_path>

Example:
  python tools/extract_weight.py runs/qwen36-35b-a3b.gguf blk.0.attn_qkv.weight /tmp/weight_raw.bin
"""
import sys
import numpy as np
from pathlib import Path

# Register SPIRAL_3BIT type
from gguf import GGMLQuantizationType
from gguf.quants import GGML_QUANT_SIZES
SPIRAL_3BIT_TYPE_ID = 48
if SPIRAL_3BIT_TYPE_ID not in GGMLQuantizationType._value2member_map_:
    m = int.__new__(GGMLQuantizationType, SPIRAL_3BIT_TYPE_ID)
    m._name_ = 'SPIRAL_3BIT'
    m._value_ = SPIRAL_3BIT_TYPE_ID
    GGMLQuantizationType._value2member_map_[SPIRAL_3BIT_TYPE_ID] = m
    GGMLQuantizationType._member_map_['SPIRAL_3BIT'] = m
    GGML_QUANT_SIZES[GGMLQuantizationType(SPIRAL_3BIT_TYPE_ID)] = (128, 50)

from gguf import GGUFReader


def main():
    if len(sys.argv) != 4:
        print("Usage: extract_weight.py <gguf_path> <tensor_name> <output_path>")
        return 1

    gguf_path, tensor_name, output_path = sys.argv[1:4]

    r = GGUFReader(gguf_path)
    t = next((x for x in r.tensors if x.name == tensor_name), None)
    if t is None:
        print(f"ERROR: tensor '{tensor_name}' not found")
        return 1

    shape = tuple(int(s) for s in t.shape)
    print(f"Found '{tensor_name}': shape={shape} type={t.tensor_type.name}")

    # Get raw bytes
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)
    Path(output_path).write_bytes(raw.tobytes())

    print(f"Wrote {raw.size} bytes to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
