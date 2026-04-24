#!/usr/bin/env python3
"""
verify_gguf_shapes.py — Compare tensor shapes between Spiral and reference GGUFs

This script reads the tensor metadata from two GGUF files and compares
the shapes of expert (3D) tensors. The goal is to find shape mismatches
that would cause ggml to compute wrong strides for mul_mv_id dispatch.

Usage:
  python verify_gguf_shapes.py --spiral <spiral.gguf> --reference <q4_k_m.gguf>

If only --spiral is provided, it prints all tensor shapes for inspection.
"""
import argparse
import struct
import sys
from pathlib import Path


# Minimal GGUF reader — just enough to extract tensor metadata
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 28: "BF16",
    30: "TQ1_0", 31: "TQ2_0", 32: "IQ4_XS",
    34: "TQ3_1S", 35: "TQ4_1S",
    48: "SPIRAL_3BIT",
}

# Block sizes and type sizes for stride computation
GGML_TYPE_TRAITS = {
    0:  (1, 4),      # F32: blck_size=1, type_size=4
    1:  (1, 2),      # F16
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    48: (128, 50),    # SPIRAL_3BIT
}


def read_string(f):
    """Read a GGUF string (uint64 length + bytes)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_value(f, vtype):
    """Read a GGUF metadata value by type."""
    if vtype == 0:  # UINT8
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1:  # INT8
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2:  # UINT16
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3:  # INT16
        return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4:  # UINT32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  # INT32
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6:  # FLOAT32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7:  # BOOL
        return struct.unpack('<?', f.read(1))[0]
    elif vtype == 8:  # STRING
        return read_string(f)
    elif vtype == 9:  # ARRAY
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10:  # UINT64
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11:  # INT64
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12:  # FLOAT64
        return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f"Unknown value type: {vtype}")


def read_gguf_tensors(path):
    """Read tensor metadata from a GGUF file.
    
    Returns:
        metadata: dict of key-value pairs
        tensors: list of (name, ndims, shape, type_id, offset) tuples
    """
    tensors = []
    metadata = {}
    
    with open(path, 'rb') as f:
        # Header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: magic={magic:#x}")
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")
        
        # Read metadata
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value
        
        # Read tensor info
        for _ in range(n_tensors):
            name = read_string(f)
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
            type_id = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, ndims, shape, type_id, offset))
    
    return metadata, tensors


def compute_strides(shape, type_id):
    """Compute ggml-style byte strides for a tensor.
    
    ggml computes:
      nb0 = type_size (for quantized) or sizeof(element)
      nb1 = (ne0 / blck_size) * type_size
      nb2 = ne1 * nb1
      nb3 = ne2 * nb2
    """
    if type_id not in GGML_TYPE_TRAITS:
        return None
    
    blck_size, type_size = GGML_TYPE_TRAITS[type_id]
    ne = shape  # ne[0], ne[1], ...
    
    nb0 = type_size
    nb1 = (ne[0] // blck_size) * type_size if len(ne) > 0 else 0
    nb2 = ne[1] * nb1 if len(ne) > 1 else 0
    nb3 = ne[2] * nb2 if len(ne) > 2 else 0
    
    return nb0, nb1, nb2, nb3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spiral", required=True, help="Path to Spiral GGUF")
    parser.add_argument("--reference", help="Path to reference GGUF (e.g., Q4_K_M)")
    parser.add_argument("--filter", default="exps", help="Filter tensor names (default: 'exps')")
    parser.add_argument("--all", action="store_true", help="Show all tensors")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"SPIRAL GGUF: {args.spiral}")
    print(f"{'='*80}")
    
    s_meta, s_tensors = read_gguf_tensors(args.spiral)
    
    # Print relevant metadata
    for key in sorted(s_meta.keys()):
        if any(k in key for k in ['expert', 'head_count', 'embedding_length', 
                                    'feed_forward', 'key_length', 'value_length',
                                    'spiral', 'block_count']):
            print(f"  {key} = {s_meta[key]}")
    
    # Print tensor info
    print(f"\nTensors matching '{args.filter}':" if not args.all else "\nAll tensors:")
    for name, ndims, shape, type_id, offset in s_tensors:
        if not args.all and args.filter not in name:
            continue
        type_name = GGML_TYPE_NAMES.get(type_id, f"TYPE_{type_id}")
        strides = compute_strides(shape, type_id)
        stride_str = ""
        if strides:
            nb0, nb1, nb2, nb3 = strides
            stride_str = f"  strides: nb0={nb0}, nb1={nb1}, nb2={nb2}"
            if ndims >= 3:
                stride_str += f", nb3={nb3}"
        
        total_bytes = 1
        if type_id in GGML_TYPE_TRAITS:
            blck_size, type_size = GGML_TYPE_TRAITS[type_id]
            # For quantized: total = (ne0/blck_size * type_size) * ne1 * ne2 * ...
            total_bytes = (shape[0] // blck_size) * type_size
            for d in range(1, ndims):
                total_bytes *= shape[d]
        
        print(f"  {name}")
        print(f"    ndims={ndims}, shape={shape}, type={type_name}")
        print(f"    offset={offset}, total_bytes={total_bytes:,}")
        if stride_str:
            print(f"   {stride_str}")
    
    if args.reference:
        print(f"\n{'='*80}")
        print(f"REFERENCE GGUF: {args.reference}")
        print(f"{'='*80}")
        
        r_meta, r_tensors = read_gguf_tensors(args.reference)
        
        # Print relevant metadata
        for key in sorted(r_meta.keys()):
            if any(k in key for k in ['expert', 'head_count', 'embedding_length',
                                        'feed_forward', 'key_length', 'value_length',
                                        'block_count']):
                print(f"  {key} = {r_meta[key]}")
        
        print(f"\nTensors matching '{args.filter}':" if not args.all else "\nAll tensors:")
        for name, ndims, shape, type_id, offset in r_tensors:
            if not args.all and args.filter not in name:
                continue
            type_name = GGML_TYPE_NAMES.get(type_id, f"TYPE_{type_id}")
            strides = compute_strides(shape, type_id)
            stride_str = ""
            if strides:
                nb0, nb1, nb2, nb3 = strides
                stride_str = f"  strides: nb0={nb0}, nb1={nb1}, nb2={nb2}"
                if ndims >= 3:
                    stride_str += f", nb3={nb3}"
            print(f"  {name}")
            print(f"    ndims={ndims}, shape={shape}, type={type_name}")
            if stride_str:
                print(f"   {stride_str}")
        
        # Compare expert tensors
        print(f"\n{'='*80}")
        print("COMPARISON: Expert tensor shapes")
        print(f"{'='*80}")
        
        s_dict = {name: (ndims, shape, type_id) for name, ndims, shape, type_id, _ in s_tensors}
        r_dict = {name: (ndims, shape, type_id) for name, ndims, shape, type_id, _ in r_tensors}
        
        for name in sorted(s_dict.keys()):
            if args.filter not in name:
                continue
            s_ndims, s_shape, s_type = s_dict[name]
            if name in r_dict:
                r_ndims, r_shape, r_type = r_dict[name]
                s_type_name = GGML_TYPE_NAMES.get(s_type, f"TYPE_{s_type}")
                r_type_name = GGML_TYPE_NAMES.get(r_type, f"TYPE_{r_type}")
                match = "✓" if s_shape == r_shape else "✗ MISMATCH"
                print(f"  {name}")
                print(f"    Spiral:    shape={s_shape} type={s_type_name}")
                print(f"    Reference: shape={r_shape} type={r_type_name}")
                print(f"    Shape match: {match}")
                
                # Compute and compare strides
                s_strides = compute_strides(s_shape, s_type)
                r_strides = compute_strides(r_shape, r_type)
                if s_strides and r_strides:
                    print(f"    Spiral strides:    nb1={s_strides[1]}, nb2={s_strides[2]}")
                    print(f"    Reference strides: nb1={r_strides[1]}, nb2={r_strides[2]}")
            else:
                print(f"  {name}: NOT in reference GGUF")


if __name__ == "__main__":
    main()