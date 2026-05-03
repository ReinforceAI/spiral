"""
spiral_dump_reader.py
=====================

Reads binary tensor dumps written by spiral-dump.cpp.

File format (128-byte header + raw data):

    Bytes  0-7:    magic "SPDUMP01"
    Bytes  8-39:   tag (null-padded 32-char string)
    Bytes 40-43:   layer (int32, -1 if not layer-specific)
    Bytes 44-47:   n_dims (uint32, typically 4)
    Bytes 48-79:   ne[4] (4× int64 shape)
    Bytes 80-111:  nb[4] (4× uint64 strides)
    Bytes 112-115: ggml_type (uint32)
    Bytes 116-119: data_size (uint32)
    Bytes 120-127: reserved
    Bytes 128+:    raw tensor data

Usage:

    from spiral_dump_reader import load_dump, scan_dumps

    # Load one file
    arr, meta = load_dump('/tmp/spiral_dump/000001_blk0_attn_qkv_output.bin')
    print(arr.shape, arr.dtype, meta)

    # Scan a directory
    for path, meta in scan_dumps('/tmp/spiral_dump'):
        print(meta['seq'], meta['tag'], meta['ne'])

    # Load by tag (latest occurrence)
    arr, meta = load_by_tag('/tmp/spiral_dump', 'blk0_attn_qkv_output')
"""
from __future__ import annotations
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np


# Mirrors enum ggml_type (subset we actually encounter)
GGML_TYPE = {
    0:  ('F32',          np.float32, 4),
    1:  ('F16',          np.float16, 2),
    24: ('I32',          np.int32,   4),
    26: ('I64',          np.int64,   8),
    # SPIRAL_3BIT type id; we don't decode raw blocks here, expose as bytes
    48: ('SPIRAL_3BIT',  np.uint8,   1),
}

HEADER_SIZE = 128
MAGIC = b'SPDUMP01'


def _parse_header(buf: bytes) -> dict:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"buffer too small for header: {len(buf)} bytes")
    if buf[:8] != MAGIC:
        raise ValueError(f"bad magic: {buf[:8]!r} != {MAGIC!r}")
    tag = buf[8:40].rstrip(b'\x00').decode('utf-8', errors='replace')
    layer    = struct.unpack_from('<i', buf, 40)[0]
    n_dims   = struct.unpack_from('<I', buf, 44)[0]
    ne       = struct.unpack_from('<4q', buf, 48)
    nb       = struct.unpack_from('<4Q', buf, 80)
    gtype    = struct.unpack_from('<I', buf, 112)[0]
    data_size = struct.unpack_from('<I', buf, 116)[0]
    return {
        'tag':       tag,
        'layer':     layer,
        'n_dims':    n_dims,
        'ne':        list(ne),
        'nb':        list(nb),
        'ggml_type': gtype,
        'data_size': data_size,
    }


def load_dump(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a single dump file. Returns (array, metadata).

    The array is reshaped to ne but transposed to numpy convention:
    ggml stores tensors with ne[0] varying fastest (innermost), so the
    numpy array shape is (ne[3], ne[2], ne[1], ne[0]) — or trimmed if
    higher dims are 1.
    """
    path = Path(path)
    raw = path.read_bytes()
    meta = _parse_header(raw)
    data_buf = raw[HEADER_SIZE:HEADER_SIZE + meta['data_size']]
    if len(data_buf) != meta['data_size']:
        raise ValueError(f"truncated data: have {len(data_buf)}, header says {meta['data_size']}")

    gt = meta['ggml_type']
    if gt not in GGML_TYPE:
        raise ValueError(f"unsupported ggml_type {gt}")
    name, dtype, _ = GGML_TYPE[gt]

    arr = np.frombuffer(data_buf, dtype=dtype)
    ne = meta['ne']

    # Reshape to numpy convention: outer dims first.
    # ne[0] is innermost in ggml; numpy wants outermost first.
    # For a 2D matrix [n_rows, n_cols], ggml has ne[0]=n_cols, ne[1]=n_rows,
    # so numpy shape is (n_rows, n_cols) = (ne[1], ne[0]).
    shape = []
    for d in reversed(ne):
        if d > 1 or len(shape) > 0:
            shape.append(int(d))
    if not shape:
        shape = [1]
    # Trim leading 1s (squeeze higher dims)
    while len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]

    expected_n = 1
    for d in shape:
        expected_n *= d
    if arr.size != expected_n and gt != 48:  # SPIRAL_3BIT is opaque blocks
        # Try a stricter shape: full 4D with all dims
        shape = list(reversed([int(d) for d in ne]))

    try:
        arr = arr.reshape(shape)
    except ValueError:
        # If reshape fails, return flat
        pass

    meta['name']  = path.name
    meta['shape'] = list(arr.shape)
    return arr, meta


def scan_dumps(directory: str | Path) -> list[tuple[Path, dict]]:
    """Scan a directory for dump files. Returns list of (path, metadata)
    sorted by sequence number embedded in the filename."""
    directory = Path(directory)
    results = []
    for p in sorted(directory.glob('*.bin')):
        try:
            with open(p, 'rb') as f:
                hdr_buf = f.read(HEADER_SIZE)
            meta = _parse_header(hdr_buf)
            # Parse seq number from filename: "<seq>_<tag>.bin"
            stem = p.stem  # without .bin
            seq_str, _, _ = stem.partition('_')
            try:
                meta['seq'] = int(seq_str)
            except ValueError:
                meta['seq'] = -1
            meta['name'] = p.name
            results.append((p, meta))
        except Exception as ex:
            print(f"warn: could not parse {p}: {ex}")
    return results


def load_by_tag(directory: str | Path, tag: str,
                layer: Optional[int] = None,
                index: int = -1) -> tuple[np.ndarray, dict]:
    """Find a dump by tag (and optionally layer). Returns the `index`-th match
    in seq order (default: last)."""
    matches = []
    for p, m in scan_dumps(directory):
        if m['tag'] == tag and (layer is None or m['layer'] == layer):
            matches.append((p, m))
    if not matches:
        raise KeyError(f"no dumps with tag={tag!r} (layer={layer}) in {directory}")
    p, _ = matches[index]
    return load_dump(p)


def main():
    """CLI: scan a directory and print summary."""
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '/tmp/spiral_dump'
    print(f"Scanning {directory}...")
    found = scan_dumps(directory)
    if not found:
        print("  (no dumps found)")
        return
    print(f"Found {len(found)} dumps:")
    for p, m in found:
        type_name = GGML_TYPE.get(m['ggml_type'], (f'?{m["ggml_type"]}',))[0]
        ne_str = '×'.join(str(d) for d in m['ne'] if d > 1) or '1'
        print(f"  seq={m.get('seq', '?'):>4}  layer={m['layer']:>3}  "
              f"tag={m['tag']:<32s}  type={type_name:<12s}  shape={ne_str}")


if __name__ == '__main__':
    main()