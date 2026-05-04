// Spiral tensor dump — debugging facility for inspecting tensor values at
// arbitrary points in the graph. All operations are no-ops unless the
// environment variable SPIRAL_DUMP_ENABLE is set.
//
// Usage:
//
//   At graph build time, in a model file or graph-context method:
//     spiral_dump::register_tensor(some_tensor, "blk0_attn_qkv_output", il);
//
//   At the START of every graph build (before any tensors are created),
//   call:
//     spiral_dump::flush();
//
//   This flushes the PREVIOUS graph's captured tensors to disk. After the
//   first graph compute completes, on the second graph build, the previous
//   graph's tensor data is already in CPU-readable memory (because
//   register_tensor calls ggml_set_output) and gets written to
//   /tmp/spiral_dump/<seq>_<tag>.bin.
//
//   Pair with tools/spiral_dump_reader.py to read the binary files into
//   numpy arrays for ground-truth verification.

#pragma once

#include <cstdint>

struct ggml_tensor;

namespace spiral_dump {

// Magic for binary file header
static constexpr char MAGIC[8] = {'S','P','D','U','M','P','0','1'};

// File header layout (128 bytes total). Mirrored in spiral_dump_reader.py.
struct file_header {
    char     magic[8];        // "SPDUMP01"
    char     tag[32];         // null-padded tag string
    int32_t  layer;           // -1 if not layer-specific
    uint32_t n_dims;          // number of valid entries in ne/nb (typically 4)
    int64_t  ne[4];           // tensor shape
    uint64_t nb[4];           // tensor strides
    uint32_t ggml_type;       // ggml_type as integer
    uint32_t data_size;       // size of raw data following the header
    uint32_t reserved[2];     // pad to 128 bytes
};
static_assert(sizeof(file_header) == 128, "spiral_dump::file_header must be 128 bytes");

// Returns true if SPIRAL_DUMP_ENABLE env var is set. Cached on first call.
bool enabled();

// Tag this tensor for capture. Calls ggml_set_output() on it so the backend
// allocates it in CPU-readable memory. Adds to internal registry. The tensor's
// data will be written to disk on the next call to flush().
//
// No-op if !enabled().
//
// tag must be <= 31 chars; longer tags are truncated.
// layer can be -1 if the tag is not layer-specific.
void register_tensor(ggml_tensor * tensor, const char * tag, int layer);

// Write all registered tensors from the previous graph compute to disk,
// then clear the registry. Called at the start of every graph build (before
// new tensors are registered).
//
// No-op if !enabled() or if registry is empty.
void flush();

}  // namespace spiral_dump