// Spiral tensor dump — implementation.
//
// See spiral-dump.h for usage.

#include "spiral-dump.h"
#include "spiral-debug.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace spiral_dump {

// One registered tensor awaiting flush.
struct entry {
    ggml_tensor * tensor;
    char          tag[32];
    int32_t       layer;
};

namespace {

std::mutex                 g_mutex;
std::vector<entry>         g_pending;
std::atomic<uint64_t>      g_seq(0);
std::string                g_dump_dir;

// Cached enabled status (-1 = not yet checked, 0 = disabled, 1 = enabled).
std::atomic<int>           g_enabled(-1);

// Returns the dump directory, creating it on first call.
const std::string & dump_dir() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char * env = getenv("SPIRAL_DUMP_DIR");
        g_dump_dir = env ? env : "/tmp/spiral_dump";
        // Create directory; ignore errors (might already exist)
        mkdir(g_dump_dir.c_str(), 0755);
    });
    return g_dump_dir;
}

}  // namespace

bool enabled() {
    int e = g_enabled.load(std::memory_order_relaxed);
    if (e == -1) {
        const char * env = getenv("SPIRAL_DUMP_ENABLE");
        e = (env != nullptr && env[0] != '0') ? 1 : 0;
        g_enabled.store(e, std::memory_order_relaxed);
        if (e == 1) {
            fprintf(stderr, "[spiral_dump] enabled, output dir = %s\n", dump_dir().c_str());
            fflush(stderr);
        }
    }
    return e == 1;
}

void register_tensor(ggml_tensor * tensor, const char * tag, int layer) {
    if (!enabled()) return;
    if (!tensor || !tag) return;

    // Mark for CPU readback after compute. ggml_set_output ensures the backend
    // allocates this tensor's data in CPU-accessible memory and that
    // ggml_backend_sched leaves it valid after compute.
    ggml_set_output(tensor);

    entry e;
    e.tensor = tensor;
    e.layer  = (int32_t)layer;
    // Truncate tag to 31 chars + null
    std::snprintf(e.tag, sizeof(e.tag), "%s", tag);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_pending.push_back(e);
}

void flush() {
    if (!enabled()) return;

    std::vector<entry> to_write;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_pending.empty()) return;
        to_write.swap(g_pending);
    }

    const std::string & dir = dump_dir();

    for (const entry & e : to_write) {
        ggml_tensor * t = e.tensor;
        if (!t) continue;

        // Defensive: skip tensors whose buffer/data isn't valid. This handles
        // the graph_reserve path: graph_reserve constructs a graph context
        // (which registers tensors) but discards the graph without computing
        // it. Those tensor pointers are stale by the time we try to flush.
        // A real, computed tensor will have a non-null buffer and data.
        if (t->buffer == nullptr || t->data == nullptr) {
            continue;
        }

        // Read tensor data from CPU-accessible buffer.
        // ggml_set_output should have made this directly addressable.
        // We use ggml_backend_tensor_get() to be safe across backends.
        const size_t nbytes = ggml_nbytes(t);
        std::vector<char> data(nbytes);
        ggml_backend_tensor_get(t, data.data(), 0, nbytes);

        // SPIRAL_DIAG: print buffer info to detect aliasing.
        // If two tensors share the same (buffer, offset) they alias.
        ggml_backend_buffer_t buf = t->buffer;
        size_t buf_size = buf ? ggml_backend_buffer_get_size(buf) : 0;
        // Compute offset of t->data within the buffer
        void * buf_base = buf ? ggml_backend_buffer_get_base(buf) : nullptr;
        ptrdiff_t offset = -1;
        if (buf_base && t->data) {
            offset = (ptrdiff_t)((char *)t->data - (char *)buf_base);
        }
        if (spiral_debug_on()) {
            fprintf(stderr, "[spiral_diag] tag=%-30s buf=%p base=%p offset=%lld nbytes=%zu buf_size=%zu\n",
                    e.tag, (void *)buf, buf_base, (long long)offset, nbytes, buf_size);
            fflush(stderr);
        }

        // Build header
        file_header hdr;
        std::memset(&hdr, 0, sizeof(hdr));
        std::memcpy(hdr.magic, MAGIC, 8);
        std::snprintf(hdr.tag, sizeof(hdr.tag), "%s", e.tag);
        hdr.layer     = e.layer;
        hdr.n_dims    = (uint32_t)GGML_MAX_DIMS;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            hdr.ne[i] = t->ne[i];
            hdr.nb[i] = t->nb[i];
        }
        hdr.ggml_type = (uint32_t)t->type;
        hdr.data_size = (uint32_t)nbytes;

        // File path: <dir>/<seq>_<tag>.bin
        const uint64_t seq = g_seq.fetch_add(1, std::memory_order_relaxed);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/%06llu_%s.bin",
                      dir.c_str(), (unsigned long long)seq, e.tag);

        FILE * f = std::fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "[spiral_dump] could not open %s for writing\n", path);
            continue;
        }
        std::fwrite(&hdr, sizeof(hdr), 1, f);
        std::fwrite(data.data(), 1, nbytes, f);
        std::fclose(f);

        fprintf(stderr, "[spiral_dump] wrote %s (%zu bytes, ne=[%lld,%lld,%lld,%lld] type=%d)\n",
                path, nbytes,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3],
                (int)t->type);
        fflush(stderr);
    }
}

}  // namespace spiral_dump
