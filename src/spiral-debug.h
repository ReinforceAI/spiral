// spiral-debug.h
//
// Single-source runtime gate for SPIRAL debug logging.
// All [spiral_diag], SPIRAL_MV, SPIRAL_MM, SPIRAL_ROT, etc. fprintfs are
// gated through spiral_debug_on(). Silent by default; enable with any of:
//
//   SPIRAL_DEBUG=1            — turn on all spiral debug output
//   SPIRAL_DEBUG_DISPATCH=1   — alias (kept for category-friendly naming)
//   SPIRAL_DEBUG_GRAPH=1      — alias
//   SPIRAL_DEBUG_DUMP=1       — alias
//
// Cost when disabled: one branch on a cached static bool. Effectively free.
//
// Usage:
//   #include "spiral-debug.h"
//   if (spiral_debug_on()) fprintf(stderr, "...\n");
//
// For fprintfs that already live inside an outer `if (...)` block, just
// AND the guard into the existing condition:
//   if (spiral_debug_on() && cond) { ... }

#pragma once

#include <cstdlib>

inline bool spiral_debug_on() {
    static const bool on =
        (std::getenv("SPIRAL_DEBUG")          != nullptr) ||
        (std::getenv("SPIRAL_DEBUG_DISPATCH") != nullptr) ||
        (std::getenv("SPIRAL_DEBUG_GRAPH")    != nullptr) ||
        (std::getenv("SPIRAL_DEBUG_DUMP")     != nullptr);
    return on;
}