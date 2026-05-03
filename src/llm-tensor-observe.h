// llm-tensor-observe.h
//
// Generic tensor observation hook. Two function pointers, default null.
// External tools can register callbacks to observe named tensors during
// graph construction and after evaluation, for debugging or profiling.
//
// When pointers are null (the default), the hooks are zero-overhead.
// When set, the registered functions are called inline at the hook points.
//
// This is intentionally generic infrastructure — not coupled to any
// specific debugger or analysis tool. Any external tool that wants to
// observe tensor values during execution can use this.
//
// Contract:
//   - Observers MUST NOT modify the tensor or the graph.
//   - Observers MUST NOT call ggml_set_output or any allocator-affecting
//     API. Doing so would alter execution behavior, defeating the
//     purpose of an observation hook.
//   - Observers SHOULD complete quickly. They are called inline on the
//     hot path during graph construction and after evaluation.

#pragma once

#include <cstddef>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

// Called from llm_graph_context::cb() for every named tensor during
// graph construction. The tool can record the tensor pointer for later
// observation; it must NOT modify the tensor or the graph.
//
// Parameters:
//   tensor — the ggml tensor being labeled
//   name   — the human-readable label from cb() (may be nullptr)
//   layer  — the layer index, or -1 if not associated with a layer
extern void (*llm_tensor_observe_fn)(struct ggml_tensor * tensor,
                                      const char * name,
                                      int layer);

// Called once after each graph evaluation completes. The tool can use
// this to walk previously-recorded tensors and read their now-computed
// values via ggml_backend_tensor_get. It must NOT modify the graph or
// tensors.
//
// Parameters:
//   gf — the computation graph that just finished evaluating
extern void (*llm_tensor_collect_fn)(const struct ggml_cgraph * gf);

// Called by the ggml backend scheduler for each node during graph
// compute. This is the CORRECT hook for reading tensor values: it
// fires AFTER an op has computed and BEFORE the next op runs, so
// the tensor's buffer is guaranteed to hold the freshly-computed
// output and has not yet been reused by downstream ops.
//
// Two-phase invocation per node, per ggml's eval_callback contract:
//
//   1. Ask phase  (ask=true):  called BEFORE the op runs. Return
//                              true to register interest in this
//                              tensor's output. Return false to
//                              skip — the read phase will not fire.
//   2. Read phase (ask=false): called AFTER the op runs IFF ask
//                              returned true. The tensor's buffer
//                              now holds its freshly-computed value;
//                              read it via ggml_backend_tensor_get.
//
// This hook obsoletes the capture/collect pattern of the older
// observe/collect functions for any use case that needs accurate
// tensor values (rather than just structural metadata). Use it
// instead of llm_tensor_observe_fn + llm_tensor_collect_fn for
// numerical analysis.
//
// The user_data parameter is forwarded from llm_tensor_eval_user_data.
//
// Returning false from ask phase is also the right call for tensors
// the tool doesn't care about — keeps overhead near zero for the
// vast majority of unnamed internal nodes.
extern bool (*llm_tensor_eval_fn)(struct ggml_tensor * t, bool ask, void * user_data);

// User data pointer passed through to llm_tensor_eval_fn. The tool
// sets this when it sets llm_tensor_eval_fn; the engine forwards it
// unchanged to every callback invocation.
extern void * llm_tensor_eval_user_data;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Inline helpers. Calling code uses these instead of checking the
// pointer manually. The compiler optimizes the null check trivially
// when the pointers are unset.
inline void llm_tensor_observe_call(struct ggml_tensor * tensor,
                                     const char * name,
                                     int layer) {
    if (llm_tensor_observe_fn) {
        llm_tensor_observe_fn(tensor, name, layer);
    }
}

inline void llm_tensor_collect_call(const struct ggml_cgraph * gf) {
    if (llm_tensor_collect_fn) {
        llm_tensor_collect_fn(gf);
    }
}
#endif
