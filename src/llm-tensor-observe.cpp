// llm-tensor-observe.cpp
//
// Definition of the tensor observation function pointers.
// Both default to null, making the hooks zero-overhead when no
// observer has registered.
//
// External tools (e.g., spiral-trace) set these pointers at runtime,
// typically from a dynamically-loaded library's constructor. See the
// header file for the contract that observers must honor.

#include "llm-tensor-observe.h"

extern "C" {

void (*llm_tensor_observe_fn)(struct ggml_tensor *, const char *, int) = nullptr;
void (*llm_tensor_collect_fn)(const struct ggml_cgraph *) = nullptr;

// New eval-callback hook for per-op tensor reads. Default null; tools
// register a function to participate in the ggml backend scheduler's
// per-node callback chain. See the header for the contract.
bool (*llm_tensor_eval_fn)(struct ggml_tensor *, bool, void *) = nullptr;
void * llm_tensor_eval_user_data = nullptr;

}
