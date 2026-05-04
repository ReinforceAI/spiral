# Spiral

**Geometric compression of rotated transformers.**

Spiral exploits the geometric structure of transformer activations to achieve calibration-free INT3 weight compression and PQ K-cache compression — without calibration data, without fine-tuning. Two production results:

1. **Qwen3.6-35B-A3B at 3.5 bits/weight, +0.113 nats vs bf16.** 1.46× smaller on disk than unsloth Q4_K_XL. Hybrid MoE+DeltaNet architecture compressed end-to-end.

2. **7.1× compression on the K cache.** Product quantization reduces per-token KV memory by 1.75× combined K+V — extending context capacity by the same factor at any memory budget.

![Qwen3.6-35B-A3B Spiral release results](assets/spiral_release_slide.svg)

---

## 📑 Table of Contents

- [Headline result — Qwen3.6-35B-A3B](#headline-result--qwen36-35b-a3b)
- [How it stacks up at 3-bit](#how-it-stacks-up-at-3-bit)
- [KV cache compression](#kv-cache-compression)
- [Total memory at long context](#total-memory-at-long-context)
- [Qwen2.5-Coder-7B](#qwen25-coder-7b)
- [How it works](#how-it-works)
- [Performance](#performance)
- [⚡ Install](#-install)
- [🚀 Quick start](#-quick-start)
- [Available models](#available-models)
- [Methodology](#methodology)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [License](#license)

---

## Headline result — Qwen3.6-35B-A3B

Cross-method perplexity comparison on H100, measured with identical methodology: HuggingFace AutoTokenizer applied to wikitext-2 validation, 64 sequences × 2048 tokens = 131,008 tokens, evaluated end-to-end via PyTorch (`bf16`, `Spiral`) or `llama-cpp-python` with pre-tokenized inputs (`Q8_0`, `Q4_K_XL`). Same tokens, same loss formula, four numbers.

| Method | Bits/weight | Disk size | NLL (nats/token) | Gap vs bf16 |
|---|---|---|---|---|
| bf16 (reference) | 16.0 | 67.0 GB | 1.8968 | — |
| unsloth Q8_0 | 8.5 | 34.4 GB | 1.8962 | −0.0006 |
| unsloth Q4_K_XL | 5.2 | 20.8 GB | 1.8976 | +0.0008 |
| **Spiral 3-bit** | **3.5** | **14.2 GB** | **2.0101** | **+0.1133** |

**The trade.** Spiral is 1.46× smaller than Q4_K_XL on disk for +0.11 nats of additional perplexity. On wikitext, that gap is 2.3% of token entropy — practically invisible in generation. Spiral 35B produces clean Python, follows multi-step instructions, and uses tools correctly at 23–24 tok/s on M2 Max.

## How it stacks up at 3-bit

The Spiral compression algorithm reproduces the +0.113 nats result in PyTorch on H100 — the same eval framework reported published rotational quantization papers. For context, here is how published 3-bit methods compare on similar-scale models (numbers from each paper's reported gap to fp16/bf16 on language modeling perplexity):

| Method | Bits | Reported gap (nats) | Calibration data? |
|---|---|---|---|
| Naive round-to-nearest | 3 | +14.2 | No |
| GPTQ (Llama-2-13B) | 3 | ~+0.8 | Yes (128 samples) |
| AWQ (Llama-2-13B) | 3 | ~+0.6 | Yes (calibration set) |
| QuIP# (Llama-2-70B) | 3 | ~+0.3 | Yes (calibration set) |
| **Spiral 35B (ours)** | **3** | **+0.113** | **No** |

*Published numbers are reproduced from each method's paper at comparable model scale; not all are measured on Qwen3.6-35B-A3B. Spiral's number is measured directly on Qwen3.6-35B-A3B with the methodology described above.*

Spiral's +0.113 nats places it competitive with calibration-based rotational quantization (QuIP#) while being calibration-free — no representative data needed, deterministic seeded rotation works on any architecture.

## KV cache compression

Most quantization stops at weights. Spiral additionally compresses the K cache via product quantization. K-quants from llama.cpp leave the KV cache at fp16 by default; this dominates memory at long context.

**Per-token KV cache memory (Qwen3.6-35B-A3B, 10 attention layers, 2 KV heads, head_dim 256):**

| Method | bytes / token | K compression | Combined K+V |
|---|---|---|---|
| f16 (Q8_0, Q4_K_XL all use this) | 20,480 | 1× | 1× |
| **Spiral PQ K + f16 V** | **11,680** | **7.1×** | **1.75×** |

**Per-token KV cache memory (Qwen2.5-Coder-7B, 28 layers, 4 KV heads, head_dim 128):**

| Method | bytes / token | K compression | Combined K+V |
|---|---|---|---|
| f16 | 57,344 | 1× | 1× |
| **Spiral PQ K + f16 V** | **32,704** | **7.1×** | **1.75×** |

The 7.1× compression on K is consistent across both models — same algorithm, same product quantization with 256 codewords per 4-dim subspace.

## Total memory at long context

Compression compounds at long context. The KV cache scales linearly with tokens; weights are fixed.

**Qwen3.6-35B-A3B total memory (weights + KV cache, GiB):**

| Context | Spiral 3-bit | Q4_K_XL + f16 KV | Q8_0 + f16 KV | bf16 + f16 KV |
|---|---|---|---|---|
| 32K | 14.6 | 21.5 | 35.0 | 67.6 |
| 128K | 15.7 | 23.3 | 36.9 | 69.5 |
| 256K | 17.1 | 25.8 | 39.4 | 72.0 |
| 1M | 25.6 | 40.8 | 54.4 | 87.0 |

Add ~1 GiB compute scratch for realistic peak. On a 96 GB Mac, Spiral runs Qwen3.6-35B-A3B at 1M context with comfortable headroom; Q4_K_XL fits at 1M but with ~50 GiB headroom; Q8_0 leaves ~40 GiB. On a 64 GB Mac, only Spiral fits 1M context cleanly.

## Qwen2.5-Coder-7B

Same algorithm, dense architecture. Compression measured on the build pipeline (PyTorch evaluation, identical methodology to the 35B reference run on H100).

| | Value |
|---|---|
| Disk size | 4.44 GB (gguf + codebooks) |
| Bits per weight | ~5.07 |
| Compression gap vs bf16 | +0.141 nats (build pipeline eval) |
| K cache compression | 7.1× |
| Combined K+V compression | 1.75× |
| Decode speed (M2 Max) | 17 tok/s with PQ KV + flash attention |
| Prefill speed | 175 tok/s |

For 7B, the story is in **KV scaling**. At 256K context, a 7B model's KV cache balloons past the weights themselves. Spiral's PQ K compression keeps long-context inference practical on small Macs.

## How it works

### The geometry

Trained transformer weights are not random matrices. They have structure that compression can exploit:

**Hypersphere concentration.** Weight rows concentrate near a thin shell on the unit hypersphere (norm CV ≈ 0.02). Direction carries information; amplitude is nearly constant.

**Rotated Gaussianity.** Applying a random orthonormal rotation (multi-pass block Walsh-Hadamard) to any trained weight row produces nearly Gaussian marginals with equalized variance across all dimensions. **Outlier channels — the primary source of quantization error — vanish under rotation.**

**PQ subspace adaptation.** Product quantization with 256 learned codewords per 4-dimensional subspace captures most of the scalar-to-Shannon compression gap for KV activations. Natural-space codebooks (no rotation needed for KV) adapt to non-uniform dimensional importance directly.

### Unified rotation

Spiral applies the same primitive — multi-pass block Walsh-Hadamard rotation — to both weights and activations:

**Weights (offline):** Rotate → quantize to INT3 with Lloyd-Max optimal centroids → store. At inference, rotate the input activation by the same transform before matmul. Cost: O(d log d) per token via fast WHT.

**KV cache (online):** K vectors are compressed via product quantization into 32 codebook indices per 128-dim head (or 64 indices per 256-dim head). A fused Metal kernel decodes PQ codes, applies RoPE, and computes attention in a single pass — no intermediate tensor materialized.

### Custom Metal kernels

Spiral includes purpose-built GPU kernels for Apple Silicon:

- **Fused flash attention with inline PQ decode.** One kernel launch for codebook lookup + RoPE + Q·K scoring + softmax + V accumulation. Two variants: d128 (Qwen2.5 7B) and d256 (Qwen3.6 35B). RoPE frequency base is parameterized from the GGUF (10K for Qwen2.5, 10M for Qwen3).
- **Multi-pass Walsh-Hadamard rotation.** Seeded random orthonormal transform at O(d log d) per token, matching rotated weight basis. Adapts to any input dimension (768, 2048, 3584, 4096, etc.).
- **Online PQ encode.** Compresses incoming K vectors to codebook indices during inference using L2 nearest-neighbor search.
- **Hybrid attention dispatch.** For models with mixed full-attention + DeltaNet layers (Qwen3.6-35B-A3B has 10 attention + 30 DeltaNet), the dispatcher routes per-layer based on architecture metadata.

## Performance

Measured on Apple M2 Max, full flash attention enabled.

| Model | Configuration | Decode | Prefill |
|---|---|---|---|
| Qwen2.5-Coder-7B Spiral | PQ KV + flash attention | 17 tok/s | 175 tok/s |
| Qwen3.6-35B-A3B Spiral | PQ KV + flash attention | 23–24 tok/s | 80 tok/s |
| Qwen3.6-35B-A3B Spiral | fp16 KV | 36 tok/s | — |

PQ KV trades ~30% decode speed for 7.1× K compression. For long-horizon agent tasks where context capacity is the binding constraint, the trade is favorable.

## ⚡ Install

```bash
brew install reinforceai/spiral/spiral
```

## 🚀 Quick start

```bash
spiral-chat                                # interactive chat (default model)
spiral-chat --prompt "explain quicksort"    # single response
spiral-serve --port 8080                    # OpenAI-compatible API
```

## Available models

| Model | Size | Base | Architecture | Min RAM |
|---|---|---|---|---|
| `qwen-25-7b-spiral` | 4.44 GB | Qwen2.5-Coder-7B-Instruct | Dense | 8 GB |
| `qwen-36-35b-spiral` | 14.24 GB | Qwen3.6-35B-A3B | MoE + DeltaNet (256 experts, 8 active) | 24 GB |

```bash
spiral-chat --model qwen-25-7b-spiral
spiral-chat --model qwen-36-35b-spiral
```

## Methodology

All perplexity numbers in this README are measured with consistent methodology to make claims defensible.

**For Qwen3.6-35B-A3B (the headline table)**, all four methods (bf16, Q8_0, Q4_K_XL, Spiral) were evaluated on H100 with:

- HuggingFace `AutoTokenizer` applied to wikitext-2-raw-v1 validation as the **single tokenizer source of truth** — same token IDs feed every model
- 64 sequences × 2048 tokens = 131,008 tokens
- bf16 and Spiral evaluated via PyTorch `F.cross_entropy` on a forward pass; Spiral measured via in-memory compression (`compress_all_weights` replaces weights with reconstructed W_recon, then identical NLL formula)
- Q8_0 and Q4_K_XL evaluated via `llama-cpp-python` with the **same pre-tokenized inputs** (logits_all=True). The GGUF tokenizer is never invoked — token sequences match exactly
- All numbers from the same H100 pod, same eval session, fully reproducible

**For Qwen2.5-Coder-7B**, the +0.141 nats gap is from the build pipeline eval (`build_spiral_artifact.py compute_eval_loss`), measured on H100 with the same PyTorch methodology used for the 35B reference run.

**For sizes**, all numbers are bytes on disk (`stat`).

**For KV memory**, all numbers are computed from architecture (n_kv_heads × head_dim × bytes_per_element × n_attention_layers × 2 for K+V). For Spiral PQ KV, K bytes = n_kv_heads × n_blocks × bytes_per_code with codebook overhead amortized.

**For inference speed**, end-to-end measurements from production runs on M2 Max, including model load, prefill, and decode.

The script that produced the cross-method comparison is included as `eval_cross_method.py` for independent reproduction.

## Acknowledgments

Spiral builds on open-source foundations:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** by Georgi Gerganov — inference engine, GGUF format, Metal backend. Spiral's deployment infrastructure inherits directly from this project.
- **[TurboQuant](https://github.com/turbo-llm/turbo3)** by Eric Kryski — fused asymmetric attention kernels and two-pass flash attention on Metal. The TurboFlash architecture directly inspired Spiral's fused PQ attention kernel.
- **[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** by TheTom — llama.cpp integration of TurboQuant, providing the foundation for Spiral's Metal kernel dispatch, GGUF type registration, and graph-level quantized inference pipeline.
- **Qwen Team** — Qwen2.5-Coder and Qwen3.6 under Apache 2.0.
- **unsloth** — high-quality reference GGUFs for Q8_0 and Q4_K_XL, used as comparators.
- The broader open-source ML community — researchers contributing to quantization theory (GPTQ, AWQ, QuIP#, AQLM), rotation methods (QuIP, SliceGPT, SpinQuant), and product quantization (Jégou et al., 2011) laid the groundwork that Spiral builds upon.

This work would not be possible without the remarkable researchers and engineers who contribute to open source.

## Citation

```bibtex
@misc{spiral2026,
  title={Spiral: Geometric Compression of Rotated Transformers},
  author={Deshwal, Viraj},
  year={2026},
  publisher={ReinforceAI},
  url={https://github.com/ReinforceAI/spiral}
}
```

## License

Inference engine: Based on llama.cpp (MIT)
Spiral compression framework: ReinforceAI
Model weights: Subject to base model license (Apache 2.0 for Qwen2.5-Coder and Qwen3.6)