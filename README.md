
# Spiral

**Geometric compression of rotated transformers.**

Spiral exploits the geometric structure of transformer activations to compress weights to INT3 and KV cache to INT2 — without calibration data, without fine-tuning. Two results:

1. **INT3 weights at +0.14 nats** — 101× quality improvement over naive 3-bit, competitive with calibration-based approaches (GPTQ, AWQ, QuIP#) that require representative data.

2. **INT2 PQ KV cache at 7.1× K compression** — product quantization reduces per-token KV memory from 56 KB to 32 KB (K+V combined), scaling context capacity by 1.75× at any memory budget. With full K+V PQ (in progress), this reaches 7.1× total compression.

## INT3 Weight Quality

Measured eval perplexity gap vs fp16:

**Qwen2.5-Coder-7B-Instruct (dense):**

| Method | Bits | Gap (nats) | Calibration Data Required |
|--------|------|-----------|--------------------------|
| Naive round-to-nearest | 3 | +14.2 | No |
| GPTQ | 3 | ~+0.8 | Yes (128 samples) |
| AWQ | 3 | ~+0.6 | Yes (calibration set) |
| QuIP# | 3 | ~+0.3 | Yes (calibration set) |
| **Spiral** | **3** | **+0.141** | **No** |

*GPTQ/AWQ/QuIP# gaps are approximate values from published literature at comparable model scales, not measured on this specific model.*

**Qwen3-Coder-30B-A3B-Instruct (MoE, 128 experts):**

| Method | Size | vs Spiral |
|--------|------|-----------|
| Q4_K_M (GGUF) | 18.6 GB | 60% larger |
| Q3_K_M (GGUF) | 15.3 GB | 32% larger |
| Q3_K_S (GGUF) | 14.2 GB | 22% larger |
| Q2_K_M (GGUF) | 11.8 GB | Similar size, higher quality loss |
| **Spiral INT3 + PQ KV** | **11.6 GB** | **+0.228 nats, + 7.1× KV compression** |

Spiral achieves Q2-level model size while maintaining Q3-level quality — measured at +0.228 nats vs fp16 baseline (2.212 nats). No standard GGUF method includes KV cache compression; Spiral adds 7.1× K compression on top, enabling 75% more context at any memory budget.

The rotation is a deterministic, seeded orthonormal transform that works on any architecture — dense or MoE, any head dimension, any RoPE frequency. No calibration data, no gradient updates, no fine-tuning.

## KV Cache Compression

Per-token KV memory comparison for a 7B model (28 layers, 4 KV heads, 128 head_dim):

| KV Method | K bits/dim | V bits/dim | Per-token KV | K Compression |
|-----------|-----------|-----------|-------------|--------------|
| F16 (standard) | 16 | 16 | 56.0 KB | 1× |
| Q8_0 | 8 | 8 | 28.0 KB | 2× |
| Q4_0 | 4 | 4 | 14.0 KB | 4× |
| **Spiral PQ (K only)** | **2.1** | **16** | **31.9 KB** | **7.1× (K)** |
| Spiral PQ (K+V, planned) | 2.1 | 2.1 | 7.9 KB | 7.1× (K+V) |

## Context Scaling

How Spiral's PQ KV extends context at every memory tier (using a 3 GB Spiral model):

| Hardware | F16 KV Context | Spiral PQ Context | Headroom |
|----------|---------------|-------------------|----------|
| 8 GB Mac | 65K tokens | 113K tokens | +74% |
| 16 GB Mac | 205K tokens | 360K tokens | +76% |
| 24 GB Mac | 345K tokens | 606K tokens | +76% |
| 64 GB Mac | 1.1M tokens | 1.9M tokens | +73% |
| 128 GB Mac | 2.2M tokens | 3.9M tokens | +77% |

For long-horizon agent tasks — multi-file code generation, repository-scale analysis, extended conversations — context capacity is the binding constraint. PQ KV trades ~34% decode speed for 75% more context at every memory tier.

## How It Works

### The Geometry

Trained transformer weights are not random matrices. They exhibit structure that compression can exploit:

**Observation 1: Hypersphere concentration.** Weight rows concentrate near a thin shell on the unit hypersphere (norm CV ≈ 0.02). Direction carries the information; amplitude is nearly constant. This enables sign/amplitude decoupling.

**Observation 2: Rotated Gaussianity.** Applying a random orthonormal rotation (Walsh-Hadamard transform) to any trained weight row produces nearly Gaussian marginals with equalized variance across all dimensions. Outlier channels — the primary source of quantization error — vanish under rotation.

**Observation 3: PQ subspace adaptation.** Product quantization with 256 learned codewords per 4-dimensional subspace captures 68.5% of the scalar-to-Shannon compression gap for KV activations. Natural-space codebooks (no rotation needed for KV) add only +0.02 nats — learned codebooks adapt to non-uniform dimensional importance inherently.

### Unified Rotation

Spiral applies the same mathematical primitive — multi-pass block Walsh-Hadamard rotation — to both weights and activations:

**Weights (offline):** Rotate → quantize to INT3 with Lloyd-Max optimal centroids → store. At inference, rotate the input activation by the same transform before matmul. Cost: O(d log d) per token via fast WHT.

**KV cache (online):** K vectors are compressed via product quantization into 32 codebook indices (34 bytes per 128-dim vector). A fused Metal kernel decodes PQ codes, applies RoPE, and computes attention in a single pass — no intermediate tensor materialized.

### Custom Metal Kernels

Spiral includes purpose-built GPU kernels for Apple Silicon:

- **Fused flash attention with inline PQ decode** — one kernel launch for codebook lookup + RoPE + Q·K scoring + softmax + V accumulation. Reduces compute buffer from 2 GB (graph-level decode) to 304 MB. RoPE frequency base is parameterized from the GGUF (supports 10K for Qwen2.5, 10M for Qwen3).
- **Multi-pass Walsh-Hadamard rotation** — seeded random orthonormal transform at O(d log d) per token, matching rotated weight basis. Adapts to any dimension (768, 2048, 3584, 4096, 18944).
- **Online PQ encode** — compresses incoming K vectors to codebook indices during inference using L2 nearest-neighbor search.
- **MoE expert dispatch** — rotation applied before expert gate/up projections and before down projections inside the MoE FFN, with type-guarded checks so non-Spiral models are unaffected.

## Performance

Measured on Apple M2 Pro (16 GB):

| Mode | Decode | Prefill |
|------|--------|---------|
| F16 KV | 29 tok/s | 140 tok/s |
| PQ KV | 19 tok/s | 190 tok/s |

## Install

```bash
brew install reinforceai/spiral/spiral
```

## Quick Start

```bash
spiral-chat                              # interactive chat
spiral-chat --prompt "explain quicksort"  # single response
spiral-serve --port 8080                  # OpenAI-compatible API
```

## Available Models

| Model | Size | Base | Architecture | Min RAM |
|-------|------|------|-------------|---------|
| `qwen-25-7b-spiral` | 3.02 GB | Qwen2.5-Coder-7B-Instruct | Dense | 8 GB |
| `qwen3-coder-30b-spiral` | 11.61 GB | Qwen3-Coder-30B-A3B-Instruct | MoE (128 experts, 8 active) | 24 GB |

```bash
spiral-chat --model qwen-25-7b-spiral
spiral-download --model qwen-25-7b-spiral
```

## Compression Breakdown

Per-component quality cost:

**Qwen2.5-Coder-7B (dense, 3.02 GB):**

| Component | Method | Compression | Quality Cost |
|-----------|--------|------------|-------------|
| Weights | Rotated Lloyd-Max INT3 | 4.2× | +0.141 nats |
| KV cache (K) | Natural-space PQ INT2 | 7.1× | +0.090 nats |
| Embeddings | Asymmetric affine INT4 | 4.0× | +0.017 nats |
| **Full pipeline** | | **4.8× model, 7.1× KV** | **+0.184 nats** |

**Qwen3-Coder-30B-A3B (MoE, 11.61 GB):**

| Component | Method | Compression | Quality Cost |
|-----------|--------|------------|-------------|
| Weights (12,480 matrices) | Rotated Lloyd-Max INT3 | 5.3× | ~+0.16 nats† |
| KV cache (K) | Natural-space PQ INT2 | 7.1× | ~+0.07 nats† |
| Embeddings | Asymmetric affine INT4 | 4.0× | ~+0.02 nats† |
| **Full pipeline** | | **5.3× model, 7.1× KV** | **+0.228 nats** |

*†Per-component estimates based on 7B component ratios. End-to-end gap (+0.228) is measured directly.*

The same compression physics applies to both dense and MoE architectures. Each expert's weight matrix is compressed independently — the rotation adapts to any input dimension (768, 2048, 4096). Router weights stay at fp16 for full-precision expert selection.

## Acknowledgments

Spiral builds on open-source foundations:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** by Georgi Gerganov — inference engine, GGUF format, Metal backend. Spiral's deployment infrastructure inherits directly from this project.

- **[TurboQuant](https://github.com/turbo-llm/turbo3)** by Eric Kryski — fused asymmetric attention kernels and two-pass flash attention on Metal. The TurboFlash architecture directly inspired Spiral's fused PQ attention kernel.

- **[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** by TheTom — llama.cpp integration of TurboQuant, providing the foundation for Spiral's Metal kernel dispatch, GGUF type registration, and graph-level quantized inference pipeline.

- **Qwen Team** — Qwen2.5-Coder under Apache 2.0.

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
Model weights: Subject to base model license (e.g., Apache 2.0 for Qwen2.5-Coder)