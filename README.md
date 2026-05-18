# Spiral

**Geometric compression of rotated transformers.**

Spiral exploits the geometric structure of transformer weights to achieve calibration-free compression competitive with the strongest community quantization schemes — without calibration data, without fine-tuning, without representative samples.

The compression is a deterministic function of the model weights. Anyone with the upstream weights reconstructs bit-identical artifacts. No imatrix construction, no domain-specific fine-tuning, no English-corpus bias baked in.

**Latest release: v0.3.0 — Spiral_4_5 (~5.0 bpw).** Production-stable on Apple Silicon (Metal) and NVIDIA (CUDA).

Per-model quality, performance, and benchmark numbers live on each model's HuggingFace card. See [Available Models](#available-models).

---

## 📑 Table of Contents

- [Latest Release: v0.3.0](#latest-release-v030)
- [⚡ Install](#-install)
- [🚀 Quick Start](#-quick-start)
- [Available Models](#available-models)
- [Version history](#version-history)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [License](#license)

---

## Latest Release: v0.3.0

Spiral v0.3.0 ships the Spiral_4_5 scheme — mixed-bit quantization with dense rotation. Production-stable on Apple Silicon (Metal) and NVIDIA (CUDA). Calibration-free.

---

## ⚡ Install

### Brew (Mac)

```bash
brew install reinforceai/spiral/spiral
```

Installs the Spiral fork of `llama.cpp` plus the wrappers: `spiral-chat`, `spiral-serve`, `spiral-download`.

### Build from source

For CUDA, or to develop against the framework:

```bash
git clone https://github.com/ReinforceAI/spiral
cd spiral

# Apple Silicon (Metal):
cmake -B build -DGGML_METAL=ON
cmake --build build -j

# NVIDIA CUDA (H100, A100, RTX with CC ≥ 7.5):
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

Standard upstream `llama.cpp` does not load Spiral-compressed models.

---

## 🚀 Quick Start

### Interactive chat

```bash
spiral-chat                                    # default model
spiral-chat --model qwen-36-35b-spiral         # specific model by name
```

First run auto-downloads the GGUF and codebook to `~/.spiral/models/<name>/`. Subsequent runs use the local cache.

### Single prompt (non-interactive)

```bash
spiral-chat --model qwen-36-35b-spiral \
    --prompt "Write a Python function to compute Fibonacci numbers iteratively" \
    --greedy
```

### OpenAI-compatible API server

```bash
spiral-serve --model qwen-36-35b-spiral --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "messages": [{"role": "user", "content": "Hello"}]
    }'
```

### Manual download

If you'd rather drive `llama.cpp` directly:

```bash
hf download Reinforce-ai/<model-name> \
    <model-name>.gguf <model-name>.spiralcb \
    --local-dir ./models/spiral/
```

```bash
SPIRAL_CODEBOOK_PATH=./models/spiral/<model-name>.spiralcb \
./build/bin/llama-cli \
    -m ./models/spiral/<model-name>.gguf \
    -ngl 99 \
    -ctk f16 -ctv f16 -fa on \
    -c 8192 -cnv
```

`SPIRAL_CODEBOOK_PATH` is required for every Spiral inference.

---

## Available Models

For per-model quality numbers, throughput measurements, and benchmark results, see the model card on each HuggingFace repository.

| Model | spiral-chat name | Version | Scheme | Base | HuggingFace |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | `qwen-25-7b-spiral` | v0.2.0 | Spiral_3 | Qwen2.5-Coder-7B | [Reinforce-ai/spiral-qwen2.5-coder-7b](https://huggingface.co/Reinforce-ai/spiral-qwen2.5-coder-7b) |
| Qwen3.6-35B-A3B | `qwen-36-35b-spiral` | v0.3.0 | Spiral_4_5 | Qwen3.6-35B-A3B | [Reinforce-ai/Qwen3.6-35B-A3B-Spiral](https://huggingface.co/Reinforce-ai/Qwen3.6-35B-A3B-Spiral) |
| Sarvam-30B | `sarvam-30b-spiral` | v1.0 | Spiral_Q4 | sarvamai/sarvam-30b | [Reinforce-ai/sarvam-30b-Spiral](https://huggingface.co/Reinforce-ai/sarvam-30b-Spiral) |

---

## Version history

### v0.3.0 (current, May 2026)
- Mixed-bit weight compression scheme (Spiral_4_5)
- CUDA support for NVIDIA H100, A100, RTX (CC ≥ 7.5)
- Multi-turn conversational use stable on both Metal and CUDA
- Calibration-free
- Status: production-grade

### v0.2.0 (April 2026)
- INT3 weight compression with KV cache product quantization (Spiral_3)
- Apple Silicon (Metal) only
- **Known issue:** multi-turn conversational mode on Mac produced hallucinated context for some prompts. Workaround: single-prompt mode.
- Status: superseded by v0.3.0. v0.3.0 recommended for new deployments.

---

## Limitations

- **Custom llama.cpp build required.** Stock upstream `llama.cpp` does not load Spiral models — the SPIRAL_INT4 ggml type and rotation hooks live in the Spiral fork.
- **KV cache compression not in v0.3.0.** Long-context workloads run at f16 KV. Future releases may revisit.
- **Research-grade release.** APIs may change before 2.0.

Per-model limitations and not-yet-measured evaluations are documented on each model's HuggingFace card.

---

## Acknowledgments

Spiral builds on open-source infrastructure:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** by Georgi Gerganov and contributors — inference engine, GGUF format, Metal and CUDA backends. The Spiral fork extends llama.cpp with the SPIRAL_INT4 ggml type, rotation hooks, and codebook loader infrastructure.
- **[TurboQuant](https://github.com/turbo-llm/turbo3)** by Eric Kryski — fused asymmetric attention kernels and tensor-core dispatch patterns.
- The broader open-source ML community — quantization theory, rotation methods, and product quantization research laid the groundwork that Spiral builds upon.

This work would not be possible without the engineers who contribute to open source inference infrastructure.

---

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

- Inference engine: Based on llama.cpp (MIT)
- Spiral compression framework: ReinforceAI
- Model weights: Subject to each base model's license (see individual HuggingFace cards)