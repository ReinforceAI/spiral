# Spiral

**Run a 7B coding model on your Mac with 200K+ token context.**

Spiral compresses Qwen2.5-Coder-7B to 3 GB using physics-derived INT3 weight quantization and INT2 PQ KV cache compression. The result: a full coding assistant that fits in 8 GB RAM with 7.5× more context than standard approaches.

## Install

```bash
brew install spiral
```

## Quick Start

```bash
# Interactive chat
spiral-chat

# Single prompt
spiral-chat --prompt "Write a Python function to find the longest palindrome substring"

# API server (OpenAI-compatible)
spiral-serve
```

On first run, Spiral automatically downloads the model (~3 GB) to `~/.spiral/models/`.

## Usage

### Chat

```bash
spiral-chat                          # interactive conversation
spiral-chat --prompt "explain quicksort"  # single response
spiral-chat --greedy                 # deterministic output
spiral-chat --no-pq                  # disable KV compression (faster, more memory)
spiral-chat --max-tokens 4096        # longer responses
```

### API Server

```bash
spiral-serve                        # starts on localhost:8080
spiral-serve --port 3000             # custom port
spiral-serve --host 0.0.0.0          # listen on all interfaces
```

Call the API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a binary search in Rust"}],
    "max_tokens": 1024
  }'
```

Compatible with any OpenAI client library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="spiral",
    messages=[{"role": "user", "content": "Explain async/await in Python"}]
)
print(response.choices[0].message.content)
```

### Model Management

```bash
spiral-download              # download/update model
spiral-download --force      # re-download even if exists
```

Models are stored in `~/.spiral/models/`. Override with `SPIRAL_HOME`:

```bash
export SPIRAL_HOME=/path/to/models
```

## Performance

Measured on Apple M2 Pro (16 GB):

| Mode | Decode Speed | Max Context (8 GB Mac) |
|------|-------------|----------------------|
| Standard (F16 KV) | 29 tok/s | 21K tokens |
| PQ KV compression | 19 tok/s | 283K tokens |

PQ KV compression trades ~10 tok/s for 7.5× more context capacity. Enable with `--pq` (default) or disable with `--no-pq`.

## How It Works

Spiral uses three compression techniques derived from transformer geometry:

1. **INT3 Rotated Weights** — Multi-pass Walsh-Hadamard rotation + Lloyd-Max 3-bit quantization. 4.2× weight compression at +0.14 nats quality cost.

2. **INT2 PQ KV Cache** — Product quantization with natural-space codebooks. 7.5× KV memory compression at +0.09 nats quality cost. Enables 200K+ token context on consumer hardware.

3. **INT4 Embeddings** — Per-row asymmetric affine quantization. 4× embedding compression at +0.02 nats quality cost.

Total: 14.5 GB → 3.0 GB with 22% perplexity cost. The model produces correct, well-structured code.

## Requirements

- macOS 13+ (Apple Silicon: M1, M2, M3, M4)
- 8 GB RAM minimum (16 GB recommended for long context)
- ~5 GB disk space (3 GB model + 2 GB working space)

## License

Model: Qwen2.5-Coder-7B-Instruct (Apache 2.0)
Engine: Based on llama.cpp (MIT)
Compression: Spiral physics framework (ReinforceAI)
