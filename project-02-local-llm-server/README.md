# Project 2 — Local LLM Server (vLLM, OpenAI-Compatible API)

**Org:** Knat LLM · **Maintained by:** Knatware Technology

Runs an open-weight model (Llama 3 or Mistral) locally with **vLLM**, exposes it behind an
**OpenAI-compatible API**, and benchmarks real throughput in tokens/second. This project is the
first step from "I can call a hosted LLM API" to "I can operate the inference server myself" —
the core skill this entire project series builds toward.

## Why vLLM

vLLM's paged attention and continuous batching give meaningfully better GPU utilization than a
naive Hugging Face `generate()` loop, and it ships an OpenAI-compatible `/v1/chat/completions`
endpoint out of the box — meaning any existing OpenAI SDK client, tool, or eval harness works
against it unmodified. That compatibility is exactly why it's the right first serving engine to learn.

## Architecture

```
┌──────────────┐   OpenAI-compatible   ┌───────────────────────┐
│  Client SDK   │ ───── HTTP/JSON ───▶ │  vLLM OpenAI Server    │
│ (openai lib,  │                      │  (Docker container,    │
│  curl, etc.)  │ ◀──── streaming ──── │  1x NVIDIA GPU)         │
└──────────────┘                       └───────────────────────┘
                                               │
                                     Hugging Face model cache
                                        (persisted volume)
```

## Prerequisites

- A machine with an NVIDIA GPU, the NVIDIA driver, and the **NVIDIA Container Toolkit** installed
  (so Docker can pass the GPU through with `--gpus`)
- A Hugging Face account/token if the chosen model is gated (e.g. Llama 3)
- Docker + Docker Compose

## Getting started

```bash
git clone <this-repo-url>
cd project-02-local-llm-server
cp .env.example .env         # set MODEL_NAME and HUGGING_FACE_HUB_TOKEN
docker compose up --build
```

First launch will download and cache model weights — expect this to take a while depending on
model size and connection speed. Subsequent restarts reuse the `hf_cache` volume.

Once running, the server behaves like the OpenAI API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "Explain paged attention in two sentences."}]
  }'
```

Or with the official Python SDK, pointed at the local base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## Benchmarking

`benchmark/benchmark_tokens.py` fires a configurable batch of prompts at the running server and
reports aggregate throughput and latency percentiles:

```bash
pip install -r requirements.txt
python benchmark/benchmark_tokens.py \
  --base-url http://localhost:8000/v1 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --num-requests 20 \
  --concurrency 4
```

It reports:
- **Aggregate throughput (tok/s)** across concurrent requests — the number that matters for
  capacity planning
- **Latency p50 / p95 / mean** — the numbers that matter for user experience

## Key config knobs (in `.env`)

| Variable                   | Purpose                                                      |
|----------------------------|---------------------------------------------------------------|
| `MODEL_NAME`                | Hugging Face repo ID of the model to serve                    |
| `MAX_MODEL_LEN`             | Max context length vLLM will allocate KV-cache space for       |
| `GPU_MEMORY_UTILIZATION`   | Fraction of GPU memory vLLM is allowed to claim for KV cache   |

## What this project deliberately practices

- Running a GPU-backed inference server in Docker (device passthrough, driver compatibility)
- Understanding the OpenAI API contract well enough to swap a hosted model for a self-hosted one
- Load-testing an inference endpoint and reading throughput/latency tradeoffs, not just "does it work"

## Natural next steps

- Feed results into **Project 6** (FP16 vs INT8 vs AWQ) for a head-to-head optimization study
- Move this same server onto Kubernetes in **Project 3**
- Scale it across multiple GPUs with tensor parallelism in **Project 7**

## Enquiries & implementation support

For enquiries, custom implementation, or production-hardening of this serving setup, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
