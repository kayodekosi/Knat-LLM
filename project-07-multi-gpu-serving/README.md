# Project 7 — Multi-GPU Serving: Tensor-Parallel Deployment Across 2 GPUs

**Org:** Knat LLM · **Maintained by:** Knatware Technology

Serves a large model sharded across **2 GPUs** using vLLM's tensor parallelism, running on a
rented multi-GPU pod (RunPod, Lambda, or Vast.ai). This is the capstone project in the series —
the point where a model is too large (or too latency-sensitive) to serve well on a single GPU, and
splitting it across hardware becomes necessary rather than optional.

## Why tensor parallelism

A single GPU has a hard memory ceiling. A 70B-parameter model in FP16 needs roughly 140GB of
weights alone — well beyond a single consumer or even most single data-center GPUs. **Tensor
parallelism** shards individual weight matrices across GPUs (splitting the matrix multiplications
themselves), so each GPU holds a slice of every layer rather than a full copy of some layers. This
is different from simply running two independent model replicas — the GPUs must communicate
during every forward pass, which is why interconnect speed (NVLink vs. PCIe) directly affects
throughput.

## Architecture

```
                     ┌───────────────────────────────────────────┐
                     │             Rented GPU pod                  │
                     │                                              │
   Client ────HTTP──▶│   vLLM OpenAI-compatible server              │
                     │        │                                    │
                     │        ├──▶ GPU 0: shard of every layer       │
                     │        └──▶ GPU 1: shard of every layer       │
                     │              (NVLink/PCIe all-reduce during   │
                     │               attention + MLP computation)     │
                     └───────────────────────────────────────────┘
```

## Getting started

This project is designed to be run on a **rented** multi-GPU instance, not a personal machine —
see `scripts/launch_runpod.md` for step-by-step provisioning notes covering RunPod, Lambda Labs,
and Vast.ai, including what to look for in the interconnect type of the offer you select.

Once you have a 2-GPU instance with Docker + the NVIDIA Container Toolkit ready:

```bash
git clone <this-repo-url>
cd project-07-multi-gpu-serving
cp .env.example .env      # set MODEL_NAME (choose one that needs 2 GPUs to fit comfortably)
docker compose up --build
```

`docker-compose.yml` requests both GPUs (`count: 2`) and sets `--tensor-parallel-size 2`, which
tells vLLM to shard the model. Confirm both GPUs are actually being used during a request:

```bash
watch -n1 nvidia-smi
```

You should see memory allocated and utilization on **both** GPU indices while a request is in
flight — if only one shows activity, tensor parallelism isn't actually active and it's worth
checking the container logs for a fallback warning.

## Benchmarking

```bash
pip install openai
python benchmark/benchmark_multigpu.py \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --num-requests 30 --concurrency 8
```

## An honest note on what "success" looks like here

Tensor parallelism across 2 GPUs is not primarily a speed optimization for a model that already
fits on one GPU — communication overhead between GPUs can make it *slower* per-token than a single
well-utilized GPU in some cases. Its real value is **enabling deployment of a model that would not
fit, or would not fit with acceptable context length, on a single GPU at all**. The benchmark
script deliberately calls this out rather than presenting the numbers as an unqualified win — the
correct comparison in a real evaluation is "can we serve this model at all" and "at what cost per
request," not just raw tok/s against an unrelated smaller model.

## What this project deliberately practices

- Configuring and validating that tensor parallelism is genuinely active (not just requested)
- Reasoning about GPU interconnect (NVLink vs. PCIe) as a real performance variable, not an afterthought
- Provisioning and tearing down rented multi-GPU infrastructure responsibly (cost-consciousness)
- Benchmarking a distributed serving setup and reporting results honestly, including when the
  comparison isn't apples-to-apples

## Natural next steps

- Combine with **Project 6**'s AWQ/INT8 findings to test quantization + tensor parallelism together
- Compare tensor parallelism against pipeline parallelism for the same model and hardware budget
- Move this onto the Kubernetes setup from **Project 3** using multi-GPU node pools, once budget
  allows for a persistent (rather than rented/ephemeral) multi-GPU node

## Enquiries & implementation support

For enquiries or help sizing and provisioning multi-GPU infrastructure for a specific model, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
