# Launching on a rented multi-GPU pod (RunPod / Lambda / Vast.ai)

This project needs 2 GPUs on the same machine with NVLink or at least fast PCIe interconnect
between them — a good fit for a rented multi-GPU pod rather than a personal workstation.

## RunPod

1. Create a Pod using a template with CUDA + Docker pre-installed (e.g. "RunPod PyTorch" template),
   selecting a GPU type with **2x GPUs** attached (e.g. 2x A100 80GB or 2x A6000, depending on
   model size and budget).
2. SSH into the pod, clone this repo, `cp .env.example .env` and fill in `HUGGING_FACE_HUB_TOKEN`.
3. Run `docker compose up --build -d`. vLLM will shard the model across both visible GPUs
   automatically once `--tensor-parallel-size 2` is set and `nvidia-smi` inside the container shows
   2 devices.
4. Confirm both GPUs are in use: `nvidia-smi` should show memory allocated on both indices while a
   request is in flight.

## Lambda Labs

Same flow — provision an on-demand instance with 2 GPUs, ensure the NVIDIA Container Toolkit is
installed (Lambda's stock images typically include it), then follow steps 2-4 above.

## Vast.ai

When selecting an instance, filter for offers with **2 GPUs** and verify the interconnect type
listed for the offer (NVLink is ideal; PCIe is workable but the parallel-communication overhead
will be higher — factor this into the benchmark comparison below). Follow steps 2-4 above once
connected.

## Cost-consciousness

Multi-GPU instances are billed per GPU-hour on all these providers. Run the benchmark
(`benchmark/benchmark_multigpu.py`), capture the results, and **tear the pod down** — this project
does not need a persistent multi-GPU instance running between sessions.
