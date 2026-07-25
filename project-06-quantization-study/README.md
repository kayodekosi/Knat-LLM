# Project 6 — Optimization Study: FP16 vs INT8 vs AWQ

**Org:** Knat LLM · **Maintained by:** Knatware Technology

The same model, served three ways — full-precision **FP16**, **INT8** post-training quantization,
and **AWQ** (activation-aware weight quantization) — benchmarked head-to-head on latency,
throughput, memory footprint, and output quality. This project turns "quantization makes things
faster" from a talking point into a measured, published tradeoff table.

## Why this project exists

Every deployment decision in this series so far (Projects 2, 3, 7) assumes a model is being served
in some precision. This project answers the question those projects take for granted: **which
precision, and what do you actually give up to get there?** The goal is a defensible, numbers-backed
answer rather than a rule of thumb repeated from a blog post.

## What's being compared

| Config | What it is                                                                 | Expected tradeoff                       |
|--------|------------------------------------------------------------------------------|-------------------------------------------|
| FP16   | The baseline — full 16-bit floating point weights                            | Highest quality, highest memory/compute    |
| INT8   | Post-training 8-bit integer quantization                                     | ~2x memory reduction, small quality risk    |
| AWQ    | 4-bit quantization that protects activation-critical weight channels          | Larger memory reduction, aims to preserve quality better than naive 4-bit |

## Running the study

**1. Produce the AWQ checkpoint** (FP16 and INT8 can typically be selected as a vLLM launch flag
directly, without a separate conversion step for INT8 in supported configurations; AWQ needs an
explicit calibration/quantization pass):

```bash
pip install -r requirements.txt
python benchmark/quantize_awq.py \
  --model-path mistralai/Mistral-7B-Instruct-v0.3 \
  --output-path ./models/mistral-7b-awq
```

**2. Serve all three configurations** as separate vLLM containers on different ports (reusing the
Docker setup from Project 2, with `--dtype float16`, `--quantization bitsandbytes` or equivalent
INT8 flag, and `--quantization awq --model ./models/mistral-7b-awq` respectively).

**3. Run the comparison benchmark:**

```bash
cp benchmark/endpoints.json.example benchmark/endpoints.json   # edit ports/model names to match your setup
python benchmark/run_benchmark.py --config benchmark/endpoints.json --num-requests 20 --concurrency 4
```

This produces `results/benchmark_results.csv` with throughput and latency percentiles for all
three configurations under an identical prompt/concurrency profile — the same prompt set and load
shape for each, so the comparison isn't confounded by different test conditions.

**4. Score quality separately.** `results/README.md` explains why speed metrics alone aren't
enough and lays out how to score output quality against a fixed evaluation set so the "what do
you give up" half of the tradeoff table isn't left blank.

## What this project deliberately practices

- Actually producing a quantized checkpoint (AWQ), not just reading about the technique
- Designing a benchmark that holds prompts, concurrency, and hardware constant across configurations,
  so the comparison is real and not confounded
- Resisting the temptation to report only throughput — pairing it with a quality-delta measurement
- Turning an optimization technique into a documented, reusable decision framework for future model
  deployments in this series

## Natural next steps

- Extend the comparison to GPTQ or SmoothQuant for a fourth data point
- Automate the quality-scoring step with an LLM-as-judge harness instead of manual review
- Fold the winning configuration back into Project 7's multi-GPU deployment for a combined
  optimization + parallelism benchmark

## Enquiries & implementation support

For enquiries or help running this study against your own model/hardware combination, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
