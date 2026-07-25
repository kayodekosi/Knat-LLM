# Results template

Populate this after running `benchmark/run_benchmark.py` against the FP16, INT8, and AWQ
endpoints. A representative results table looks like this (numbers below are placeholders —
replace with your own hardware's measurements):

| Config | Throughput (tok/s) | Latency P50 (s) | Latency P95 (s) | VRAM used | Relative quality |
|--------|--------------------:|-----------------:|-----------------:|-----------|-------------------|
| FP16   | —                    | —                 | —                 | ~14 GB (7B model) | baseline |
| INT8   | —                    | —                 | —                 | ~7-8 GB   | evaluate via a fixed eval set (see below) |
| AWQ    | —                    | —                 | —                 | ~4-5 GB   | evaluate via a fixed eval set (see below) |

## Measuring "quality," not just speed

Throughput and latency are easy to measure and easy to over-index on. To make the quality column
honest, run a small fixed evaluation set (e.g. 20-50 prompts with known-good reference answers, or
a benchmark like a subset of MMLU/GSM8K) through all three configurations and score them
identically — either with exact-match/rubric scoring for factual tasks, or a consistent LLM-as-judge
prompt for open-ended tasks. Report the delta from the FP16 baseline, not just an absolute score,
since the FP16 run's number IS the reference point.

## What to expect, directionally

- **INT8** typically preserves quality very close to FP16 for most tasks, with a meaningful
  memory and throughput improvement.
- **AWQ** (activation-aware weight quantization) often gets closer to FP16 quality than naive
  INT8 at a similar or smaller memory footprint, because it protects the weights that matter most
  for activations rather than quantizing uniformly — but this needs to be verified per model/task,
  not assumed.
- The "right" choice depends on the deployment's actual constraint: VRAM-bound → favor AWQ/INT8;
  latency-SLA-bound → benchmark all three under realistic concurrency, since the throughput gain
  from quantization can matter more than the raw memory saved.
