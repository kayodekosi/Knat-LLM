"""
Knat LLM — Project 7: benchmarks the tensor-parallel deployment and reports
throughput/latency, alongside a note on how to compare it fairly against a
single-GPU baseline (e.g. the Project 2 server) for the same model family.

Usage:
    python benchmark/benchmark_multigpu.py --base-url http://localhost:8000/v1 \
        --model meta-llama/Meta-Llama-3-70B-Instruct --num-requests 30 --concurrency 8
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

PROMPTS = [
    "Explain tensor parallelism versus pipeline parallelism, and when each is preferred.",
    "What communication pattern does tensor-parallel attention require between GPUs?",
    "Describe the tradeoffs of NVLink versus PCIe for multi-GPU inference.",
    "Write a function that computes the nth Fibonacci number iteratively.",
    "Summarize why larger models often require sharding across multiple GPUs to serve at all.",
]


def run_single(client, model, prompt, max_tokens):
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return elapsed, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    latencies, token_counts = [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_single, client, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens)
            for i in range(args.num_requests)
        ]
        start_all = time.perf_counter()
        for future in as_completed(futures):
            elapsed, tokens = future.result()
            latencies.append(elapsed)
            token_counts.append(tokens)
        total_time = time.perf_counter() - start_all

    total_tokens = sum(token_counts)
    print("\n=== Knat LLM — Multi-GPU Benchmark ===")
    print(f"Model:            {args.model}")
    print(f"Requests:         {args.num_requests} (concurrency={args.concurrency})")
    print(f"Wall time:        {total_time:.2f}s")
    print(f"Throughput:       {total_tokens / total_time:.2f} tok/s")
    print(f"Latency p50:      {statistics.median(latencies):.2f}s")
    print(f"Latency p95:      {sorted(latencies)[int(len(latencies)*0.95)-1]:.2f}s")
    print(
        "\nTo make this a fair comparison against a single-GPU baseline (Project 2), "
        "benchmark a model that ALSO fits on one GPU under both configurations, or "
        "clearly label this result as 'enables serving a model that would not fit at "
        "all on a single GPU' rather than a like-for-like speed comparison."
    )


if __name__ == "__main__":
    main()
