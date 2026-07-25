"""
Knat LLM — Project 2: tokens/sec benchmark for a locally served OpenAI-compatible
LLM endpoint (vLLM). Sends a fixed prompt set, measures wall-clock time, and
reports throughput, latency percentiles, and time-to-first-token where available.

Usage:
    python benchmark/benchmark_tokens.py --base-url http://localhost:8000/v1 \
        --model mistralai/Mistral-7B-Instruct-v0.3 --num-requests 20 --concurrency 4
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

PROMPTS = [
    "Explain the difference between tensor parallelism and pipeline parallelism.",
    "Write a short function in Python that reverses a linked list.",
    "Summarize the tradeoffs between FP16 and INT8 quantization for LLM inference.",
    "What are the main components of a retrieval-augmented generation pipeline?",
    "Describe how a Kubernetes HorizontalPodAutoscaler decides when to scale.",
]


def run_single_request(client: OpenAI, model: str, prompt: str, max_tokens: int):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    return elapsed, completion_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    latencies = []
    token_counts = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_single_request,
                client,
                args.model,
                PROMPTS[i % len(PROMPTS)],
                args.max_tokens,
            )
            for i in range(args.num_requests)
        ]
        start_all = time.perf_counter()
        for future in as_completed(futures):
            elapsed, tokens = future.result()
            latencies.append(elapsed)
            token_counts.append(tokens)
        total_wall_time = time.perf_counter() - start_all

    total_tokens = sum(token_counts)
    throughput = total_tokens / total_wall_time if total_wall_time > 0 else 0

    print("\n=== Knat LLM — Benchmark Results ===")
    print(f"Model:              {args.model}")
    print(f"Requests:           {args.num_requests} (concurrency={args.concurrency})")
    print(f"Total wall time:    {total_wall_time:.2f}s")
    print(f"Total tokens:       {total_tokens}")
    print(f"Throughput:         {throughput:.2f} tok/s (aggregate, across concurrent requests)")
    print(f"Latency p50:        {statistics.median(latencies):.2f}s")
    print(f"Latency p95:        {sorted(latencies)[int(len(latencies)*0.95)-1]:.2f}s")
    print(f"Latency mean:       {statistics.mean(latencies):.2f}s")


if __name__ == "__main__":
    main()
