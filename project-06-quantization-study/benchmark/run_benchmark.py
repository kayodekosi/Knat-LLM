"""
Knat LLM — Project 6: head-to-head optimization study across FP16, INT8, and AWQ.

This script assumes three already-running OpenAI-compatible endpoints (e.g. three
vLLM containers, one per precision, on different ports) and benchmarks all three
under an identical prompt/load profile so the comparison is apples-to-apples.

Usage:
    python benchmark/run_benchmark.py --config benchmark/endpoints.json
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

EVAL_PROMPTS = [
    "Summarize the key differences between REST and gRPC in three sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the CAP theorem to a junior backend engineer.",
    "List three tradeoffs of using a vector database versus a relational database for search.",
    "Describe what a horizontal pod autoscaler does in Kubernetes.",
]


def run_single(client: OpenAI, model: str, prompt: str, max_tokens: int):
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return elapsed, tokens, resp.choices[0].message.content


def benchmark_endpoint(name, base_url, model, num_requests, concurrency, max_tokens):
    client = OpenAI(base_url=base_url, api_key="not-needed")
    latencies, token_counts, samples = [], [], []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single, client, model, EVAL_PROMPTS[i % len(EVAL_PROMPTS)], max_tokens)
            for i in range(num_requests)
        ]
        start_all = time.perf_counter()
        for future in as_completed(futures):
            elapsed, tokens, text = future.result()
            latencies.append(elapsed)
            token_counts.append(tokens)
            samples.append(text)
        total_time = time.perf_counter() - start_all

    total_tokens = sum(token_counts)
    return {
        "config": name,
        "throughput_tok_s": round(total_tokens / total_time, 2) if total_time else 0,
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[max(int(len(latencies) * 0.95) - 1, 0)], 3),
        "sample_output": samples[0][:200] if samples else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/endpoints.json",
                         help="JSON file listing the endpoints to compare (see endpoints.json.example)")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--out", default="results/benchmark_results.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        endpoints = json.load(f)

    rows = []
    for entry in endpoints:
        print(f"Benchmarking {entry['name']}...")
        rows.append(
            benchmark_endpoint(
                entry["name"], entry["base_url"], entry["model"],
                args.num_requests, args.concurrency, args.max_tokens,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
