"""
Knat LLM — Project 6: produces an AWQ-quantized checkpoint from a base model,
to be served by a separate vLLM instance for the benchmark comparison.

Usage:
    python benchmark/quantize_awq.py --model-path mistralai/Mistral-7B-Instruct-v0.3 \
        --output-path ./models/mistral-7b-awq
"""
import argparse

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--calib-samples", type=int, default=128,
                         help="Number of calibration samples used to compute activation-aware scales")
    args = parser.parse_args()

    print(f"Loading base model: {args.model_path}")
    model = AutoAWQForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Running AWQ quantization (this uses a small calibration set to pick which "
          "activations matter most, rather than quantizing everything uniformly)...")
    model.quantize(tokenizer, quant_config=QUANT_CONFIG)

    model.save_quantized(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"AWQ-quantized model saved to {args.output_path}")
    print("Serve it with vLLM using: --quantization awq --model", args.output_path)


if __name__ == "__main__":
    main()
