#!/usr/bin/env python3
"""
Tinker Training Cost Calculator

Tokenizes a JSONL training file using the correct model tokenizer
and calculates the estimated training cost.

Usage:
    python calculate_cost.py <jsonl_file> --model <model_name> [--epochs <n>]

Example:
    python calculate_cost.py training_data.jsonl --model Qwen3-8B --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers library required. Install with: pip install transformers")
    sys.exit(1)


# Tinker pricing (Train cost per million tokens) - As of January 5, 2026
# Source: https://thinkingmachines.ai/tinker/
TRAIN_PRICING = {
    "Qwen3-4B-Instruct-2507": 0.22,
    "Qwen3-8B": 0.40,
    "Qwen3-30B-A3B": 0.36,
    "Qwen3-VL-30B-A3B-Instruct": 0.53,
    "Qwen3-32B": 1.47,
    "Qwen3-235B-Instruct-2507": 2.04,
    "Qwen3-VL-235B-A22B-Instruct": 3.07,
    "Llama-3.2-1B": 0.09,
    "Llama-3.2-3B": 0.18,
    "Llama-3.1-8B": 0.40,
    "Llama-3.1-70B": 3.16,
    "DeepSeek-V3.1": 3.38,
    "GPT-OSS-120B": 0.52,
    "GPT-OSS-20B": 0.36,
    "Kimi-K2-Thinking": 2.93,
}

# Model to HuggingFace tokenizer mapping
MODEL_TOKENIZERS = {
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-235B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct",
    "Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-235B-A22B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
    "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3",
    "GPT-OSS-120B": "Qwen/Qwen3-8B",
    "GPT-OSS-20B": "Qwen/Qwen3-8B",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Instruct",
}


def extract_text_from_example(example: dict) -> str:
    """Extract text content from a training example in various formats."""
    # Chat format (messages array)
    if "messages" in example:
        parts = []
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    # Simple text format
    if "text" in example:
        return example["text"]

    # Instruction format (Alpaca-style)
    if any(k in example for k in ["instruction", "input", "output"]):
        parts = []
        if "instruction" in example:
            parts.append(example["instruction"])
        if "input" in example and example["input"]:
            parts.append(example["input"])
        if "output" in example:
            parts.append(example["output"])
        return " ".join(parts)

    # Fallback: stringify the entire example
    return json.dumps(example)


def count_tokens_in_jsonl(file_path: Path, tokenizer) -> dict:
    """Count tokens in a JSONL file using the provided tokenizer."""
    total_tokens = 0
    num_examples = 0
    errors = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                text = extract_text_from_example(example)
                tokens = len(tokenizer.encode(text))
                total_tokens += tokens
                num_examples += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                errors += 1
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                errors += 1

    return {
        "total_tokens": total_tokens,
        "num_examples": num_examples,
        "errors": errors,
        "avg_tokens_per_example": total_tokens / num_examples if num_examples > 0 else 0,
    }


def calculate_cost(total_tokens: int, epochs: int, price_per_million: float) -> float:
    """Calculate training cost in USD."""
    training_tokens = total_tokens * epochs
    return (training_tokens * price_per_million) / 1_000_000


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Tinker training cost for a JSONL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python calculate_cost.py data.jsonl --model Qwen3-8B
    python calculate_cost.py data.jsonl --model Llama-3.1-70B --epochs 5
    python calculate_cost.py data.jsonl --list-models
        """,
    )
    parser.add_argument("file", nargs="?", help="Path to JSONL training file")
    parser.add_argument("--model", "-m", help="Model name (see --list-models)")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models and exit")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("\nAvailable Models (sorted by training cost):\n")
        print(f"{'Model':<35} {'Train $/M':<12} {'Tokenizer'}")
        print("-" * 80)
        for model, price in sorted(TRAIN_PRICING.items(), key=lambda x: x[1]):
            tokenizer = MODEL_TOKENIZERS.get(model, "N/A")
            print(f"{model:<35} ${price:<11.2f} {tokenizer}")
        print(f"\nPricing as of January 5, 2026")
        print(f"Source: https://thinkingmachines.ai/tinker/")
        return

    # Validate arguments
    if not args.file:
        parser.error("JSONL file path is required (or use --list-models)")

    if not args.model:
        parser.error("--model is required. Use --list-models to see available models.")

    if args.model not in TRAIN_PRICING:
        print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
        print(f"Use --list-models to see available models.", file=sys.stderr)
        sys.exit(1)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    tokenizer_name = MODEL_TOKENIZERS[args.model]
    print(f"Loading tokenizer: {tokenizer_name}...", file=sys.stderr)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        print(f"\nNote: Some tokenizers require authentication. Try:", file=sys.stderr)
        print(f"  huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    # Count tokens
    print(f"Counting tokens in: {file_path}...", file=sys.stderr)
    stats = count_tokens_in_jsonl(file_path, tokenizer)

    # Calculate cost
    price_per_million = TRAIN_PRICING[args.model]
    training_tokens = stats["total_tokens"] * args.epochs
    cost = calculate_cost(stats["total_tokens"], args.epochs, price_per_million)

    # Prepare results
    results = {
        "file": str(file_path),
        "model": args.model,
        "tokenizer": tokenizer_name,
        "epochs": args.epochs,
        "num_examples": stats["num_examples"],
        "dataset_tokens": stats["total_tokens"],
        "avg_tokens_per_example": round(stats["avg_tokens_per_example"], 1),
        "training_tokens": training_tokens,
        "price_per_million_usd": price_per_million,
        "estimated_cost_usd": round(cost, 4),
        "errors": stats["errors"],
    }

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Tinker Training Cost Estimate")
        print(f"{'='*50}")
        print(f"File:                  {file_path}")
        print(f"Model:                 {args.model}")
        print(f"Tokenizer:             {tokenizer_name}")
        print(f"{'='*50}")
        print(f"Examples:              {stats['num_examples']:,}")
        print(f"Dataset tokens:        {stats['total_tokens']:,}")
        print(f"Avg tokens/example:    {stats['avg_tokens_per_example']:,.1f}")
        print(f"{'='*50}")
        print(f"Epochs:                {args.epochs}")
        print(f"Training tokens:       {training_tokens:,}")
        print(f"Price per M tokens:    ${price_per_million:.2f}")
        print(f"{'='*50}")
        print(f"ESTIMATED COST:        ${cost:.4f}")
        print(f"{'='*50}")
        if stats["errors"] > 0:
            print(f"\nWarning: {stats['errors']} lines had errors and were skipped.")
        print(f"\nPricing as of January 5, 2026")
        print(f"Source: https://thinkingmachines.ai/tinker/")


if __name__ == "__main__":
    main()
