#!/usr/bin/env python3
"""
MLX-LM Benchmarking Script
Measures tokens per second for prompt processing and token generation
"""

import argparse
import csv
import sys
import time
from typing import List, Dict
import statistics

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import wired_limit

def create_random_prompt(tokenizer, prompt_length: int) -> mx.array:
    """Create a random prompt of specified length using random tokens from vocabulary."""
    # Use a subset of vocabulary to avoid special tokens
    vocab_size = min(tokenizer.vocab_size, 32000)  # Cap at 32k for safety
    # Generate random token IDs, avoiding first few which are often special
    random_ids = mx.random.randint(100, vocab_size, (prompt_length,))
    return random_ids

def benchmark_single_run(
    model: nn.Module,
    tokenizer,
    prompt_length: int,
    num_tokens: int,
    prefill_step_size: int = 2048
) -> Dict[str, float]:
    """Run a single benchmark and return timing results."""
    
    # Create random prompt
    prompt = create_random_prompt(tokenizer, prompt_length)
    
    # Create KV cache
    prompt_cache = make_prompt_cache(model)
    
    # Create a generation stream (similar to original code)
    generation_stream = mx.new_stream(mx.default_device())
    
    # Measure prompt processing time
    with wired_limit(model, [generation_stream]):
        prompt_start = time.perf_counter()
        
        # Process prompt in chunks (similar to original generate_step)
        y = prompt
        while y.shape[0] > prefill_step_size:
            with mx.stream(generation_stream):
                model(y[:prefill_step_size][None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
        
        # Process remaining prompt
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            mx.eval(logits)
        
        prompt_end = time.perf_counter()
        prompt_time = prompt_end - prompt_start
        prompt_tps = prompt_length / prompt_time
        
        # Generate tokens
        generation_start = time.perf_counter()
        
        for i in range(num_tokens):
            with mx.stream(generation_stream):
                # Generate next token (using last token as input)
                logits = model(mx.array([0])[None], cache=prompt_cache)  # Dummy token
                mx.eval(logits)
            
            # Clear cache periodically to prevent memory issues
            if i % 256 == 0:
                mx.clear_cache()
        
        generation_end = time.perf_counter()
        generation_time = generation_end - generation_start
        generation_tps = num_tokens / generation_time if generation_time > 0 else 0
        
        # Get peak memory usage
        peak_memory_gb = mx.get_peak_memory() / 1e9
        
    return {
        'prompt_tokens': prompt_length,
        'prompt_time': prompt_time,
        'prompt_tps': prompt_tps,
        'generation_tokens': num_tokens,
        'generation_time': generation_time,
        'generation_tps': generation_tps,
        'peak_memory_gb': peak_memory_gb,
        'total_time': prompt_time + generation_time
    }

def run_benchmarks(
    model_path: str,
    prompt_length: int,
    num_tokens: int,
    num_runs: int,
    output_file: str = None
) -> None:
    """Run multiple benchmarks and output results in CSV format."""
    
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    print(f"\nRunning {num_runs} benchmark runs:")
    print(f"- Prompt length: {prompt_length} tokens")
    print(f"- Generation length: {num_tokens} tokens")
    print("-" * 50)
    
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...", end='', flush=True)
        
        result = benchmark_single_run(
            model=model,
            tokenizer=tokenizer,
            prompt_length=prompt_length,
            num_tokens=num_tokens
        )
        
        result['run'] = run + 1
        results.append(result)
        
        print(f" Done! Prompt: {result['prompt_tps']:.1f} t/s, "
              f"Generation: {result['generation_tps']:.1f} t/s")
    
    # Calculate statistics
    prompt_tps_values = [r['prompt_tps'] for r in results]
    gen_tps_values = [r['generation_tps'] for r in results]
    
    stats = {
        'prompt_tps_mean': statistics.mean(prompt_tps_values),
        'prompt_tps_std': statistics.stdev(prompt_tps_values) if len(prompt_tps_values) > 1 else 0,
        'prompt_tps_min': min(prompt_tps_values),
        'prompt_tps_max': max(prompt_tps_values),
        'generation_tps_mean': statistics.mean(gen_tps_values),
        'generation_tps_std': statistics.stdev(gen_tps_values) if len(gen_tps_values) > 1 else 0,
        'generation_tps_min': min(gen_tps_values),
        'generation_tps_max': max(gen_tps_values),
    }
    
    # Output results
    output = output_file if output_file else sys.stdout
    
    if output == sys.stdout:
        print("\n" + "=" * 50)
        print("RESULTS (CSV Format):")
        print("=" * 50)
    
    # Write detailed results
    fieldnames = ['run', 'prompt_tokens', 'prompt_time', 'prompt_tps', 
                  'generation_tokens', 'generation_time', 'generation_tps', 
                  'peak_memory_gb', 'total_time']
    
    if output == sys.stdout:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        
        # Print summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS:")
        print("=" * 50)
        print(f"Prompt TPS - Mean: {stats['prompt_tps_mean']:.2f}, "
              f"Std: {stats['prompt_tps_std']:.2f}, "
              f"Min: {stats['prompt_tps_min']:.2f}, "
              f"Max: {stats['prompt_tps_max']:.2f}")
        print(f"Generation TPS - Mean: {stats['generation_tps_mean']:.2f}, "
              f"Std: {stats['generation_tps_std']:.2f}, "
              f"Min: {stats['generation_tps_min']:.2f}, "
              f"Max: {stats['generation_tps_max']:.2f}")
    else:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"\nResults saved to: {output_file}")
        
        # Still print summary to console
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS:")
        print("=" * 50)
        print(f"Prompt TPS - Mean: {stats['prompt_tps_mean']:.2f}, "
              f"Std: {stats['prompt_tps_std']:.2f}, "
              f"Min: {stats['prompt_tps_min']:.2f}, "
              f"Max: {stats['prompt_tps_max']:.2f}")
        print(f"Generation TPS - Mean: {stats['generation_tps_mean']:.2f}, "
              f"Std: {stats['generation_tps_std']:.2f}, "
              f"Min: {stats['generation_tps_min']:.2f}, "
              f"Max: {stats['generation_tps_max']:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX-LM models for tokens per second performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-30B-A3B-8bit",
        help="Path to model or HuggingFace repo ID"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Length of the prompt in tokens (default: 512)"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: print to stdout)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Run benchmarks
    run_benchmarks(
        model_path=args.model,
        prompt_length=args.prompt_length,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        output_file=args.output
    )

if __name__ == "__main__":
    main()