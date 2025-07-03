#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
import glob
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from verify_diff import verify_diff, verify_diff_detailed

# Default sampling parameters
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 20

def run_llama_cpp(model_path: str, prompt_file: str, output_file: str, 
                  temp: float = DEFAULT_TEMP, top_p: float = DEFAULT_TOP_P,
                  min_p: float = DEFAULT_MIN_P, top_k: int = DEFAULT_TOP_K) -> bool:
    """Run llama.cpp with the given model and prompt."""
    cmd = [
        os.path.expanduser("~/projects/llama.cpp/build/bin/llama-cli"),
        "-m", model_path,
        "-f", prompt_file,
        "--temp", str(temp),
        "--min-p", str(min_p),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "-c", "65536",
        "--no-display-prompt",
        "--single-turn"
    ]
    
    try:
        with open(output_file, 'w') as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running llama.cpp: {e}")
        return False

def run_mlx(model_path: str, prompt_content: str, output_file: str, 
            max_tokens: int = 8192, temp: float = DEFAULT_TEMP, 
            top_p: float = DEFAULT_TOP_P, min_p: float = DEFAULT_MIN_P,
            top_k: int = DEFAULT_TOP_K) -> bool:
    """Run MLX with the given model and prompt."""
    cmd = [
        "mlx_lm.generate",
        "--model", model_path,
        "-m", str(max_tokens),
        "--temp", str(temp),
        "--top-p", str(top_p),
        "--min-p", str(min_p),
        "--top-k", str(top_k),
        "-p", "-"
    ]
    
    try:
        with open(output_file, 'w') as out:
            result = subprocess.run(cmd, input=prompt_content, stdout=out, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running MLX: {e}")
        return False

def process_single_file(typo_file: str, model_path: str, backend: str = "llama.cpp", 
                       base_file: str = "argh/argh.h", max_tokens: int = 8192,
                       temp: float = DEFAULT_TEMP, top_p: float = DEFAULT_TOP_P,
                       min_p: float = DEFAULT_MIN_P, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Process a single typo file and return results."""
    result = {
        "file": typo_file,
        "success": False,
        "error": None,
        "backend": backend
    }
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_diff.txt', delete=False) as diff_file:
            diff_output = diff_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_prompt.txt', delete=False) as prompt_file:
            prompt_path = prompt_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_output.txt', delete=False) as output_file:
            llm_output = output_file.name
        
        # Generate diff
        diff_cmd = ["diff", typo_file, base_file]
        with open(diff_output, 'w') as f:
            subprocess.run(diff_cmd, stdout=f, stderr=subprocess.PIPE)
        
        # Create prompt
        with open("prompt_header.txt", 'r') as header:
            prompt_content = header.read()
        
        with open(typo_file, 'r') as typo:
            prompt_content += typo.read()
        
        # Run the appropriate backend
        if backend == "llama.cpp":
            with open(prompt_path, 'w') as prompt:
                prompt.write(prompt_content)
            
            if not run_llama_cpp(model_path, prompt_path, llm_output, temp, top_p, min_p, top_k):
                result["error"] = "Failed to run llama.cpp"
                return result
        elif backend == "mlx":
            if not run_mlx(model_path, prompt_content, llm_output, max_tokens, temp, top_p, min_p, top_k):
                result["error"] = "Failed to run MLX"
                return result
        else:
            result["error"] = f"Unknown backend: {backend}"
            return result
        
        # Verify result
        verification = verify_diff_detailed(diff_output, llm_output)
        result["success"] = verification["success"]
        result["verification_details"] = verification
        
    except Exception as e:
        result["error"] = str(e)
    finally:
        # Clean up temp files
        for temp_file in [diff_output, prompt_path, llm_output]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Generalized benchmark runner for typo detection")
    parser.add_argument("--model", "-m", required=True, help="Path to the model file or model name for MLX")
    parser.add_argument("--backend", choices=["llama.cpp", "mlx"], default="llama.cpp", help="Backend to use (default: llama.cpp)")
    parser.add_argument("--input", "-i", help="Single input file to process (if not specified, runs all typo files)")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Number of times to repeat each test")
    parser.add_argument("--base-file", "-b", default="argh/argh.h", help="Base file to diff against")
    parser.add_argument("--output-json", "-o", help="Output results to JSON file (default: auto-generated temp file)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum tokens for MLX generation (default: 8192)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p")
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k")
    
    args = parser.parse_args()
    
    # Handle model path based on backend
    model_path = args.model
    if args.backend == "llama.cpp":
        model_path = os.path.expanduser(model_path)
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
    
    # Determine input files
    if args.input:
        input_files = [args.input]
    else:
        input_files = sorted(glob.glob("argh/typos/*.h"))
    
    if not input_files:
        print("Error: No input files found")
        sys.exit(1)
    
    # Set up JSON output file
    if args.output_json:
        json_output_file = args.output_json
    else:
        # Create temp file for JSON output
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_output_file = f"benchmark_results_{timestamp}.json"
    
    print(f"Backend: {args.backend} | Model: {os.path.basename(model_path)}")
    print(f"Files: {len(input_files)} | Repeats: {args.repeat} | Output: {json_output_file}")
    print("-" * 60)
    
    all_results = []
    total_runs = 0
    successful_runs = 0
    
    for input_file in input_files:
        file_results = []
        file_successes = 0
        
        print(f"\n{os.path.basename(input_file)}:", end='')
        
        for run_num in range(args.repeat):
            result = process_single_file(input_file, model_path, args.backend, args.base_file, 
                                       args.max_tokens, args.temp, args.top_p, args.min_p, args.top_k)
            result["run"] = run_num + 1
            all_results.append(result)
            file_results.append(result)
            
            total_runs += 1
            if result["success"]:
                successful_runs += 1
                file_successes += 1
                print(" ✓", end='', flush=True)
            else:
                print(" ✗", end='', flush=True)
        
        # Show per-file stats
        success_rate = file_successes / args.repeat * 100
        print(f" → {file_successes}/{args.repeat} ({success_rate:.0f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Overall: {successful_runs}/{total_runs} tests passed ({successful_runs/total_runs*100:.1f}%)")
    
    # Always save results to JSON
    results_data = {
        "backend": args.backend,
        "model": model_path,
        "base_file": args.base_file,
        "sampling": {
            "temp": args.temp,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k
        },
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
        "results": all_results
    }
    if args.backend == "mlx":
        results_data["max_tokens"] = args.max_tokens
    
    with open(json_output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Detailed results saved to: {json_output_file}")

if __name__ == "__main__":
    main()