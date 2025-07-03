#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
import glob
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from verify_diff import verify_diff

def run_llama_cpp(model_path: str, prompt_file: str, output_file: str) -> bool:
    """Run llama.cpp with the given model and prompt."""
    cmd = [
        os.path.expanduser("~/projects/llama.cpp/build/bin/llama-cli"),
        "-m", model_path,
        "-f", prompt_file,
        "--temp", "0.6",
        "--min-p", "0.0",
        "--top-k", "20",
        "--top-p", "0.95",
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

def process_single_file(typo_file: str, model_path: str, base_file: str = "argh/argh.h") -> Dict[str, Any]:
    """Process a single typo file and return results."""
    result = {
        "file": typo_file,
        "success": False,
        "error": None
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
        
        with open(prompt_path, 'w') as prompt:
            prompt.write(prompt_content)
        
        # Run llama.cpp
        if not run_llama_cpp(model_path, prompt_path, llm_output):
            result["error"] = "Failed to run llama.cpp"
            return result
        
        # Verify result
        result["success"] = verify_diff(diff_output, llm_output)
        
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
    parser.add_argument("--model", "-m", required=True, help="Path to the model file")
    parser.add_argument("--input", "-i", help="Single input file to process (if not specified, runs all typo files)")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Number of times to repeat each test")
    parser.add_argument("--base-file", "-b", default="argh/argh.h", help="Base file to diff against")
    parser.add_argument("--output-json", "-o", help="Output results to JSON file")
    
    args = parser.parse_args()
    
    # Expand model path
    model_path = os.path.expanduser(args.model)
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
    
    print(f"Running benchmark with model: {model_path}")
    print(f"Processing {len(input_files)} file(s), {args.repeat} repetition(s) each")
    print("-" * 80)
    
    all_results = []
    total_runs = 0
    successful_runs = 0
    
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        
        for run_num in range(args.repeat):
            if args.repeat > 1:
                print(f"  Run {run_num + 1}/{args.repeat}...", end=' ')
            
            result = process_single_file(input_file, model_path, args.base_file)
            result["run"] = run_num + 1
            all_results.append(result)
            
            total_runs += 1
            if result["success"]:
                successful_runs += 1
                print("PASSED" if args.repeat > 1 else "  Result: PASSED")
            else:
                print("FAILED" if args.repeat > 1 else "  Result: FAILED")
                if result["error"]:
                    print(f"    Error: {result['error']}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {successful_runs}/{total_runs} tests passed ({successful_runs/total_runs*100:.1f}%)")
    
    # Save results to JSON if requested
    if args.output_json:
        results_data = {
            "model": model_path,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "results": all_results
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

if __name__ == "__main__":
    main()