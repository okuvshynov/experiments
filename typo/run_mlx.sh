#!/bin/bash

# Usage: ./run_mlx.sh <typo_file> [base_file]
# If base_file is not provided, it will try to auto-detect it

TYPO_FILE="$1"
BASE_FILE="$2"

if [ -z "$TYPO_FILE" ]; then
    echo "Usage: $0 <typo_file> [base_file]"
    exit 1
fi

# Auto-detect base file if not provided
if [ -z "$BASE_FILE" ]; then
    # Extract project directory from typo file path
    PROJECT_DIR=$(dirname $(dirname "$TYPO_FILE"))
    
    # Look for base file (any .h/.hpp file not in typos directory)
    BASE_FILE=$(find "$PROJECT_DIR" -maxdepth 1 -type f \( -name "*.h" -o -name "*.hpp" \) | head -1)
    
    if [ -z "$BASE_FILE" ]; then
        echo "Error: Could not auto-detect base file. Please provide it as second argument."
        exit 1
    fi
    echo "Auto-detected base file: $BASE_FILE"
fi

diff "$TYPO_FILE" "$BASE_FILE" > /tmp/diff_output
cat prompt_header.txt "$1" > /tmp/typo_prompt.txt
cat prompt_header.txt "$1" | mlx_lm.generate --model mlx-community/Qwen3-235B-A22B-4bit-DWQ  -m 8192 -p - > /tmp/llm_output

python3 verify_diff.py /tmp/diff_output /tmp/llm_output

