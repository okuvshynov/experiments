diff "$1" argh/argh.h > /tmp/diff_output
cat prompt_header.txt "$1" | mlx_lm.generate --model mlx-community/Qwen3-235B-A22B-4bit-DWQ  -m 8192 -p - > /tmp/llm_output

python3 verify_diff.py /tmp/diff_output /tmp/llm_output

