diff argh/argh.h "$1"
cat prompt_header.txt "$1" | mlx_lm.generate --model mlx-community/Qwen3-30B-A3B-4bit-DWQ-053125  -m 8192 -p -
cat prompt_header.txt "$1" | mlx_lm.generate --model mlx-community/Qwen3-235B-A22B-4bit-DWQ  -m 8192 -p -
