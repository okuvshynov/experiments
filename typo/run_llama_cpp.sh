diff "$1" argh/argh.h > /tmp/diff_output
cat prompt_header.txt "$1" > /tmp/typo_prompt.txt
~/projects/llama.cpp/build/bin/llama-cli -m ~/projects/llms/gguf/qwen30b3a/Qwen3-30B-A3B-UD-Q8_K_XL.gguf -f /tmp/typo_prompt.txt --temp 0.6 --min-p 0.0 --top-k 20 --top-p 0.95 -c 65536 --no-display-prompt --single-turn > /tmp/llm_output

python3 verify_diff.py /tmp/diff_output /tmp/llm_output

