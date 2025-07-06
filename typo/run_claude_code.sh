diff "$1" data/argh/argh.h > /tmp/diff_output_cc

cat "$1" | claude -p "$(cat prompt_header.txt)" > /tmp/llm_output_cc

python3 verify_diff.py /tmp/diff_output_cc /tmp/llm_output_cc

