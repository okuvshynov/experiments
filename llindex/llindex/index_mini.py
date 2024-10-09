import sys
import json
import tiktoken

from collections import defaultdict
from pathlib import Path

def token_counter_claude(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text, disallowed_special=())
    return len(tokens)

def main():
    index_file = sys.argv[1]
    with open(index_file, 'r') as f:
        data = json.load(f)

    files_data, dir_data = data['files'], data['dirs']
    all_files = '\n'.join(files_data.keys())
    all_dirs = '\n'.join(dir_data.keys())
    dir_summaries = '\n'.join(d['processing_result'] for d in dir_data.values() if 'processing_result' in d)
    print(token_counter_claude(all_dirs + all_files))
    print(token_counter_claude(all_dirs))
    print(token_counter_claude(dir_summaries))


if __name__ == '__main__':
    main()
