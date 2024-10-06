import sys
import json
import tiktoken
from collections import defaultdict
from pathlib import Path

def token_counter_claude(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text, disallowed_special=())
    return len(tokens)

def aggregate_by_directory(file_dict):
    dir_stats = defaultdict(lambda: [0, 0])
    
    for file_path, v in file_dict.items():
        path = Path(file_path)
        for parent in path.parents:
            dir_stats[str(parent) + '/'][0] += 1
            if 'processing_result' in v:
                dir_stats[str(parent) + '/'][1] += 1
    
    return {dir_path: tuple(stats) for dir_path, stats in dir_stats.items()}

def main():
    index_file = sys.argv[1]
    with open(index_file, 'r') as f:
        data = json.load(f)

    tokens = token_counter_claude(json.dumps(data))
    print(f'index of size {len(data)} entries, with approximately {tokens} tokens')
    processed_tokens = 0
    total_tokens = 0
    for k, v in data.items():
        tokens = v['approx_tokens']
        if 'processing_result' in v:
            processed_tokens += tokens
        total_tokens += tokens

    print(f'Total approximate tokens in all files: {total_tokens}')
    print(f'Total approximate tokens in processed files: {processed_tokens}')

    completed = {k: v for k, v in data.items() if 'processing_result' in v}
    print(f'completed {len(completed)} entries')
    if len(sys.argv) > 2:
        l = int(sys.argv[2])
        for k, v in completed.items():
            print('---------')
            print(f'Completed file {k}')
            print(f'Summary: {v["processing_result"][:l]}')

    # dir stats:
    dir_stats = aggregate_by_directory(data)
    print('fully completed directories')
    fully_completed_directories = 0
    total_directories = 0
    partially_completed_directories = 0
    for k, v in dir_stats.items():
        if v[1] == v[0]:
            fully_completed_directories += 1
        elif v[1] > 0:
            partially_completed_directories += 1
        total_directories += 1
    print('Dir stats:')
    print(f'Fully completed directories: {fully_completed_directories}')
    print(f'Partially completed directories: {partially_completed_directories}')
    print(f'total dirs: {total_directories}')

if __name__ == '__main__':
    main()
