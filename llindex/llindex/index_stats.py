import sys
import json
import tiktoken

def token_counter_claude(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text, disallowed_special=())
    return len(tokens)

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

if __name__ == '__main__':
    main()
