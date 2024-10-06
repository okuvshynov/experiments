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
    print(f'index of size {len(data)} entries, approximately {tokens} tokens')
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
