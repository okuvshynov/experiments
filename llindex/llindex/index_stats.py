import sys
import json

def main():
    index_file = sys.argv[1]
    with open(index_file, 'r') as f:
        data = json.load(f)

    print(f'index of size {len(data)}')
    completed = {k: v for k, v in data.items() if 'processing_result' in v}
    print(f'completed {len(completed)} entries')
    if len(sys.argv) > 2:
        for k, v in completed.items():
            print('---------')
            print(f'Completed file {k}')
            print(f'Summary: {v["processing_result"][:512]}')

if __name__ == '__main__':
    main()
