import json
import logging
import os
import requests
import sys
import time

from lucas.index_format import format_default
from lucas.tools.toolset import Toolset
from lucas.utils import merge_by_key

from lucas.clients.mistral import MistralClient

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    if len(sys.argv) < 2:
        logging.error("Error: Please provide the codebase path as a command line argument.")
        sys.exit(1)

    codebase_path = sys.argv[1]

    if not os.path.isdir(codebase_path):
        logging.error(f"Error: The directory '{codebase_path}' does not exist.")
        sys.exit(1)

    index_file = os.path.join(codebase_path, ".llidx")

    if not os.path.isfile(index_file):
        logging.error(f"Error: The index file '{index_file}' does not exist.")
        sys.exit(1)

    with open(index_file, 'r') as f:
        index = json.load(f)

    logging.info('loaded index')
    index_formatted = format_default(index)
    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, 'prompts', 'query_with_tools.txt')) as f:
        prompt = f.read()


    message = sys.argv[2]
    task = f'<task>{message}</task>'
    user_message = prompt + index_formatted + '\n\n' + task

    client = MistralClient()
    toolset = Toolset(codebase_path)

    reply = client.send(user_message, toolset)
    print(reply['message']['content'])

if __name__ == '__main__':
    main()

