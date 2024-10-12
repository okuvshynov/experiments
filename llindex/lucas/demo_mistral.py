import json
import logging
import os
import requests
import sys
import time

from lucas.index_format import format_default
from lucas.tools.toolset import Toolset
from lucas.utils import merge_by_key

script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir, 'prompts', 'query_with_tools.txt')) as f:
    mistral_prompt = f.read()

def interact(user_message):
    api_key = os.environ.get("MISTRAL_API_KEY")
    url = 'https://api.mistral.ai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    toolset = Toolset(sys.argv[1])

    usage = {} 

    messages = [{"role": "user", "content": user_message}]
    for i in range(10):
        tool_choice = "auto" if i > 0 else "any"
        payload = json.dumps({
            "model": "mistral-large-latest",
            "max_tokens": 8192,
            "tools": toolset.definitions_v0(),
            "messages": messages,
            "tool_choice": tool_choice
        })
        logging.info('sending request')
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()

        usage = merge_by_key(usage, data['usage'])
        logging.info(f'Aggregate usage: {usage}')
        reply = data['choices'][0]

        messages.append(reply['message'])

        if reply["finish_reason"] == "tool_calls":
            for tool_call in reply['message']['tool_calls']:
                args = json.loads(tool_call['function']['arguments'])
                tool_args = {
                    'name': tool_call['function']['name'],
                    'id': tool_call['id'],
                    'input': args
                }
                result = toolset.run(tool_args)
                if result is not None:
                    messages.append({"role":"tool", "name":tool_args['name'], "content":result['content'], "tool_call_id": tool_args['id']})

                else:
                    logging.warning(f'unknown tool: {tool_args}')
                    continue
        else:
            # got final reply
            logging.info('received final reply')
            return reply
    logging.warning('no reply after 5 interactions')
    return None

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

    message = sys.argv[2]
    task = f'<task>{message}</task>'
    user_message = mistral_prompt + index_formatted + '\n\n' + task
    reply = interact(user_message)
    print(reply['message']['content'])


if __name__ == '__main__':
    main()

