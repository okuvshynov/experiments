import http.client
import json
import logging
import os
import sys

from lucas.index_format import format_default
from lucas.tools.toolset import Toolset
from lucas.utils import merge_by_key

script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir, 'prompts', 'query_with_tools.txt')) as f:
    sonnet_prompt = f.read()

def interact(user_message):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    conn = http.client.HTTPSConnection("api.anthropic.com")
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    toolset = Toolset(sys.argv[1])

    usage = {}

    messages = [{"role": "user", "content": user_message}]
    for i in range(10):
        tool_choice = {"type": "auto"} if i > 0 else {"type": "any"}
        payload = json.dumps({
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 8192,
            "tools": toolset.definitions(),
            "messages": messages,
            "tool_choice": tool_choice
        })
        logging.info('sending request')
        conn.request("POST", "/v1/messages", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode("utf-8"))
        usage = merge_by_key(usage, data['usage'])
        logging.info(f'Aggregate usage: {usage}')

        if 'content' not in data:
            logging.error(f'not content in {data}')
            break

        messages.append({"role": "assistant", "content": data['content']})

        if data["stop_reason"] == "tool_use":
            message = {"role": "user", "content": []}
            for content_piece in data['content']:
                if content_piece['type'] == 'tool_use':
                    result = toolset.run(content_piece)
                    if result is not None:
                        message["content"].append(result)
                    else:
                        logging.warning(f'unknown tool: {tool_use_name}')
                        continue
            messages.append(message)
        else:
            # got final reply
            logging.info('received final reply')
            return data['content']
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
    user_message = sonnet_prompt + index_formatted + '\n\n' + task
    reply = interact(user_message)
    if len(reply) == 1 and 'text' in reply[0]:
        print(reply[0]['text'])
    else:
        logging.error(f'Reply: {reply}')


if __name__ == '__main__':
    main()

