import logging
from lltools.get_files import GetFilesTool
import http.client
from llindex.index_format import format_default
import sys
import os
import json

sonnet_prompt="""
You are given a summary of a code repository in the following xml-like format:
<dir>
    <path>...</path>
    <summary>Description of this directory</summary>
    <dirs>
        <dir>...</dir>
        <dir>...</dir>
    </dirs>
    <files>
        <file>file/path/here</file>
        <file>file/path/here</file>
        ...
    </files>
</dir>

Each directory will have a summary, all files will be listed.

You will be given your task in <task></task> tags.

You will have access to a tool to get content of the files you need to accomplish that task. Use the summaries provided to identify the files you need. Feel free to use the tool more than once if you discovered that you need more information.
"""

def interact(user_message):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    conn = http.client.HTTPSConnection("api.anthropic.com")
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    tool = GetFilesTool(sys.argv[1])
    messages = [{"role": "user", "content": user_message}]
    for i in range(5):
        tool_choice = {"type": "auto"} if i > 0 else {"type": "any"}
        payload = json.dumps({
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 8192,
            "tools": [tool.definition()],
            "messages": messages,
            "tool_choice": tool_choice
        })
        logging.info('sending request')
        conn.request("POST", "/v1/messages", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode("utf-8"))
        messages.append({"role": "assistant", "content": data['content']})

        if data["stop_reason"] == "tool_use":
            message = {"role": "user", "content": []}
            for content_piece in data['content']:
                if content_piece['type'] == 'tool_use':
                    logging.info(f'requested tool: {content_piece["input"]}')
                    tool_use_id = content_piece['id']
                    tool_use_name = content_piece['name']
                    tool_use_args = content_piece['input']
                    if tool_use_name != 'get_files':
                        logging.warning(f'unknown tool: {tool_use_name}')
                        continue
                    tool_result = tool.run(tool_use_args)
                    message["content"].append({"type": "tool_result", "tool_use_id" : tool_use_id, "content": tool_result})
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

