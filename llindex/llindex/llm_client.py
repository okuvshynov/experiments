import logging
import os
import re
import time

from typing import List, Dict, Any

from llindex.crawler import FileEntryList

# We need these as we look them up dynamically
from llindex.groq import GroqClient
from llindex.local_llm import LocalClient
from llindex.mistral import MistralClient

index_prompt="""
You will be given content for multiple files from code repository. It will be formatted as a list of entries like this:

<input_file>
<index>1</index>
<path>path/filename</path>
<content>
Content here....
</content>
</input_file>
<input_file>
<index>2</index>
<path>path/filename</path>
<content>
Content here....
</content>
</input_file>
...
<input_file>
<index>N</index>
<path>path/filename</path>
<content>
Content here....
</content>
</input_file>

index is just a number from 1 to N where N is the number of input files.

Your job is to provide a description of each provided file.
Description for each file should be detailed, contain both high level description and every important detail. Include relationships between files if you have identified them.

Write output in the following format:

<files>
<file>
<index>1</index>
<path>path/filename</path>
<summary>
Summary here...
</summary>
</file>
<file>
<index>2</index>
<path>path/filename</path>
<summary>
Summary here...
</summary>
</file>
...
<file>
<index>N</index>
<path>path/filename</path>
<summary>
Summary here...
</summary>
</file>
</files>

Make sure you processed all files and kept original index for each file.
Output only the XML above, avoid adding extra text.

===========================================================

"""

def parse_results(content):
    pattern = re.compile(r'<file>\s*<index>.*?</index>\s*<path>(.*?)</path>\s*<summary>(.*?)</summary>\s*</file>', re.DOTALL)
    matches = pattern.findall(content)
    
    result = {path.strip(): summary.strip() for path, summary in matches}
    
    return result

def format_file(relative_path, root, index):
    res  = f"<file>"
    res += f"<index>{index}</index>"
    res += f"<path>{relative_path}</path>"
    
    try:
        with open(os.path.join(root, relative_path), 'r', encoding='utf-8') as file:
            res += f"<content>{file.read()}</content>"
    except Exception as e:
        logging.error(f'unable to read file: {relative_path}')
        return None
    res += "</file>"
    
    return res

def format_message(root: str, files: List[Dict[str, Any]]) -> str:
    file_results = list(filter(lambda x: x is not None, (format_file(f['path'], root, i) for i, f in enumerate(files))))
    return index_prompt + ''.join(file_results)

def llm_summarize_files(root: str, files: FileEntryList, client):
    message = format_message(root, files)
    start = time.time()
    reply = client.query(message)
    duration = time.time() - start
    logging.info(f'LLM client query took {duration:.3f} seconds.')
    if reply is not None:
        return parse_results(reply)
    return {}

def client_factory(config):
    class_name = config.pop('type')
    cls = globals()[class_name]
    return cls(**config)
