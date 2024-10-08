import logging
import os
import re
import time

from typing import List, Dict, Any

from llindex.crawler import FileEntryList
from llindex.chunk_ctx import ChunkContext, DirContext

# We need these as we look them up dynamically
from llindex.groq import GroqClient
from llindex.local_llm import LocalClient
from llindex.mistral import MistralClient

dir_index_prompt="""
Your job is to summarize the content of a single directory in a code repository.

You will be given summaries for each file and directory which are direct children of the directory you are processing. It will be formatted as a list of entries like this:

<file>
<path>path/filename</path>
<summary>
Summary here...
</summary>
</file>
<file>
<path>path/filename</path>
<summary>
Summary here...
</summary>
</file>
<dir>
<path>path/dirname</path>
<summary>
Summary here...
</summary>
</dir>
...

Your summary should be detailed, contain both high level description and every important detail. Include relationships between files, directories and modules if you have identified them.

Write output in the following format:

<dir>
<path>path/dirname</path>
<summary>
Summary here...
</summary>
</dir>

Output only the XML above, avoid adding extra text.

===========================================================

"""

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

def parse_dir_results(content):
    pattern = re.compile(r'<dir>\s*<path>(.*?)</path>\s*<summary>(.*?)</summary>\s*</dir>', re.DOTALL)
    matches = pattern.findall(content)
    
    result = {path.strip(): summary.strip() for path, summary in matches}
    
    return result

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

def llm_summarize_files(chunk_context: ChunkContext):
    chunk_context.message = format_message(chunk_context.directory, chunk_context.files)
    start = time.time()
    reply = chunk_context.client.query(chunk_context)
    duration = time.time() - start
    logging.info(f'LLM client query took {duration:.3f} seconds.')
    chunk_context.metadata['llm_duration'] = duration
    if reply is not None:
        return parse_results(reply)
    return {}

def client_factory(config):
    class_name = config.pop('type')
    cls = globals()[class_name]
    return cls(**config)

def llm_summarize_dir(dir_path: str, child_summaries: List[str], context: DirContext):
    context.message = dir_index_prompt + '\n'.join(child_summaries)
    start = time.time()
    reply = context.client.query(context)
    duration = time.time() - start
    logging.info(f'LLM client query took {duration:.3f} seconds.')
    context.metadata['llm_duration'] = duration
    logging.info(reply)
    if reply is not None:
        return parse_dir_results(reply)
    return {}

    
