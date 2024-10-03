import json
import re
import os
import requests
import logging
import time
import xml.etree.ElementTree as ET

from typing import List, Dict, Any, Tuple
from xml.dom import minidom

from llindex.token_counters import token_counter_claude
from llindex.crawler import FileEntry, Index, FileEntryList

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

For every file in the input, write output in the following format:

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
<index>2</index>
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

Make sure you processed all files and kept original index for each file.

===========================================================

"""

def parse_results(content):
    pattern = re.compile(r'<file>\s*<index>.*?</index>\s*<path>(.*?)</path>\s*<summary>(.*?)</summary>\s*</file>', re.DOTALL)
    matches = pattern.findall(content)
    
    result = {path.strip(): summary.strip() for path, summary in matches}
    
    return result

def format_file(relative_path, root, index):
    file_element = ET.Element("file")
    
    index_element = ET.SubElement(file_element, "index")
    index_element.text = str(index)
    
    name_element = ET.SubElement(file_element, "path")
    name_element.text = str(relative_path)
    
    content_element = ET.SubElement(file_element, "content")
    
    try:
        with open(os.path.join(root, relative_path), 'r', encoding='utf-8') as file:
            content_element.text = file.read()
    except Exception as e:
        content_element.text = f"Error reading file: {str(e)}"
    
    xml_string = minidom.parseString(ET.tostring(file_element)).toprettyxml(indent="  ")
    return xml_string

def format_message(root: str, files: List[Dict[str, Any]]) -> str:
    file_results = [format_file(f['path'], root, i) for i, f in enumerate(files)]
    return index_prompt + ''.join(file_results)

def llm_summarize_files(root: str, files: FileEntryList, client):
    message = format_message(root, files)
    reply = client.query(message)
    if reply is not None:
        return parse_results(reply)
    return None

