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

class GroqClient:
    def __init__(self, tokens_rate=20000, period=60):
        # Load API key from environment variable
        self.api_key: str = os.environ.get('GROQ_API_KEY')

        if self.api_key is None:
            logging.error("GROQ_API_KEY environment variable not set")

        # we need that for client-side rate limiting
        self.history: List[Tuple[float, int]] = []
        self.tokens_rate: int = tokens_rate
        self.period: int = period
        self.url = 'https://api.groq.com/openai/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def wait_time(self):
        total_size = sum(size for _, size in self.history)
        if total_size < self.tokens_rate:
            return 0
        current_time = time.time()
        running_total = total_size
        for time_stamp, size in self.history:
            running_total -= size
            if running_total <= self.tokens_rate:
                return max(0, time_stamp + self.period - current_time)

    def query(self, message):
        req = {
            "max_tokens": 4096,
            "model": "llama-3.1-70b-versatile",
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        payload = json.dumps(req)

        payload_size = token_counter_claude(payload)
        if payload_size > self.tokens_rate:
            logging.error(f'unable to send message of {payload_size} tokens. Limit is {self.tokens_rate}')
            return None

        current_time = time.time()
        self.history = [(t, s) for t, s in self.history if current_time - t <= self.period]
        self.history.append((current_time, payload_size))

        wait_for = self.wait_time()

        if wait_for > 0:
            logging.info(f'groq client-side rate-limiting. Waiting for {wait_for} seconds')
            time.sleep(wait_for)

        # Send POST request
        response = requests.post(self.url, headers=self.headers, data=payload)

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"{response.text}")
            return None

        res = response.json()
        content = res['choices'][0]['message']['content']
        return parse_results(content)
