import json
import logging
import os
import sys

from datetime import datetime
from filelock import FileLock
from typing import List, Dict, Any

from llindex.llm_client import format_message, GroqClient
from llindex.crawler import process_directory, chunk_files

class Indexer:
    def __init__(self, client):
        self.client = client

    def process(self, directory: str, files: List[Dict[str, Any]]) -> List[str]:
        logging.info(f'processing {len(files)} files')
        message = format_message(directory, files)
        result = self.client.query(message)

        logging.info(result)

        res = []
        for file in files:
            relative_path = file['path']
            if relative_path not in result:
                logging.error(f'missing file {relative_path} in the reply.')
            else:
                res.append(result[relative_path])

        return res


    def run(self, directory: str, size_limit: int, previous_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process directory with size limit and return results for files that should be processed."""
        files = process_directory(directory, directory, previous_index)
        chunks = chunk_files(files, size_limit)
        
        results = {}
        for chunk in chunks:
            processing_results = self.process(directory, chunk)
            timestamp = datetime.now().isoformat()
            for file, result in zip(chunk, processing_results):
                file_result = {
                    "path": file["path"],
                    "size": file["size"],
                    "checksum": file["checksum"],
                    "processing_result": result,
                    "processing_timestamp": timestamp,
                    "approx_tokens": file["approx_tokens"] 
                }
                results[file["path"]] = file_result
        
        # Add previously processed files that weren't reprocessed
        for file in files:
            if "processing_result" in file:
                results[file["path"]] = file
        
        return results

def load_json_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def onepass(indexer, directory, index_file):
    lock = FileLock(f"{index_file}.lock", timeout=10)
    with lock:
        current = load_json_from_file(index_file)
    new = indexer.run(directory, 8000, current)
    with lock:
        logging.info(new)
        save_to_json_file(new, index_file)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    directory = sys.argv[1]
    groq = GroqClient()
    indexer = Indexer(groq)
    onepass(indexer, directory, "/tmp/llindex.0")

if __name__ == '__main__':
    main()
