import json
import logging
import os
import sys

from datetime import datetime
from filelock import FileLock
from typing import List, Dict, Any

from llindex.llm_client import llm_summarize_files, client_factory
from llindex.crawler import Crawler, FileEntry, Index, FileEntryList
from llindex.config import open_yaml
from llindex.token_counters import token_counter_claude
from llindex.chunks import chunk_tasks

class Indexer:
    def __init__(self, config):
        self.client = client_factory(config['llm_client'])
        self.chunk_size = config['chunk_size']
        self.directory = os.path.expanduser(config['dir'])
        self.index_file = os.path.expanduser(config['index_file'])
        self.crawler = Crawler(self.directory, config['crawler'])

    def process(self, directory: str, files: FileEntryList) -> List[str]:
        logging.info(f'processing {len(files)} files')
        result = llm_summarize_files(directory, files, self.client)
        logging.info(f'received {len(result)} summaries')

        res = []
        for file in files:
            relative_path = file['path']
            if relative_path not in result:
                logging.warning(f'missing file {relative_path} in the reply.')
            else:
                res.append(result[relative_path])

        return res


    def run(self, previous_index) -> FileEntryList:
        """Process directory with size limit and return results for files that should be processed."""
        to_process, to_reuse = self.crawler.run(previous_index)
        chunks = chunk_tasks(to_process, self.chunk_size)

        logging.info(f'{self.directory} {previous_index}')
        
        results = {}
        for chunk in chunks:
            processing_results = self.process(self.directory, chunk)
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
        for file in to_reuse:
            if "processing_result" in file:
                results[file["path"]] = file
        
        n_tokens = token_counter_claude(json.dumps(results))
        logging.info(f'computed index size of approximately {n_tokens} tokens')
        return results

    def onepass(self):
        lock = FileLock(f"{self.index_file}.lock", timeout=10)
        with lock:
            current = load_json_from_file(self.index_file)
        new = self.run(current)
        with lock:
            logging.info(new)
            save_to_json_file(new, self.index_file)

def load_json_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.path.dirname(__file__), 'indexer.yaml')
    config = open_yaml(config_path)
    indexer = Indexer(config)
    indexer.onepass()

if __name__ == '__main__':
    main()
