import json
import time
import logging
import os
import sys

from datetime import datetime
from typing import List

from llindex.llm_client import llm_summarize_files, client_factory
from llindex.crawler import Crawler, FileEntryList
from llindex.config import open_yaml
from llindex.token_counters import token_counter_factory
from llindex.chunks import chunk_tasks
from llindex.chunk_ctx import ChunkContext

class Indexer:
    def __init__(self, config):
        self.client = client_factory(config['llm_client'])
        self.token_counter = token_counter_factory(config['token_counter'])
        self.chunk_size = config['chunk_size']
        self.directory = os.path.expanduser(config['dir'])
        self.index_file = os.path.expanduser(config['index_file'])
        self.crawler = Crawler(self.directory, config['crawler'])
        self.freq = config.get('freq', 60)

    def process(self, chunk_context: ChunkContext, current_index) -> List[str]:
        logging.info(f'processing {len(chunk_context.files)} files')
        chunk_context.metadata['model'] = self.client.model_id()
        chunk_context.client = self.client
        chunk_context.token_counter = self.token_counter
        result = llm_summarize_files(chunk_context)
        logging.info(f'received {len(result)} summaries')

        timestamp = datetime.now().isoformat()
        for file in chunk_context.files:
            file_result = {
                "path": file["path"],
                "size": file["size"],
                "checksum": file["checksum"],
                "processing_timestamp": timestamp,
                "approx_tokens": file["approx_tokens"] 
            }
            relative_path = file['path']
            if relative_path not in result:
                file_result["skipped"] = True
                chunk_context.missing_files.append(relative_path)
                logging.warning(f'missing file {relative_path} in the reply.')
            else:
                file_result["processing_result"] = result[relative_path]
            current_index[relative_path] = file_result

    def count_tokens(self, files):
        for k, v in files.items():
            with open(os.path.join(self.directory, k), 'r') as f:
                v['approx_tokens'] = self.token_counter(f.read())

    def run(self) -> FileEntryList:
        """Process directory with size limit and return results for files that should be processed."""
        previous_index = load_json_from_file(self.index_file)
        to_process, to_reuse = self.crawler.run(previous_index)
        results = {}
        for file in to_reuse:
            results[file["path"]] = file
        for file in to_process:
            results[file["path"]] = file

        self.count_tokens(results)
        
        save_to_json_file(results, self.index_file)

        logging.info(f'Indexing: {len(to_process)} files')
        logging.info(f'Reusing: {len(to_reuse)} files')
        chunks = chunk_tasks(to_process, self.chunk_size)
        for chunk in chunks:
            chunk_context = ChunkContext(directory=self.directory, files=chunk)
            self.process(chunk_context, results)
            save_to_json_file(results, self.index_file)
            logging.info(f'processed files: {chunk_context.files}')
            logging.info(f'files missing: {chunk_context.missing_files}')
            logging.info(f'metrics: {chunk_context.metadata}')
        
        return results

    def loop(self):
        # naive loop, sleep for N seconds and then process again.
        # if nothing was changed, we won't query model. 
        # still, better to use something watchdog
        while True:
            logging.info('starting next iteration')
            self.run()
            time.sleep(self.freq)

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
    indexer.loop()

if __name__ == '__main__':
    main()
