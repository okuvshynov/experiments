import json
import time
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

from datetime import datetime
from typing import List

from llindex.llm_client import llm_summarize_files, client_factory, llm_summarize_dir
from llindex.crawler import Crawler, FileEntryList
from llindex.config import open_yaml
from llindex.token_counters import token_counter_factory
from llindex.chunks import chunk_tasks
from llindex.chunk_ctx import ChunkContext, DirContext

class Indexer:
    def __init__(self, config):
        self.client = client_factory(config['llm_client'])
        self.token_counter = token_counter_factory(config['token_counter'])
        self.chunk_size = config['chunk_size']
        self.directory = os.path.expanduser(config['dir'])
        self.index_file = os.path.expanduser(config['index_file'])
        self.crawler = Crawler(self.directory, config['crawler'])
        self.freq = config.get('freq', 60)

    def create_directory_structure(self, file_metadata):
        directory_structure = defaultdict(lambda: (set(), set()))
        
        for file_path in file_metadata.keys():
            dir_path = os.path.dirname(file_path)
            
            # Add file to its immediate parent directory
            directory_structure[dir_path][0].add(file_path)
            
            # Add all parent directories to the structure
            while dir_path:
                parent_dir = os.path.dirname(dir_path)
                dir_name = os.path.basename(dir_path)
                
                if dir_name:
                    directory_structure[parent_dir][1].add(dir_path)
                
                dir_path = parent_dir
        
        # Convert sets to sorted lists and defaultdict to regular dict
        return {
            dir_path: (sorted(files), sorted(subdirs))
            for dir_path, (files, subdirs) in directory_structure.items()
        }

    def process_files(self, chunk_context: ChunkContext, current_index) -> List[str]:
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

    def process_directory(self, directory, dir_struct, file_index, dir_index):
        child_files, child_dirs = dir_struct[directory]

        summaries = []

        for child_dir in child_dirs:
            if child_dir not in dir_index:
                self.process_directory(child_dir, dir_struct, file_index, dir_index)
            summaries.append(f'<dir>{child_dir}<summary>{dir_index[child_dir]}</summary></dir>')

        for child_file in child_files:
            if 'processing_result' in file_index[child_file]:
                summaries.append(f'<file>{child_file}<summary>{file_index[child_file]["processing_result"]}</summary></file>')
        context = DirContext(directory)
        context.metadata['model'] = self.client.model_id()
        context.client = self.client
        context.token_counter = self.token_counter
        dir_index[directory] = llm_summarize_dir(directory, summaries, context)

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
            self.process_files(chunk_context, results)
            save_to_json_file(results, self.index_file)
            logging.info(f'processed files: {chunk_context.files}')
            logging.info(f'files missing: {chunk_context.missing_files}')
            logging.info(f'metrics: {chunk_context.metadata}')
        
        ## now aggregate directories
        dir_tree = self.create_directory_structure(results)
        to_process = dir_tree.keys()
        dir_index = {}

        while len(dir_index) < len(to_process):
            for directory in to_process:
                if directory in dir_index:
                    continue
                self.process_directory(directory, dir_tree, results, dir_index)

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
