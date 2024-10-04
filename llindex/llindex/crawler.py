import os
import logging
import fnmatch

from typing import List, Dict, Any, Tuple
from llindex.file_info import get_file_info

FileEntry = Dict[str, Any]
Index = Dict[str, FileEntry]
FileEntryList = List[FileEntry]


class Crawler:
    def __init__(self, root, conf: Dict[str, any]):
        self.root = root
        if 'includes' in conf:
            includes = conf['includes'].split(',')
            self.includes = [p.strip() for p in includes]
        else:
            self.includes = ["*"]
        if 'excludes' in conf:
            excludes = conf['excludes'].split(',')
            self.excludes = [p.strip() for p in excludes]
        else:
            self.excludes = []

    def should_process(self, path):
        included = any(fnmatch.fnmatch(path, p) for p in self.includes)
        excluded = any(fnmatch.fnmatch(path, p) for p in self.excludes)
        return included and not excluded

    def run(self, prev_index):
        result = []
        reused = []
        for root_path, _, files in os.walk(self.root):
            for file in files:
                full_path = os.path.join(root_path, file)
                relative_path = os.path.relpath(full_path, self.root)
                if self.should_process(relative_path):
                    logging.info(f'processing {relative_path}')
                    file_info = get_file_info(full_path, self.root)
                    if file_info is None:
                        continue
                    if relative_path in prev_index and prev_index[relative_path]["checksum"] == file_info["checksum"]:
                        # Reuse previous result if checksum hasn't changed
                        reused.append(prev_index[relative_path])
                    else:
                        result.append(file_info)
                else:
                    logging.info(f'skipping {relative_path}')
        return result, reused
