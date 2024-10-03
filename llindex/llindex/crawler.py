import os

from llindex.file_info import get_file_info
from typing import List, Dict, Any, Tuple

FileEntry = Dict[str, Any]
Index = Dict[str, FileEntry]
FileEntryList = List[FileEntry]

class Crawler:
    def __init__(self, root, prev: Index):
        self.root = root
        self.prev = prev

    def should_process(self, filename):
        if filename.endswith(".vim"):
            return True
        return False

    def process_directory(self, directory: str) -> FileEntryList:
        """Process directory recursively and return file information for files that should be processed."""
        result = []
        for root_path, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root_path, file)
                relative_path = os.path.relpath(full_path, self.root)
                if self.should_process(relative_path):
                    file_info = get_file_info(full_path, self.root)
                    if file_info is None:
                        continue
                    if relative_path in self.prev and self.prev[relative_path]["checksum"] == file_info["checksum"]:
                        # Reuse previous result if checksum hasn't changed
                        result.append(self.prev[relative_path])
                    else:
                        result.append(file_info)
        return result

    def chunk_files(self, files: FileEntryList, token_limit: int) -> Tuple[List[FileEntryList], FileEntryList]:
        """Split files into chunks respecting the size limit."""
        chunks = []
        current_chunk = []
        current_size = 0
        reused = []

        for file in files:
            if "processing_result" in file:
                # Skip files that have already been processed
                reused.append(file)
                continue
            if current_size + file["approx_tokens"] > token_limit:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [file]
                current_size = file["approx_tokens"]
            else:
                current_chunk.append(file)
                current_size += file["approx_tokens"]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks, reused

    def chunk_into(self, size: int):
        files = self.process_directory(self.root)
        return self.chunk_files(files, size)
