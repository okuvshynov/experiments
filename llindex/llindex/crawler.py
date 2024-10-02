import os

from llindex.file_info import get_file_info
from typing import List, Dict, Any

def should_process(filename):
    if filename.endswith(".vim"):
        return True
    return False

def process_directory(directory: str, root: str, previous_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process directory recursively and return file information for files that should be processed."""
    result = []
    for root_path, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root_path, file)
            relative_path = os.path.relpath(full_path, root)
            if should_process(relative_path):
                file_info = get_file_info(full_path)
                if file_info is None:
                    continue
                if relative_path in previous_results and previous_results[relative_path]["checksum"] == file_info["checksum"]:
                    # Reuse previous result if checksum hasn't changed
                    result.append(previous_results[relative_path])
                else:
                    result.append(file_info)
    return result

def chunk_files(files: List[Dict[str, Any]], token_limit: int) -> List[List[Dict[str, Any]]]:
    """Split files into chunks respecting the size limit."""
    chunks = []
    current_chunk = []
    current_size = 0

    for file in files:
        if "processing_result" in file:
            # Skip files that have already been processed
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

    return chunks