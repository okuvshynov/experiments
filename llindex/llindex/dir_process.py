import os
import logging
from datetime import datetime

from llindex.file_info import get_file_info
from typing import List, Dict, Any

def should_process(filename):
    return True

def process_directory(directory: str, root: str, previous_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process directory recursively and return file information for files that should be processed."""
    result = []
    for root_path, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root_path, file)
            relative_path = os.path.relpath(full_path, root)
            if should_process(relative_path):
                file_info = get_file_info(relative_path)
                if file_info is None:
                    continue
                if relative_path in previous_results and previous_results[relative_path]["checksum"] == file_info["checksum"]:
                    # Reuse previous result if checksum hasn't changed
                    result.append(previous_results[relative_path])
                else:
                    result.append(file_info)
    return result

def process(files: List[Dict[str, Any]]) -> List[str]:
    """Mock function to simulate server processing."""
    logging.info(f'processing {len(files)} files')
    return [f"Processed {file['path']}" for file in files]

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

def process_directory_with_limit(directory: str, size_limit: int, previous_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process directory with size limit and return results for files that should be processed."""
    files = process_directory(directory, directory, previous_results)
    chunks = chunk_files(files, size_limit)
    
    results = []
    for chunk in chunks:
        processing_results = process(chunk)
        timestamp = datetime.now().isoformat()
        for file, result in zip(chunk, processing_results):
            file_result = {
                "name": file["path"],
                "size": file["size"],
                "checksum": file["checksum"],
                "processing_result": result,
                "processing_timestamp": timestamp,
                "approx_tokens": file["approx_tokens"] 
            }
            results.append(file_result)
    
    # Add previously processed files that weren't reprocessed
    for file in files:
        if "processing_result" in file:
            results.append(file)
    
    return results

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    process_directory_with_limit(".", 512, {})

if __name__ == '__main__':
    main()
