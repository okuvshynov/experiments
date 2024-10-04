from llindex.crawler import FileEntryList, FileEntry, Index

from typing import List, Dict, Any

def chunk_tasks(files: FileEntryList, token_limit: int) -> List[FileEntryList]:
    chunks = []
    current_chunk = []
    current_size = 0

    for file in files:
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
