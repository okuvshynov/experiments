import os
import hashlib
import logging
from typing import List, Dict, Any

from token_counters import token_counter_claude

def get_file_info(path: str) -> Dict[str, Any]:
    """Get file information including:
        - path
        - size in bytes
        - checksum
        - size in tokens (cl100k)
    """
    size = os.path.getsize(path)
    approx_token_count = 0
    with open(path, "r") as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            logging.warn(f'non-utf file, skipping {path}')
            return None
        print(path)
        file_hash = hashlib.md5()
        file_hash.update(content.encode("utf-8"))
        approx_token_count += token_counter_claude(content)
    
    return {
        "path": path,
        "size": size,
        "checksum": file_hash.hexdigest(),
        "approx_tokens": approx_token_count
    }

if __name__ == '__main__':
    print(get_file_info(__file__))
