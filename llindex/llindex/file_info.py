import os
import hashlib
import logging
from typing import Dict, Any

def get_file_info(full_path: str, root: str) -> Dict[str, Any]:
    """Get file information including:
        - path
        - size in bytes
        - checksum
        - size in tokens (cl100k)
    """
    size = os.path.getsize(full_path)
    with open(full_path, "r") as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            logging.warning(f'non-utf file, skipping {full_path}')
            return None
        file_hash = hashlib.md5()
        file_hash.update(content.encode("utf-8"))
    
    return {
        "path": os.path.relpath(full_path, root),
        "size": size,
        "checksum": file_hash.hexdigest(),
    }
