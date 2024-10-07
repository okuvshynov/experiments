from dataclasses import dataclass, field

from llindex.crawler import FileEntryList
from typing import List, Dict, Any, Set

@dataclass
class ChunkContext:
    directory: str
    client: Any = None
    token_counter: Any = None
    message: str = None
    files: List[FileEntryList] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    missing_files: FileEntryList = field(default_factory=list)

