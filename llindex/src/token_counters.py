# we need this to estimate size, not to actually run tokenization

import sys
import tiktoken

def token_counter_claude(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)
