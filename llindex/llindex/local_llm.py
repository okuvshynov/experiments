import logging
import os
import requests
import time
import json

from llindex.token_counters import token_counter_claude

class LocalClient:
    def __init__(self, max_tokens=4096, endpoint='http://localhost/v1/chat/completions'):
        self.max_tokens: int = max_tokens
        self.endpoint = endpoint
        self.headers = {
            'Content-Type': 'application/json',
        }

    def query(self, message):
        req = {
            "n_predict": self.max_tokens,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        payload = json.dumps(req)

        # Send POST request
        response = requests.post(self.endpoint, headers=self.headers, data=payload)

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"{response.text}")
            return None

        res = response.json()
        content = res['choices'][0]['message']['content']
        return content

