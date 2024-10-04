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

    def wait_time(self):
        total_size = sum(size for _, size in self.history)
        if total_size < self.tokens_rate:
            return 0
        current_time = time.time()
        running_total = total_size
        for time_stamp, size in self.history:
            running_total -= size
            if running_total <= self.tokens_rate:
                return max(0, time_stamp + self.period - current_time)

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
        logging.info(res)
        content = res['choices'][0]['message']['content']
        return content

