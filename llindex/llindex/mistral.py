import logging
import os
import requests
import time
import json

from typing import List, Tuple

from llindex.token_counters import token_counter_claude

class MistralClient:
    def __init__(self, tokens_rate=200000, period=60, max_tokens=4096, model='mistral-large-latest'):
        # Load API key from environment variable
        self.api_key: str = os.environ.get('MISTRAL_API_KEY')

        if self.api_key is None:
            logging.error("MISTRAL_API_KEY environment variable not set")

        # we need that for client-side rate limiting
        self.history: List[Tuple[float, int]] = []
        self.tokens_rate: int = tokens_rate
        self.period: int = period
        self.max_tokens: int = max_tokens
        self.model: str = model
        self.url = 'https://api.mistral.ai/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
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
            "max_tokens": self.max_tokens,
            "model": self.model,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        payload = json.dumps(req)

        payload_size = token_counter_claude(payload)
        if payload_size > self.tokens_rate:
            logging.error(f'unable to send message of {payload_size} tokens. Limit is {self.tokens_rate}')
            return None

        current_time = time.time()
        self.history = [(t, s) for t, s in self.history if current_time - t <= self.period]
        self.history.append((current_time, payload_size))

        wait_for = self.wait_time()

        if wait_for > 0:
            logging.info(f'mistral client-side rate-limiting. Waiting for {wait_for} seconds')
            time.sleep(wait_for)

        # Send POST request
        response = requests.post(self.url, headers=self.headers, data=payload)

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"{response.text}")
            return None

        res = response.json()
        content = res['choices'][0]['message']['content']
        return content

