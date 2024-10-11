import logging
import os
import requests
import time
import json

from typing import List, Tuple

from llindex.context import ChunkContext

class GroqClient:
    def __init__(self, tokens_rate=20000, period=60, max_tokens=4096, model='llama-3.1-70b-versatile'):
        # Load API key from environment variable
        self.api_key: str = os.environ.get('GROQ_API_KEY')

        if self.api_key is None:
            logging.error("GROQ_API_KEY environment variable not set")

        # we need that for client-side rate limiting
        self.history: List[Tuple[float, int]] = []
        self.tokens_rate: int = tokens_rate
        self.period: int = period
        self.max_tokens: int = max_tokens
        self.model: str = model
        self.url = 'https://api.groq.com/openai/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
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

    def query(self, context: ChunkContext):
        req = {
            "max_tokens": self.max_tokens,
            "model": self.model,
            "messages": [
                {"role": "user", "content": context.message}
            ]
        }
        payload = json.dumps(req)

        payload_size = context.token_counter(payload)
        if payload_size > self.tokens_rate:
            err = f'unable to send message of {payload_size} tokens. Limit is {self.tokens_rate}'
            logging.error(err)
            context.metadata['error'] = err
            return None

        current_time = time.time()
        self.history = [(t, s) for t, s in self.history if current_time - t <= self.period]
        self.history.append((current_time, payload_size))

        wait_for = self.wait_time()

        if wait_for > 0:
            context.metadata['groq_wait'] = wait_for
            logging.info(f'groq client-side rate-limiting. Waiting for {wait_for} seconds')
            time.sleep(wait_for)

        # Send POST request
        response = requests.post(self.url, headers=self.headers, data=payload)

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"{response.text}")
            context.metadata['error'] = response.text
            return None

        # TODO: log usage here as well
        res = response.json()
        # TODO: check that it's a success
        content = res['choices'][0]['message']['content']
        return content

    def model_id(self):
        return f'groq:{self.model}'

