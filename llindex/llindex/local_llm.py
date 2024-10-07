import logging
import requests
import json
import time

from llindex.chunk_ctx import ChunkContext

class LocalClient:
    def __init__(self, max_tokens=4096, endpoint='http://localhost/v1/chat/completions'):
        self.max_tokens: int = max_tokens
        self.endpoint = endpoint
        self.headers = {
            'Content-Type': 'application/json',
        }
        # TODO: fix tokenizer url completion
        self.tokenize_endpoint = endpoint[:-len('v1/chat/completions')] + 'tokenize'

    def query(self, context: ChunkContext):
        #logging.info(f'sending: {message}')
        req = {
            "n_predict": self.max_tokens,
            "messages": [
                {"role": "user", "content": context.message}
            ]
        }
        payload = json.dumps(req)
        tokens = self.token_count(context.message)
        logging.info(f'Calling local server with {tokens} tokens')

        try:
            response = requests.post(self.endpoint, headers=self.headers, data=payload)
        except requests.exceptions.ConnectionError:
            context.metadata['error'] = 'Connection Error'
            logging.error(f'Connection error')
            return None

        # Check if the request was successful
        if response.status_code != 200:
            context.metadata['error'] = response.text
            logging.error(f"{response.text}")
            return None

        res = response.json()
        logging.info(res)
        context.metadata['usage'] = res['usage']
        content = res['choices'][0]['message']['content']
        logging.info(content)
        return content

    def model_id(self):
        return f'local'

    def token_count(self, text):
        req = {
            "content": text
        }
        payload = json.dumps(req)
        try:
            response = requests.post(self.tokenize_endpoint, headers=self.headers, data=payload)
        except requests.exceptions.ConnectionError:
            logging.error(f'Connection error')
            return 0
        res = response.json()
        return len(res['tokens'])
