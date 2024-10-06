import logging
import requests
import json

class LocalClient:
    def __init__(self, max_tokens=4096, endpoint='http://localhost/v1/chat/completions'):
        self.max_tokens: int = max_tokens
        self.endpoint = endpoint
        self.headers = {
            'Content-Type': 'application/json',
        }

    def query(self, message):
        #logging.info(f'sending: {message}')
        req = {
            "n_predict": self.max_tokens,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        payload = json.dumps(req)

        # Send POST request
        try:
            response = requests.post(self.endpoint, headers=self.headers, data=payload)
        except requests.exceptions.ConnectionError:
            logging.error(f'Connection error')
            return None

        # Check if the request was successful
        if response.status_code != 200:
            logging.error(f"{response.text}")
            return None

        res = response.json()
        logging.info(f'Local LLM usage: {res["usage"]}')
        content = res['choices'][0]['message']['content']
        logging.info(content)
        return content

