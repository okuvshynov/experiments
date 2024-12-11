import logging
import requests
import json
import time

N_PREDICT = 2048

SYS_PROMPT = """
When you receive a tool call response, use the output to format an answer to the orginal user question.

You are a helpful assistant with tool calling capabilities.
"""

USER_MESSAGE = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
    "name": "get_current_conditions",
    "description": "Get the current weather conditions for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g., San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["Celsius", "Fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location."
        }
        },
        "required": ["location", "unit"]
    }
    }
}

Question: what is the weather like in San Fransisco?
"""

ENDPOINT = "http://localhost:8080/v1/chat/completions"

def query():
    req = {
        "n_predict": 2048,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_MESSAGE}
        ]
    }
    payload = json.dumps(req)
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.post(ENDPOINT, headers=headers, data=payload)
    print(response.json())


if __name__ == '__main__':
    query()
