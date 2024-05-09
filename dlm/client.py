import json
import requests

#prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease give a detailed description of concurrency and parallelism in Python. Provide some examples.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def main():
    message = {
        "system": "You are a helpful, respectful and honest assistant.",
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": "Please give a detailed description of concurrency and parallelism in Python. Provide some examples."},
        ]
    }

    url = "http://localhost:5555/messages" 

    message = json.dumps(message)
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=message)
    if response.status_code == 200:
        print("Server responded with: ", response.text)
    else:
        print("Failed to get a valid response, status code: ", response.status_code)

if __name__ == "__main__":
    main()
