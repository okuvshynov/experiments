import requests

def chat(addr):
    headers = {'Content-Type': 'application/json'}

    history = []

    url = f'{addr}/messages'

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat...")
            break

        history.append({"role": "user", "content": user_input})

        data = {
            "system": "You are a helpful, respectful and honest assistant.",
            "max_tokens": 2048,
            "messages": history
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            assistant_message = response.json()['content']['text']
            print("AI:", assistant_message)
            history.append({"role": "assistant", "content": assistant_message})
        else:
            print("Failed to get response:", response.text)
            break

if __name__ == "__main__":
    chat("http://localhost:5555")
