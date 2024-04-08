import requests
import time

url = 'http://127.0.0.1:5678/test'
# Create a payload of approximately 10-20Kb. Here it's a simple string.
payload = {'data': 'X' * 10240}  # Adjust the multiplication factor as needed

start_time = time.time()
response = requests.post(url, json=payload)
end_time = time.time()

latency = end_time - start_time
print(f"Latency: {latency * 1000} ms")