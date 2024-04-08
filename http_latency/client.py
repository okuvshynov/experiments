import requests
import time
import fewlines.metrics as fm

url = 'http://192.168.1.81:5678/test'
# Create a payload of approximately 10-20Kb. Here it's a simple string.
payload = {'data': 'X' * 10240}  # Adjust the multiplication factor as needed


with requests.Session() as session:
    session.headers.update({'Connection': 'keep-alive'})

    # Send multiple requests within the same session
    while True:
        start_time = time.time()
        response = session.post(url, json=payload)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        fm.add("rountrip_latency_ms", 1000.0 * (end_time - start_time))
        for h in fm.histogram('rountrip_latency_ms', n_lines=2, left_margin=40, custom_range=(0, 200)):
            print(h)

        time.sleep(0.01)
