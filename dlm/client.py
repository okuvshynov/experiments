import zmq
import json

prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease give a detailed description of concurrency and parallelism in Python. Provide some examples.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def main():
    context = zmq.Context()

    expert = "tcp://localhost:5555" 
    # calling main model
    main_socket = context.socket(zmq.REQ)
    main_socket.connect(expert)

    # Prepare JSON data
    data = {"prompt": prompt, "n_predict": 256}

    message = json.dumps(data)

    # Send JSON to the server
    print("Sending:", message)
    main_socket.send_string(message)

    # Wait for the server's response
    response = main_socket.recv()
    print("Received:", response.decode())

if __name__ == "__main__":
    main()
