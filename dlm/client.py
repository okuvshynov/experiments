import zmq
import json

def main():
    context = zmq.Context()

    # calling main model
    main_socket = context.socket(zmq.REQ)
    main_socket.connect("tcp://localhost:5555")

    # Prepare JSON data
    data = {"prompt": "USS Gyatt is known for"}
    message = json.dumps(data)

    spec_socket = context.socket(zmq.REQ)
    spec_socket.connect("tcp://localhost:5566")

    # Send JSON to the server
    print("Sending:", message)
    main_socket.send_string(message)
    spec_socket.send_string(message)

    # Wait for the server's response
    response = main_socket.recv()
    print("Received:", response.decode())
    response = spec_socket.recv()
    print("Received:", response.decode())

if __name__ == "__main__":
    main()
