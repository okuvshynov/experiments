import zmq
import json

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # Prepare JSON data
    data = {"prompt": "USS Gyatt is known for"}
    message = json.dumps(data)

    # Send JSON to the server
    print("Sending:", message)
    socket.send_string(message)

    # Wait for the server's response
    response = socket.recv()
    print("Received:", response.decode())

if __name__ == "__main__":
    main()
