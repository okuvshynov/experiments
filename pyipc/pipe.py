import sys
from multiprocessing import Process, Pipe

def child_process(pipe):
    # The child process waits for a message from the parent
    while True:
        message = pipe.recv()
        print(f"Child received message: {message}")
        # Sending a response back to the parent
        pipe.send("Message received by child!")
        if message == 'q':
            break
    pipe.close()

if __name__ == '__main__':
    # Create a bidirectional Pipe
    parent_conn, child_conn = Pipe()

    # Start the child process with the child side of the Pipe
    p = Process(target=child_process, args=(child_conn,))
    p.start()

    # Send a message to the child
    parent_conn.send("Hello from parent!")
    
    # Wait for a response from the child
    response = parent_conn.recv()
    print(f"Parent received response: {response}")

    parent_conn.send('q')

    p.join()