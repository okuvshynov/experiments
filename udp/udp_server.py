#!/usr/bin/env python3
"""
UDP Echo Server for latency testing
Run this on the Mac Studio (10.0.0.2)
"""
import socket
import sys

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 4321))
    print("[*] UDP server listening on port 4321")
    print("[*] Waiting for packets...")
    
    packet_count = 0
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            sock.sendto(data, addr)
            packet_count += 1
            if packet_count % 10000 == 0:
                print(f"[*] Echoed {packet_count} packets")
    except KeyboardInterrupt:
        print(f"\n[*] Server stopped. Total packets: {packet_count}")

if __name__ == "__main__":
    main()

