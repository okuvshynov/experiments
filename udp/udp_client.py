#!/usr/bin/env python3
"""
UDP Latency Test Client
Run this on the Mac Pro (10.0.0.1)
Usage: python3 udpclient.py 10.0.0.2
"""
import socket
import time
import sys
import statistics

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 udpclient.py <server_ip>")
        print("Example: python3 udpclient.py 10.0.0.2")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 65400))
    
    print(f"[*] Sending to {server_ip}:4321 from port 65400")
    print(f"[*] Press Ctrl+C to stop")
    print()
    
    rtts = []
    start_time = time.time()
    
    try:
        while True:
            t1 = time.time()
            sock.sendto(b'\x00' * 32, (server_ip, 4321))
            data, _ = sock.recvfrom(1024)
            t2 = time.time()
            
            rtt = (t2 - t1) * 1_000_000  # microseconds
            rtts.append(rtt)
            
            # Print stats every second
            if time.time() - start_time >= 1.0:
                if rtts:
                    avg = statistics.mean(rtts)
                    min_rtt = min(rtts)
                    max_rtt = max(rtts)
                    stdev = statistics.stdev(rtts) if len(rtts) > 1 else 0
                    pps = len(rtts)
                    print(f"pps={pps:5d} avg={avg:7.2f}us min={min_rtt:7.2f}us max={max_rtt:7.2f}us stdev={stdev:6.2f}us")
                rtts = []
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("\n[*] Stopped")

if __name__ == "__main__":
    main()
