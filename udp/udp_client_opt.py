"""
Optimized UDP Latency Test Client with Busy Polling
Run with: sudo python3 udpclient_optimized.py 10.0.0.2

This version uses:
- Non-blocking sockets with busy polling
- Higher process priority (requires sudo)
- More aggressive timing
"""
import socket
import time
import sys
import statistics
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: sudo python3 udpclient_optimized.py <server_ip>")
        print("Example: sudo python3 udpclient_optimized.py 10.0.0.2")
        print("Note: Requires sudo for process priority")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    
    # Set high process priority
    try:
        os.nice(-20)
        print("[+] Process priority set to highest")
    except PermissionError:
        print("[!] Warning: Run with sudo for highest priority")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 65400))
    sock.setblocking(False)  # Non-blocking for busy polling
    
    print(f"[*] Sending to {server_ip}:4321 (BUSY POLLING MODE)")
    print(f"[*] This will use 100% of one CPU core")
    print(f"[*] Press Ctrl+C to stop")
    print()
    
    rtts = []
    start_time = time.time()
    
    try:
        while True:
            t1 = time.time()
            sock.sendto(b'\x00' * 32, (server_ip, 4321))
            
            # Busy poll for response
            while True:
                try:
                    data, _ = sock.recvfrom(1024)
                    break
                except BlockingIOError:
                    pass  # Keep polling
            
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
