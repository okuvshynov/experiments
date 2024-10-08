import http.server
import json
import plistlib
import subprocess
import threading

from collections import defaultdict


class Metrics:
    def __init__(self, width):
        self.data = defaultdict(list)
        self.lock = threading.Lock()
        self.width = width

    def append(self, slice):
        with self.lock:
            for k, v in slice.items():
                self.data[k].append(v)
                if len(self.data[k]) > self.width:
                    self.data[k].pop(0)

    def get(self):
        with self.lock:
            return {k: v[:] for k, v in self.data.items()}


class MacOSReader:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        out = subprocess.check_output(['pagesize'])
        self.page_size = int(out.decode('utf-8').strip())
        out = subprocess.check_output(['sysctl', 'hw.memsize'])
        self.mem_size = int(out.decode('utf-8').split(':')[-1].strip())

    def get_vm_stat(self):
        """
        Call vm_stat and parse its output to return active and wired memory.
        """
        # Call vm_stat and capture its output
        output = subprocess.check_output(['vm_stat']).decode('utf-8')

        # Split the output into lines
        lines = output.split('\n')

        # Initialize variables to store active and wired memory
        active_pages = 0
        wired_pages = 0

        # Iterate over the lines to find the active and wired memory
        for line in lines:
            if 'Pages active:' in line:
                active_pages = int(line.split(':')[-1].strip().rstrip('.'))
            elif 'Pages wired down:' in line:
                wired_pages = int(line.split(':')[-1].strip().rstrip('.'))

        # Calculate active and wired memory in bytes
        active = 1.0 * active_pages * self.page_size / self.mem_size
        wired = 1.0 * wired_pages * self.page_size / self.mem_size

        return active + wired, wired

    def parse_powermetrics_data(self, data):
        res = {'gpu': 1.0 - data['gpu']['idle_ratio']}
        ecpu_idle = 0.0
        ecpu_n_cpus = 0
        pcpu_idle = 0.0
        pcpu_n_cpus = 0

        for cluster in data['processor']['clusters']:
            n_cpus = len(cluster['cpus'])
            idle = 0.0
            for cpu in cluster['cpus']:
                idle += cpu['idle_ratio']

            if cluster['name'].startswith('E'):
                ecpu_idle += idle
                ecpu_n_cpus += n_cpus
            elif cluster['name'].startswith('P'):
                pcpu_idle += idle
                pcpu_n_cpus += n_cpus

        if ecpu_n_cpus > 0:
            res['ecpu'] = 1.0 - ecpu_idle / ecpu_n_cpus
        if pcpu_n_cpus > 0:
            res['pcpu'] = 1.0 - pcpu_idle / pcpu_n_cpus

        return res

    def start(self, metrics):
        process = subprocess.Popen(
            [
                'sudo', 'powermetrics',
                '-i', f'{self.interval_ms}',
                '-f', 'plist',
                '-s', 'gpu_power,cpu_power'
            ],
            stdout=subprocess.PIPE
        )

        buffer = b''
        while True:
            # Read from process output until we receive '</plist>'
            while True:
                chunk = process.stdout.read(128)
                if not chunk:
                    break
                buffer += chunk
                plist_end_pos = buffer.find(b'</plist>')
                if plist_end_pos != -1:
                    break

            # Parse plist output and extract relevant data
            plist_data = buffer[:plist_end_pos + len(b'</plist>')]
            data = plistlib.loads(plist_data.strip(b'\n\x00'))
            parsed_data = self.parse_powermetrics_data(data)

            # Remove the parsed plist data from the buffer
            buffer = buffer[plist_end_pos + len(b'</plist>'):]

            parsed_data['rss'], parsed_data['wired'] = self.get_vm_stat()
            metrics.append(parsed_data)


class MetricsRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = self.server.metrics
            data = metrics.get()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            res = {
                'metrics': data,
            }
            self.wfile.write(json.dumps(res).encode())
            return

        if self.path == '/plain':
            metrics = self.server.metrics
            data = metrics.get()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            for key, value in data.items():
                values = " ".join(str(v) for v in value)
                self.wfile.write(f"{key} {values}\n".encode())
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not Found')

if __name__ == '__main__':
    metrics = Metrics(64)
    reader = MacOSReader(interval_ms=500)
    thread = threading.Thread(target=reader.start, args=(metrics, ))
    thread.daemon = True
    thread.start()

    endpoint = ('0.0.0.0', 8087)
    server = http.server.ThreadingHTTPServer(endpoint, MetricsRequestHandler)
    server.metrics = metrics
    server.serve_forever()
