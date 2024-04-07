from fewlines import metrics as fm
from fewlines import dashboard as fd
import numpy as np
import time

for i, v in enumerate(np.random.lognormal(mean=1.0, sigma=0.7, size=1000)):
    fm.add('ssd_read_latency', v, time.time() - i)
for i, v in enumerate(np.random.lognormal(mean=3.0, sigma=0.7, size=1500)):
    fm.add('nw_recv_latency', v, time.time() - i)

print("\n## Default one-line dashboards with wildcard")
for s in fd.histograms('*latency'):
    print(s)

for s in fd.timeseries('*latency'):
    print(s)

print()
print('\n## two histograms with separate scales with larger height and horizon colors')
for s in fd.dashboard({"charts": [('*latency', 'histogram')], "n_lines": 3, "color": 'green'}):
    print(s) 

print()
print('\n## two histograms sharing the scale as they are part of the same group with larger height and horizon colors')
for s in fd.dashboard({"charts": [[('*latency', 'histogram')]], "n_lines": 3, "color": 'green'}):
    print(s)

print()
conf = {
    "title": "Custom Dashboard",
    "charts": [
        ('ssd_read_latency', 'timeseries', {'n_lines': 3, 'color': None}),
        [
            ('ssd_read_latency', 'timeseries'),
            ('ssd_read_latency', 'timeseries', {'agg': 'max'}),
            ('ssd_read_latency', 'timeseries', {'agg': 'min'}),
        ],
        ('ssd_read_latency', 'histogram', {'n_lines': 4, 'color': 'green'}),
        ('ssd_read_latency', 'histogram', {'n_lines': 4, 'color': 'gray'}),
        ('ssd_read_latency', 'histogram', {'n_lines': 6, 'color': None}),
    ],
    "time": -600, # default -3600
    "bins": 60, # default 60
    "left_margin": 40, # default 30
    "n_lines": 3,
    "color": None,
}
print('\n## detailed complicated config with different aggregations')
for s in fd.dashboard(conf):
    print(s)