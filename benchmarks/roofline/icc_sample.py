#!/usr/bin/env python3
"""Read-only ICC vote sampling. These values are NOT a DDR clock readout."""
import json
from pathlib import Path
import time

summary = Path('/sys/kernel/debug/interconnect/interconnect_summary')
samples = []
started = time.monotonic()
for _ in range(60):
    node = None
    rows = []
    for line in summary.read_text().splitlines()[2:]:
        if line and not line[0].isspace():
            node = line.split()[0]
        if node and node.split('@')[0] in ('llcc_mc', 'ebi', 'chm_apps', 'qns_llcc'):
            if not line[0].isspace() or '.pmu' in line:
                rows.append(line.strip())
    samples.append({'monotonic': time.monotonic(), 'votes': rows})
    time.sleep(.25)
print(json.dumps({'kind':'icc_bandwidth_votes', 'units':'kB/s requested, NOT actual DDR frequency',
                  'started_monotonic':started, 'samples':samples}))
