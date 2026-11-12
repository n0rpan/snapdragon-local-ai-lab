#!/usr/bin/env python3
"""Read-only ddr/llcc/ddr-qos trace. No model, device, frequency or governor writes."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from snapdragon_ai_lab.devfreq_probe import Sampler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--interval-ms', type=int, default=10)
    a = p.parse_args()
    if not 0 < a.seconds <= 600:
        p.error('--seconds must be in (0,600]')
    with Sampler(a.out, interval_ms=a.interval_ms):
        time.sleep(a.seconds)
    return 0
if __name__ == '__main__':
    sys.exit(main())
