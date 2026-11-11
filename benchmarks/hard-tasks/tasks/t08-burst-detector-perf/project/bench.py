"""Benchmark: python3 bench.py [N]  (default N=200000)."""
import hashlib
import random
import sys
import time

from ratelimit.analyze import flag_bursts, top_paths


def make_events(n, seed=7):
    rng = random.Random(seed)
    clients = [f"10.{i // 256}.{i % 256}.1" for i in range(2000)]
    paths = [f"/api/v1/item/{i}" for i in range(1200)]
    span = max(1, n // 50)
    return [(float(rng.randint(0, span)), rng.choice(clients), rng.choice(paths)) for _ in range(n)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    events = make_events(n)
    t0 = time.perf_counter()
    flagged = flag_bursts(events, 60.0, 3)
    t1 = time.perf_counter()
    top = top_paths(events, 10)
    t2 = time.perf_counter()
    digest = hashlib.sha256(repr((flagged, top)).encode()).hexdigest()
    print(f"n={n} flag_bursts={t1 - t0:.3f}s top_paths={t2 - t1:.3f}s total={t2 - t0:.3f}s")
    print(f"flagged={len(flagged)} sha256={digest}")


if __name__ == "__main__":
    main()
