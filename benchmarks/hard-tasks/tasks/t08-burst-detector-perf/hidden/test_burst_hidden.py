import json
import os
import random
import subprocess
import sys
import unittest

PROJECT = os.environ["PROJECT"]
sys.path.insert(0, PROJECT)

from ratelimit.analyze import flag_bursts, top_paths  # noqa: E402

PERF_DIGEST = "f25c7b2531eca18ee6e7ae1462398b1e5a569d1d74e9d396f2139aef45ac4ed6"

PERF_CODE = r'''
import hashlib, json, os, random, sys, time
sys.path.insert(0, os.environ["PROJECT"])
from ratelimit.analyze import flag_bursts, top_paths
rng = random.Random(4242)
n = 200000
clients = ["h%04d" % i for i in range(2000)]
paths = ["/r/%d" % i for i in range(1000)]
events = [(float(rng.randint(0, 4000)), rng.choice(clients), rng.choice(paths)) for _ in range(n)]
t0 = time.perf_counter()
flagged = flag_bursts(events, 60, 3)
top = top_paths(events, 10)
elapsed = time.perf_counter() - t0
digest = hashlib.sha256(repr((flagged, top)).encode()).hexdigest()
print(json.dumps({"elapsed": elapsed, "digest": digest, "flagged": len(flagged)}))
'''


def naive_flag(events, window, limit):
    flagged = []
    for i, (t_i, client, _) in enumerate(events):
        count = 0
        for t_j, other, _ in events:
            if other == client and t_i - window < t_j <= t_i:
                count += 1
        if count > limit:
            flagged.append(i)
    return flagged


def naive_top(events, n):
    counts = {}
    for _, _, path in events:
        counts[path] = counts.get(path, 0) + 1
    pairs = list(counts.items())
    pairs.sort(key=lambda pc: (-pc[1], pc[0]))
    return pairs[:n]


def small_case(rng):
    n = rng.randint(0, 60)
    mode = rng.random()
    if mode < 0.4:
        ts = lambda: rng.randint(0, 12)
    elif mode < 0.7:
        ts = lambda: rng.randint(0, 24) * 0.5
    else:
        ts = lambda: rng.randint(0, 6)
    clients = ["a", "b", "c", "d"][: rng.randint(1, 4)]
    events = [(ts(), rng.choice(clients), rng.choice(["/x", "/y", "/z"])) for _ in range(n)]
    window = rng.choice([0, 0.5, 1, 1.5, 2, 3, 5])
    limit = rng.randint(0, 5)
    return events, window, limit


class EquivalenceTest(unittest.TestCase):
    def test_random_small_cases(self):
        rng = random.Random(20240611)
        for case in range(400):
            events, window, limit = small_case(rng)
            with self.subTest(case=case):
                self.assertEqual(flag_bursts(events, window, limit), naive_flag(events, window, limit))
                k = rng.randint(0, 4)
                self.assertEqual(top_paths(events, k), naive_top(events, k))

    def test_medium_case_3000_events(self):
        rng = random.Random(99)
        events = [(float(rng.randint(0, 300)), rng.choice("abcde"),
                   rng.choice(["/a", "/b", "/c", "/d"])) for _ in range(3000)]
        self.assertEqual(flag_bursts(events, 20, 4), naive_flag(events, 20, 4))
        self.assertEqual(top_paths(events, 4), naive_top(events, 4))

    def test_return_types(self):
        events = [(1.0, "a", "/p"), (1.5, "a", "/q")]
        self.assertIsInstance(flag_bursts(events, 2, 0), list)
        self.assertIsInstance(top_paths(events, 2), list)
        self.assertEqual(top_paths(events, 2), [("/p", 1), ("/q", 1)])
        self.assertEqual(top_paths([], 5), [])
        self.assertEqual(flag_bursts([], 5, 0), [])


class PerformanceTest(unittest.TestCase):
    def test_200k_events_under_4s_and_exact_result(self):
        env = dict(os.environ, PROJECT=PROJECT, PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run([sys.executable, "-B", "-c", PERF_CODE], env=env,
                                  capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self.fail("200k-event workload did not finish within 30 s")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertLess(out["elapsed"], 4.0, "flag_bursts + top_paths took %.2f s" % out["elapsed"])
        self.assertEqual(out["digest"], PERF_DIGEST)


if __name__ == "__main__":
    unittest.main()
