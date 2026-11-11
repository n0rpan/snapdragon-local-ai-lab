bench.py is far too slow for production volumes. `python3 bench.py 200000` takes forever because flag_bursts and top_paths in ratelimit/analyze.py are naive.

Make both functions fast enough that `python3 bench.py 200000` finishes in under 3 seconds on this machine. Results must not change: the semantics in the docstrings of ratelimit/analyze.py are the contract, including tie-breaking, the window boundaries, and events with equal timestamps. Keep the function signatures.

Run the visible tests with: python3 -m unittest discover -s tests -t .
