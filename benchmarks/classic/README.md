# Real OpenCode normal13 adapter

`normal13.py` runs the existing synthetic protected parse-port repair through the
pinned real OpenCode binary and the supplied classic llama-server profile. It
creates only a fresh private output directory. Wrap it in the existing roofline
runner for the inference lease, memory guard, power policy and thermal sampling:

```sh
python benchmarks/roofline/run.py command --output /absolute/outer.jsonl \
  --timeout-seconds 900 --load-timeout-seconds 180 \
  --native-ready-log /absolute/fresh-normal13/native-ready -- \
  python benchmarks/classic/normal13.py --profile /absolute/effective-profile.json \
  --output /absolute/fresh-normal13
```

The profile is one resolved classic profile with `model_path`, `expected_bytes`,
`server_binary`, `server_sha256`, `threads`, `context_size`, `extra_args`, and
`environment`. The driver checks model metadata and hashes the engine/OpenCode;
it never hashes or otherwise reads the model payload. `--opencode PATH` permits
relocating the pinned executable, with its existing SHA256 still required.

Network, home, configuration and state are isolated with the existing bwrap
layout. Only the model/engine, synthetic task and pinned executable enter it.
The protected checker remains read-only. The inner caller retains its existing
output bounds, 900-second agent deadline, reader joining and child cleanup.
The outer runner's 900-second total limit can end the trial sooner.

`normal13/inside.patch` records the complete adaptation from the original inner
caller: profile context/threads, optional slots in a fresh private directory,
and a readiness marker after successful native health. No task, checker, oracle,
or agent request was changed. Enabling slots here does not test the controller's
PrefixStore restore/isolation behavior; that needs a separate controller test.

The driver freezes/checks repository inputs and copied runner/config inputs,
checks the protected test hash, and emits one JSON summary to stdout and
`summary.json`. `result.json` is the unchanged inner caller's detailed output.
Failures remain failures. RAM recovery, guard results, lease release and sensor
observations belong to the outer runner and must accompany the summary before
qualification. This adapter does not deploy or qualify a production profile.

Model-free checks:

```sh
python -m unittest discover -s benchmarks/classic -p test_normal13.py
```
