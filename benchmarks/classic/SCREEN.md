# Real classic screen

Run `screen.py` under the existing lease/memory/power runner:

```sh
python benchmarks/roofline/run.py command --output /absolute/screen-outer.jsonl \
  --timeout-seconds 900 --load-timeout-seconds 180 \
  --native-ready-log /absolute/fresh-screen/native-ready -- \
  python benchmarks/classic/screen.py --profile /absolute/effective-profile.json \
  --output /absolute/fresh-screen
```

The profile uses the same fields as `normal13.py`. This starts one real classic
CPU server with the supplied threads, context, flags and environment. Model
identity is checked by stat; engine bytes are checked against the profile SHA.
Model payload is never read by the driver. The readiness marker follows actual
native health. Output lives only in the fresh private directory.

Two original fixed code tasks (`merge_intervals`, `parse_port`) use the unchanged
copied checker, which AST-filters and executes generated code only inside
bwrap. Each gets at most one repair with safe independent-check feedback.
The five original chat/tool cases retain their prompts, expected values and
strict final JSON schema. `screen/functional.patch` removes only an unused
outer-framework import. Source provenance and checked source hashes are in
`screen/SOURCES.json` and `screen/source-manifest.json`.

The final stdout JSON and `summary.json` contain actual case results and source,
profile, engine and cleanup checks. Raw responses are synthetic/private output
artifacts. This screen supplies quality evidence; the outer guard/lease/sensors,
OpenCode normal13, controller lifecycle and PrefixStore behavior need their own
actual observations. Optional private server slots do not establish a cache hit.

Model-free checks:

```sh
python -m unittest discover -s benchmarks/classic -p test_screen.py
```
