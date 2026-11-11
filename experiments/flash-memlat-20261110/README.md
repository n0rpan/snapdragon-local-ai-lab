# Flash post-memlat prerequisites

Header-only audit, no model load or payload reads:

```sh
python3 experiments/flash-memlat-20261110/header_audit.py --runtime ~/.config/local-ai/llm/runtime.json --output /tmp/flash-header.json
```

The reader is the offline NPU-budget GGUF parser with short-array retention extended to 64 entries; the audit adds the pinned fork's Q2_0 type42 (64 elements /18 bytes). Runtime identities are recorded, not recomputed by reading 84GB of weights. The output distinguishes stored bytes from measured runtime traffic.

P1 uses the existing synthetic `ascii-port-code-v1` fixture and independent Python oracle, unchanged decode4/batch6/IO4/cache12GiB/ubatch4096/F16/no-speculation settings. The canonical lease excludes builds and other model runs. This is a short screen, not the protected agent task. Native top10 and all512 experts are retained. Private launchers and raw synthetic answer remain in the ops backup; scalar results are under benchmarks/roofline/.

Locality replay (IDs-only CSV with turn/phase/step/layer/slot/expert, plus the numeric native summary):

```sh
python3 experiments/flash-memlat-20261110/analyze_routes.py --trace expert-ids.csv --header flash-header.json --native-result summary.json --output locality.json
```

The replay starts empty, accounts actual mixed-format bytes per expert/layer, coalesces duplicate prefill routes, and protects the current layer batch. Its twelve-GiB overall hit rate rounds to the observed 77.8%. Larger-cache rows are offline projections only. Native tracing adds graph nodes/barriers: its latency is not a speed benchmark. The private richer report adds optimistic all-accepted union-of-experts estimates; no MTP speed is inferred from them.
