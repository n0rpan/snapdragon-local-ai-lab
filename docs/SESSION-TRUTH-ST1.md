# ST-1: back-to-back protected sessions

All six protected OpenCode tasks passed 13/13 checks. Each model ran twice with fresh clients against one unchanged production backend process. The first HTTP request body was byte-identical within every pair. This is one pair per model, not a general speed benchmark.

| Model | Run | Checks | Task s | Wrapper s | Turn 1 cached tokens | Turn 1 new tokens | Turn 1 prefill ms | Turn 1 HTTP wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ornith | 1 | 13/13 | 33.587043 | 35.305095 | 5376 | 173 | 4947.962489 | 6755.948544 |
| ornith | 2 | 13/13 | 30.628659 | 32.284760 | 5376 | 173 | 1354.021901 | 3123.790741 |
| qwen3.6 | 1 | 13/13 | 28.950557 | 30.297223 | 5376 | 176 | 6775.003241 | 9872.016907 |
| qwen3.6 | 2 | 13/13 | 19.766921 | 21.215155 | 5376 | 176 | 1412.312182 | 3072.232723 |
| Flash 18 GiB | 1 | 13/13 | 184.620747 | 186.427667 | 0 | 5547 | 99897.387691 | 107482.289076 |
| Flash 18 GiB | 2 | 13/13 | 86.060774 | 87.330140 | 5540 | 7 | 852.110011 | 8455.762625 |


Task time is the normal13 agent duration. Wrapper time includes client setup and cleanup. Turn 1 HTTP wall includes prefill, decode and request overhead; prefill is the backend's actual prompt_ms. Full precision is preserved in [the numeric receipt](../benchmarks/results/public/ST-1.json).

Ornith and Qwen already had turn 1 cache hits in run 1: 5376 cached tokens in both runs, with 173 and 176 new tokens respectively. Their observed prefill changes do not compare a fully cold prefix against a hit.

Flash 18 GiB changed from 0 cached/5547 new tokens to 5540 cached/7 new tokens in the same process. Turn 1 prefill was 99.897388 s then 0.852110 s. The two protected tasks took 184.620747 s and 86.060774 s. Disk prefix storage was OFF. This demonstrates same-process reuse; persistence across unload/reload was not tested, and no general speedup follows from this one pair.

The production backend identity stayed stable within every pair. Standard memory guards remained active, the no-OOM/own-swap checks passed, each pair ended unloaded, QoS reset to 0, and the default Ornith selection and production files stayed unchanged. The full run took 412.767661 s within the 600 s cap. Ornith/Qwen used 32K context and 8 threads; Flash used 16K context, 8 threads and 18 GiB expert cache. MoE engine SHA 71d927e4ca0887c8adfee2ca375586b55ad7f6258b4cc2c6e32196630ea5d84e; Flash SHA b565c250e4b3339ee589dfe0f416d1d2000e7d1d54d7522ec4b6bec5bc4a89ff.

An earlier attempt stopped during backend attachment before any protected task, because the harness checked for the native process too early. It is retained as a harness-only failure, not an engine/task regression.

Heat and queue work stays closed under SESSION-TRUTH; SX2/F106/F110 stay closed and there is no G2 replay. O35 drops the 009 hard-task work. Next is the separately specified Flash disk-prefix persistence ticket, followed by larger-image work and v0.1.1. No tuning, production change or additional model run is part of this report.
