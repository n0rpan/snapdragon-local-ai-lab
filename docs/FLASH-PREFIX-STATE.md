# Flash persistent-prefix prerequisite

**V4 passed the complete tiny native prerequisite: all 16 calls.** Across three divergent public continuations, cold, split and fresh-process import produced byte-identical complete logits and canonical serialized state. Immediate reserialization after import, two deterministic exports and all five negative cases also passed. There was no numeric tolerance or argmax substitute. [Exact V4 build/native/source identities and comparisons](../benchmarks/results/flash-prefix-prerequisite-v4.json).

The fixture has two synthetic experts and two Qwen4exp layers covering recurrent state, PLE history and compressed attention. It uses a 32-token public prefix, 28-token remaining continuation, vocabulary 128, context 128, microbatch 32, F16 KV and Flash Attention off. Scalar token positions are valid in the pinned compatibility path: position `p` expands to `[p,p,p,0]`. No position correction or fixture substitution was needed.

The guarded build took 152.839 seconds with 761,270,272 bytes cgroup peak; native qualification took 0.6756 seconds with 29,794,304 bytes peak. No hard memory/PID events, new swap activity or remaining children were recorded. These timings describe the prerequisite, not a persistent-prefix speedup.

## Preserved V3 failure

The preceding V3 build succeeded, but its complete test suite failed in the dirty-state negative. Its positive cold/split/import comparisons and partial/hole/wrapped-map negatives had passed. Calling `seq_add` then aborted on this model's multi-position assertion, confirmed by the owned-process backtrace and source. Corrupt-state rejection and the second export did not run. [The V3 failed receipt and cause remain unchanged](../benchmarks/results/flash-prefix-prerequisite-v3.json).

V4 changed that negative to mark owned pool metadata dirty directly and require the same exporter rejection. It retained the positive comparisons and other negative cases, and completed them all. This does not retroactively turn V3 into a pass.

## Remaining actual-model and application gates

V4 covers only the tiny synthetic fixture. The next gate is the actual original **512-expert/native-top-10** model at its intended **4,096-token** static-prefix frontier and real context/microbatch/attention settings. Measure serialization memory, prove exact restore and divergent continuations, and bind a private atomic snapshot to model, tokenizer, build, settings and public instruction/tool identities. Exclude user turns and tool results. Session integration, invalidation, model switching, idle release and complete-task benefit remain required. The installed Flash profile still has in-RAM reuse only; this result does not qualify a production cache or trigger a full task automatically.
