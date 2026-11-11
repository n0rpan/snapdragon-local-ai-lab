# Flash F123, installed 2026-11-11

F123 replaces package2's engine and explicit prefill-order setting: server `001e0b85…` / commit `4b6ab92`, with `--flash-pf-order 2`. It keeps the 14 GiB cache, 16K context, 8/11 compute threads and IO CPU8. [Portable evidence](../benchmarks/results/flash-f123-20261111.json).

Order2 keeps resident-first prefill reads, then orders residents and misses separately by their final prompt use. This leaves recent prompt experts in the cache for following decode. The binary default remains order1; the installed argv explicitly selects order2.

## Matched decode

B A A B compared F123/order2 against the installed package2/order0 engine, with explicit 14 GiB and the same 75-token p2-prose request. Every arm generated 64 tokens four times; the first run was dropped and the remaining three contributed their median.

| Arm | Median ms/token |
|---|---:|
| B: F123/order2 | 78.13724 |
| A: package2/order0 | 83.81780 |
| A: package2/order0 | 86.89462 |
| B: F123/order2 | 84.12587 |

The means of those medians were **81.13155 vs 85.35621 ms/token**, a **4.95% reduction**. The ratio 0.95050554 passed the frozen <=1.010 gate. All 16 generated texts had the same SHA256, `4cd7f213e8c3b06837afce3e09a794ca27405f74dc572ff23a8b30f6f1019698`.

This is a short-prefill decode control. The JSON retains every warm sample, including a 191.058 ms/token sample in the first B arm; the declared median method was unchanged. It does not establish a general workload speedup.

## Prefill reads and time

C disabled ordering, N enabled order2, and O used order1. Each arm prefills 2,048 tokens twice with 14 GiB, 11 compute threads on CPUs 0–10, IO CPU11, and readahead off.

| Arm | Cold reads GiB | Repeat reads GiB | Cold seconds | Repeat seconds |
|---|---:|---:|---:|---:|
| C: order0 | 36.65269 | 36.62790 | 30.235 | 30.9275 |
| N: order2 | 36.65269 | 23.05271 | 33.758 | 30.9062 |
| O: order1 | 36.65269 | 23.05271 | 32.586 | 31.0408 |

N reduced repeat-pass reads from 36.62790 to 23.05271 GiB: **37.06% fewer**, passing the <=0.85 ratio gate. N and O had identical measured reads. Both saved states compared byte-for-byte equal to C, and all three arms were deterministic.

Repeat-prefill time was flat: **30.9062 vs 30.9275 seconds** for N/C. Cold N took 33.758s versus C 30.235s and O 32.586s. These are single arms; the read saving does not support a prefill-speed gain claim.

## Qualification, installation and rollback

The protected OpenCode task passed 13/13 with order2 in **182.334 seconds**, with no tool call printed as assistant text. Scope peak was **22.28 GiB**, process VmHWM 22.20 GiB, and host minimum **3.64 GiB**. There was no OOM, model swap or guard failure. This is one qualification trajectory, not a matched task-speed gain.

Isolated capture and live post-install regression passed. Controller runtime source, systemd unit, admission threshold, other model profiles, OpenCode settings and UI were unchanged. The final checks left the controller unloaded and CPU8 QoS at zero. The 29 GiB server scope, zero model swap and 1 GiB host floor remain in force; lookahead and fused phases remain off.

One-command rollback to the previous package2/order0 installation:

```sh
~/.local/share/ai-stack/rollbacks/latest-flash-f123/rollback-live.sh
```

The earlier [package2 record](FLASH-PACKAGE2.md) remains historical evidence, including its order0 qualification and earlier order1 regression.
