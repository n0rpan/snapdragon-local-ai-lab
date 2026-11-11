# Isolated Flash timing replay

These helpers record the hardware-lane M0 cache, queue-depth and placement screens. They use real weights/kernels and replayed routes with **dummy activations**, so neither generated token IDs nor a completed coding-task speedup can be inferred.

The original M0 source worktree was advanced by the parallel M1 lane. A SHA preflight caught the overwritten executable before the next experiment. An independent detached BF97 checkout/target was built and its seven tests passed. Its SHA differs from the original build; samples are grouped by exact executable and never pooled across builds.

`arms.py` uses the existing controller MemoryGuard, host 1.5 GiB availability floor, power guard and a 28 GiB/zero-swap test scope. Configure fresh output/arm JSON with `FLASH_M0_OUTPUT` and `FLASH_M0_ARMS`; each arm supplies cache size and CLI overrides. The historical helper paths are private lab paths, retained to identify the exact experiment. `run_isolated.py` owns the canonical lease across the whole child run; do not wrap it in a second flock.

An optional `crypto_mask` tests only the unbound kcryptd workqueue's CPU mask. It snapshots the original, writes the requested temporary mask, and restores it in `finally`; each arm records before/during/after and fails if restoration differs. The sysfs change needs sudo. No dm-crypt table, key, persistent workqueue setting or boot setting is touched.

One first QD confirmation arm completed but failed its final guard assertion before the old script serialized metadata. It is retained and excluded. The harness now serializes that failure and tolerates only the brief process/cgroup disappearance race at shutdown; it does not relax RAM, PSI, OOM or swap checks. A fresh independent ABBA replaces that failed confirmation.

`summarize.py` reads existing receipts only. No new measurement runs or weight payload reads occur. All changes remain experiments; production Flash and its model are unchanged.
