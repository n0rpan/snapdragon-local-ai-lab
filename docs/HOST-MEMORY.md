# Host RAM, disk swap and model limits

The owner approved the following policy on 2026-11-09: give the inference service about 28 GiB of hard RAM; allow inactive desktop/browser pages to swap to disk; never swap the model service. Desktop sluggishness is acceptable. The controller still stops its native child below 1.5 GiB `MemAvailable`, on sustained corroborated full-memory stalls, or on service OOM/swap errors.

## Measurement and chosen cap

The laptop exposes 32,988,520,448 bytes (30.723 GiB) of RAM. Eighteen quiet inference-lease-free samples during the owner's confirmed normal browser use, 15:51:51–16:00:23 UTC, measured a maximum whole-host `MemTotal - MemAvailable` of 5,888,679,936 bytes (5.484 GiB). This includes the OS, Hyprland, browser, Codex CLI/app-server, Claude Code, node/tool processes and terminals; none of their memory was subtracted. The heaviest sampled desktop included 18 Chromium processes, three Codex processes and two Claude processes. This is a measured observation window, not a guarantee about future desktop usage.

A conservative RAM-only proposal was superseded by the owner's explicit preference for a smaller hard desktop allowance plus disk swap. The installed drop-in `$HOME/.config/systemd/user/local-ai-chat.service.d/model-profiles.conf` is:

```ini
[Service]
MemoryHigh=infinity
MemoryMax=28G
MemorySwapMax=0
CPUQuota=1200%
Nice=15
```

`28G` is 30,064,771,072 bytes, leaving 2,923,749,376 bytes (2.723 GiB) outside that service ceiling. The observed 5.484 GiB desktop can exceed this hard allowance; inactive pages may therefore move to swap. `MemoryHigh` no longer adds an early reclaim/throttling zone. `MemoryMax` still includes charged file cache and anonymous allocations: reaching it can reclaim mapped model pages and reduce speed, or cause an OOM if the working set cannot fit. It is a safety ceiling, not extra physical RAM.

The controller owns one load-aware guard per native process and the canonical inference lease. Loading is exempt only from the PSI stop for a bounded startup period; the hard available-memory floor, OOM and own-swap checks remain active. After readiness, high PSI must persist for ten seconds and actual post-ready stall counters must corroborate it. This avoids treating the decaying load-phase PSI average as a new idle failure. Host swap/pageout activity alone never aborts a run. Idle-unload, switching and RAM release passed on all three retained classic profiles before promotion.

The numeric-only [desktop sampler](../tools/desktop_memory_sampler.py) runs every 30 seconds through the normal-use window in `local-ai-desktop-memory-sample.service`. It records host availability/PSI, service memory and application counts; no arguments, conversation text or images. Private output: `~/.local/state/local-ai/desktop-memory-20261109.jsonl`. Revisit the cap if representative usage shows repeated pressure; no automatic cap increase is installed.

## Disk swap with a bounded zswap cache

The former 30.7 GiB logical zram device is disabled by the owned override `/etc/systemd/zram-generator.conf.d/99-local-ai-disk-swap.conf` (`[zram0]`, `zram-size=0`). The vendor `90-omarchy.conf` is untouched. An initial `90-local-ai` override lost the lexical ordering to the vendor file; that attempt was corrected before the transaction was completed.

A dedicated Btrfs subvolume at `/var/lib/local-ai-swap` contains a 16 GiB swapfile made with `btrfs filesystem mkswapfile`, with NOCOW/no compression, file mode 0600 and directory mode 0700. Its single owned `/etc/fstab` line is:

```fstab
/var/lib/local-ai-swap/swapfile none swap defaults,nofail,pri=10 0 0
```

**State on 2026-11-12: zswap is off.** `/sys/module/zswap/parameters/enabled` reads `N` although `local-ai-zswap.service` reported success (compressor and pool size were applied; the final enable did not stick or was reverted, and the old script never read it back). Until the corrected script reads back `Y`, swap on this host is plain NVMe swap with no compressed pool, and nothing below may assume zswap.

Intended setup: `/usr/local/libexec/local-ai-zswap-setup` and the enabled `/etc/systemd/system/local-ai-zswap.service` enable zswap at runtime, with zstd, `max_pool_percent=5`, and the shrinker enabled, before `swap.target`. The repository copies in `packaging/zswap/` read every value back and fail the unit when zswap is not enabled, so a failure shows as a failed unit instead of `active (exited)`. Five percent bounds the compressed pool to about 1.54 GiB and leaves more of the requested hard desktop allowance available; cold pages can write back to NVMe. The existing `zswap.enabled=0` boot argument is unchanged: the service enables it after the kernel starts. The old zram device was stopped/reset; `/proc/swaps` now lists only the NVMe file. `vm.swappiness=150` is unchanged.

Logical swap used is not the same as bytes physically written to disk: some pages remain in zswap's RAM pool. Safe zswap counters (`stored_pages`, `pool_total_size`, `written_back_pages`) distinguish them. Never read DMA-heap debugfs. The model service has both `MemorySwapMax=0` and a guard that rejects any service swap usage; idle desktop pages can use zswap/NVMe independently.

## Swap-tolerant guard for a preferred 20/18 GiB Flash cache (F127, opt-in)

An explicit Flash cache preference of 20 or 18 GiB is admitted on MemTotal: idle desktop and agent pages are expected to go to swap. Then MemAvailable is the wrong stop signal. In ticket 064 the 1 GiB MemAvailable floor stopped a 20 GiB run while memory PSI was 0.49 % (some) / 0.29 % (full) and a Hyprland IPC round trip took 13.5 ms.

With `oxe_flash.swap_tolerant_guard` set, and only while the engine reports `preferred-swap-tolerant`, the controller's guard stops the engine on:

| Stop | Default |
|---|---|
| Engine's own swap (`memory.swap.current`) nonzero, or `memory.swap.max` not 0 | always |
| OOM in the engine's scope (`oom`, `oom_kill`, `oom_group_kill`) or a global OOM kill (`/proc/vmstat oom_kill`) | always |
| PSI full avg10 ≥ X % for Y s, corroborated by real stalls ≥ X % over the same window | X = 10, Y = 20 |
| PSI some avg10 ≥ Z % for W s, same corroboration | Z = 30, W = 30 |
| MemAvailable + SwapFree below a last-resort floor (no RAM and no swap left) | 1 GiB |

PSI stops wait for a grace window, until the model is ready and at least `grace_seconds` (default 60) after load start: reclaim during load reached about 27 % PSI in 064. Every other mode (auto, 16, 14, a MemAvailable fallback, every non-Flash model) keeps the standard 1 GiB floor. `true` uses the defaults; a dict overrides single values, for example `{"enabled": true, "psi_full_seconds": 30}` (keys `psi_full_avg10_percent`, `psi_full_seconds`, `psi_some_avg10_percent`, `psi_some_seconds`, `grace_seconds`, `last_resort_floor_bytes`). Off by default.

## Verification and rollback

The private controller completed the protected Ornith task with 13/13 independent checks twice: 121.41 seconds cold and 63.15 seconds in the cached second session. The deployed source, default selection, 28 GiB cap and nonwaking status were checked after installation. The promoted controller then passed the largest-resident (Qwen MoE + MTP)16K resource check:15,808 prompt tokens at 92.96 tok/s and 512 generated tokens at 22.36 tok/s, zero global/service OOM kills and zero service swap. Service current-memory peak was29,810,642,944 bytes (27.76 GiB), minimum host availability4,024,258,560 bytes (3.75 GiB), peak 98.0 °C. Early/late128-token decode windows were21.66/22.86 tok/s; there was no within-run decline. This is not a matched pre-swap speed comparison. The browser had closed. The owner explicitly accepted the concurrent Codex/Claude/subagent load as his worst case and closed the browser-open requirement; desktop-pressure validation is therefore passed for that accepted workload. The receipt retains `browser_open_condition_met=false` so it does not imply a browser was present. The owner confirmed the earlier browser/Codex/Claude sample was a worst case and that the 28 GiB cap must remain rather than increasing the reserve. Resource receipt (raw rows `benchmarks/roofline/production-full-context-resource-check.jsonl`, lab notebook, not in the public snapshot).

Before/after system backups are in `/var/lib/local-ai-swap-backups/20261109T161327Z/`. Runtime/config rollback copies are in the private production stage's `promotion-20261109T165115Z/`; its runtime copy already points to the permanent Ornith model path. To roll back:

1. Unload models and stop measured jobs. Check enough available RAM before any `swapoff`; it can need substantial memory.
2. Disable `local-ai-zswap.service`, write `N` to `/sys/module/zswap/parameters/enabled`, and remove only the owned `99-local-ai-disk-swap.conf` override. Reload systemd, regenerate/start the vendor zram swap configuration, and verify it is active before removing disk swap.
3. Turn off `/var/lib/local-ai-swap/swapfile` only when it is safe. Remove only its owned fstab line, then remove the unused swapfile/subvolume and owned zswap service/setup script. Preserve unrelated fstab edits and backups.
4. Restore the backed-up local-ai unit/drop-in/runtime/CLI/OpenCode config as a set, reload the user manager and restart the unloaded controller. Do not restore the obsolete Work-based Ornith model path.

These rollback steps are instructions, not commands to execute during normal inference. Neither swap persistence across a fresh boot nor every future workload is established by configuration checks alone; inspect the enabled service and `/proc/swaps` after the next approved reboot.
