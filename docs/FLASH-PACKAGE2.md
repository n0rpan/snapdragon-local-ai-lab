# Flash package 2, installed 2026-11-11

> **Superseded engine/order:** F123 is installed as of 2026-11-11: commit `4b6ab92`, server `001e0b85…`, explicit `--flash-pf-order 2`. See [Flash F123](FLASH-F123.md) and its [portable evidence](../benchmarks/results/flash-f123-20261111.json). The package2/order0 results below are retained unchanged as historical evidence.

Flash now loads with a 14 GiB expert cache on the real desktop. The previous 16 GiB profile required 26.04 GiB available and refused at 25.6 GiB; the new measured admission is 24.715 GiB. The engine remains full quality: all experts and native routing are retained. [Evidence](../benchmarks/results/flash-pkg2-20261111.json).

Installed engine: `8e5240a` / server `a6912dd3…`; context 16K, 8 decode threads, 11 batch threads, IO CPU8, plain expert sidecar. The 29 GiB scope, zero model swap and 1 GiB host floor are unchanged. The long-prompt guard is on. Prefill ordering is explicitly off (`--flash-pf-order 0`); lookahead and fused phases are off.

The new binary includes the tool-parameter parser fixes and long-prompt admission. The legacy controller gains only the two 14 GiB cache allowlists. Its other 18 files, other model profiles, default selection, OpenCode settings and UI are preserved.

## Matched decode control

Same boot, same frozen 75-token p2-prose request, explicit 14 GiB throughout. Each arm generates 64 tokens four times; the first run is dropped.

| B A A B | Median ms/token |
|---|---:|
| New, order0 | 80.425 |
| Old e16dfac9 | 80.164 |
| Old e16dfac9 | 80.813 |
| New, order0 | 85.077 |

Pair means are 82.751 vs 80.488 ms/token: 2.81% slower, within the frozen 3% limit. This is not a speedup claim. Earlier full numeric/tool qualification runs had different cache histories and are not this control.

Resident-first prefill ordering initially showed a 4.4% regression. An expert-cache replay attributed it to eviction of hot decode experts after short prefill; disabling it restored the accepted matched result. Preserving its repeat-prefill read saving without that decode cost is F123, a separate follow-up.

The optional A L L A / S S screen also used order0 and 14 GiB. Plain averaged 84.429 ms/token, lookahead 87.047, and lookahead plus fused phases 88.838. Neither optional setting earned promotion. This does not generalize the earlier 18 GiB fixed353 lookahead win to short p2 requests.

## Qualification and rollback

354 teacher-forced rows are byte-identical to the qualified fast path; mean KLD against the exact path is 0.0006545, below 0.0011. Self-teacher is 192/192. Reference wording differences remain near-ties (maximum gap 0.3098 nats).

The protected task passed 13/13 in 184.84 seconds on the final order0 argv. Peak RSS was 22.23 GiB, scope peak 22.31 GiB and host minimum 3.94 GiB, with no OOM or model swap. Its time is one qualification trajectory, not a matched task-speed gain.

The staged-controller capture and live post-install suite passed chat, tool call/follow-up, 4K context and cancellation. The controller was left unloaded and CPU8 QoS restored to zero. The old engine and controller files remain available.

One-command rollback: `~/.local/share/ai-stack/rollbacks/latest-flash-pkg2/rollback-live.sh`. The guarded script restores the pre-upgrade unit/runtime and restarts the service; it refuses stale or tampered stages. Per-model fallback plans were rebound to preserve the other installed profiles.

The binary's 18/16/14 auto-selection passed no-load checks, but production starts with the newly measured 14 GiB tier. Larger new-package production peaks and the remaining 5500-token guard/control pair are not claimed. F118 measured peak external growth at 0.565 GiB and proposes a 0.80 GiB allowance; see the measured growth analysis below.

## Harness growth (F118)

[Recorded memory samples](../benchmarks/results/flash-harness-growth-20261111.json) separate loaded-model memory growth from the additional OpenCode/desktop/agent growth during the protected task. Across the failed 18 GiB, passed 16 GiB and final package2 14 GiB runs, the largest external growth envelope was **0.565 GiB**. The proposed external allowance is **0.80 GiB**, replacing the earlier 1.40 GiB full-RSS estimate in a future test. The final 14 GiB model itself grew by 3.47 GiB of cgroup charge; that cost remains separate.

The conservative rounded admission estimate for 16 GiB becomes **26.125 GiB**. A second estimate, based on the final 14 GiB run's observed host consumption and an extra 0.25–0.50 GiB margin, gives **25.37–25.62 GiB** and projects roughly 1.94 GiB remaining at 16 GiB on that same starting desktop. These estimates differ because charged/shared/reclaimable pages and global MemAvailable are different measurements. They justify a guarded 16 GiB trial, not an automatic admission-policy change. Production remains at 14 GiB; neither a new-package 16 GiB nor 18 GiB tier is qualified by this analysis.
