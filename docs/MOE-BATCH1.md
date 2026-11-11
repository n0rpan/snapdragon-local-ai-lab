# MoE production batch 1

Release: controller `moe-batch1-76de3f7965d8`, engine `oxe-server` `b9ad2eb05c33`. Replaces the `649ac806` setup **for Ornith only**: Qwen stays at `649ac806`, 16K context, `--ctx-checkpoints 4`. Ticket 008 qualifies parked sessions for Qwen later, but that is not deployed.
Paths below are placeholders: `<stage>` is the local stage directory (`~/.local/state/local-ai/moe-batch1-<date>/`).

## What changed
1. **Ornith context 16K to 32K** (`context_size` 16384 to 32768, plus the OpenCode `limit.context` and `input` for the Ornith model).
2. **Parked sessions and turn checkpoints.** `--ctx-checkpoints 4` to `8`, and `--session-ram-bytes 2147483648` (2 GiB session budget) on the **Ornith** `oxe_args` only. Before a request would discard 512 or more prompt tokens of the live conversation (title request, `task` subagent, plan/build switch), the server copies the whole conversation state to RAM and restores it as a byte copy later. Checkpoints at message boundaries let an edit or retry resume there instead of at the system prompt. The disk session cache stays off.
3. **Controller request queue** (lab branch `claude/controller-queue`, ticket 020). Inference requests wait for the single backend slot in arrival order, including through a model load. After `LOCAL_LLM_QUEUE_WAIT_SECONDS` (default 120) a request fails with the old 503. `/_local/status` adds `queued_requests`. Only `controller.py` changed.

## Why (measured numbers)
- **32K memory (ticket 008).** The full 32K plus parked-session memory proof is ticket 008, not the short ticket 020 prompts. Peak RSS: Ornith 32K 15.12 GiB, Qwen 32K 19.84 GiB (Qwen is not deployed), no swap, no OOM, within the 28 GiB cap.
- **Live check on the installed release.** Native VmHWM of the engine process was 16.38 GiB (VmSwap 0). The cgroup peak was only 1.63 GiB, because shared file pages (the mmapped model) are charged elsewhere. Do not compare the two numbers directly; the native VmHWM is the one to hold against the cap.
- **32K speed (ticket 008C, warm ABBA).** Ornith decode median of means 22.608 ms/token at 16K vs 22.581 ms/token at 32K (-0.12%, flat; the gate was 1%). All token hashes equal.
- **Parked sessions (ticket 008).** Exactness gates passed for Ornith and for Qwen with MTP depth 2: token ids equal the uninterrupted session and unparked state is byte-identical. A 32K-token session parks as 817 MiB and resumes in 42 ms with 1 new prompt token evaluated. Re-prefilling such a conversation used to cost about 55 to 90 s for 10 to 16K tokens (estimate at about 180 tok/s). Tokens per second does not change. See `P1-promotion-summary.md` (lab handoff notes) for the full table and risks.
- **Queue (ticket 020).** The M2 recording (title on, `task` allowed, oxe `649ac806`) had nine HTTP 503 "request already running" replies: the build agent's first request during the title request and model load, and one of two parallel `task` subagents. OpenCode retried every 5 to 9 s, so that subagent started at 120.1 s instead of 64.7 s, about 55 s late. With the queue the target is zero 503s.

## Install (staged, atomic)
Nothing is installed by staging. `tools/install_staged_release.py` reads `<stage>/manifest.json`, checks that each live file (`runtime`, `unit`, `opencode`) still has the recorded preimage sha256, writes each staged replacement to a temp file in the target directory and renames it into place, then re-hashes. It refuses on any mismatch (including an already-installed file) and never prints file contents.

```
python3 tools/install_staged_release.py <stage>            # dry run (default): verify only
python3 tools/install_staged_release.py <stage> --apply    # install
```
It does not stop or start services: stop `local-ai-chat.service` first, `systemctl --user daemon-reload` and start it after.

## Verify
- Dry run reports every preimage as matching before `--apply`; after it, the live sha256 of each file equals the staged file.
- Runtime audit: `~/.config/local-ai/llm/runtime.json` has zero retired refs (also recorded in `DELETION-MANIFEST-20261109.md`).
- `/_local/status` shows `queued_requests`; three overlapping chat requests all return 200 and finish in arrival order; with `LOCAL_LLM_QUEUE_WAIT_SECONDS=5` and a long first request, the second gets 503 after about 5 s.
- Re-recorded M2 session through the new controller: zero 503s, and `session_replay.py info` shows the demanded-overlap line.
- Memory stays under the 28 GiB cap at 32K with a parked session.

## Rollback (one command)
```
python3 <stage>/rollback/rollback.py
```
The `latest-moe-batch1` pointer in the stage area names the current rollback set.
Stops `local-ai-chat.service`, restores the three original files from `<stage>/rollback/*.original`, reloads systemd and starts the service. The restore copies contain the original OpenCode config, which includes secrets: keep the stage directory local and never commit it.

## Qualification receipts
Verdicts from `<stage>/qualification/` (summary `passed: true`; no raw content here):
- Controller FIFO: PASS. Three overlapping requests returned 200 and finished in order 0, 1, 2, with `queued_requests` 2 while the first ran.
- Bounded wait: PASS. With `LOCAL_LLM_QUEUE_WAIT_SECONDS=5` the second request got 503 at 5.00 s.
- Protected OpenCode task: PASS, 13/13 checks in 56.7 s.
- Memory: no kernel OOM, no swap; native VmHWM 16.38 GiB, cgroup peak 1.63 GiB (see above).
- Still to come: the re-recorded M2 session (020B) 503 count and D6 line, and the three fixed-workload replays (`tools/run_batch_replays.sh`).
