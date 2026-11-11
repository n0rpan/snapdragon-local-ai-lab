# Hard-task set (MASTER-PLAN WP-M3)

Purpose: decide O-1. Does Qwen3.8-Flash-Next earn full priority over Ornith-1.5-35B-A3B
(and Qwen3.6-35B-A3B) on hard agentic coding? The owner's rule: **Flash keeps full priority only if it
solves >= 1 task that Ornith fails, or >= 2 more tasks of the set; otherwise it is escalation-only.**
`summarize.py` computes that verdict from the results.

11 synthetic, self-contained tasks in the style of real OpenCode work. Every task is judged by an
independent checker on the **final repository state** (hidden tests the model never sees), never on the
model's text. Deterministic, offline, stdlib only (Python 3, bash/awk, gcc).

| Task | Category | What makes it hard |
|---|---|---|
| t01-inventory-reservations | multi-file feature | one spec, five modules (store, persistence with old-format load, report columns, CLI) |
| t02-calc-failing-tests | debugging from failing tests | three root causes in three files; symptoms point elsewhere |
| t03-fair-allocation | algorithm with edge cases | weighted water-filling with exact Fractions and a three-level tie-break |
| t04-discount-refactor | refactor with tests | extract drifted duplicate logic, keep checkout semantics, add a rule once |
| t05-compute-cache-race | concurrency bug | per-key single flight, no global lock during compute, exceptions, invalidate in flight |
| t06-logfmt-parser | parsing bug | exact error positions, escapes, round-trip property |
| t07-config-audit | read a codebase, answer a question | reachability + KeyError semantics across 7 modules; answer in ANSWER.json |
| t08-burst-detector-perf | performance with a measurable target | O(n^2) -> O(n log n) with tie and boundary semantics intact; 200k events < 4 s |
| t09-csv-report-shell | shell / data wrangling | three CSV dialects (BOM, CRLF, decimal comma, quoted commas), dedupe, exact decimals |
| t10-c-tail-ringbuffer | C memory bugs | off-by-one heap overflow, ring wrap order, dropped last line; checked under ASan/UBSan |
| t11-build-planner | graph algorithm | exact DFS post-order, cycle path, staleness rules, 3000-deep chain (iterative) |

Calibration target: a strong 35B MoE solves roughly 40-60%. The hidden tests check only what TASK.md and
the project's own docs/docstrings specify.

## Layout

```
tasks/<id>/task.json        id, title, category, language, check_timeout_s
tasks/<id>/TASK.md          the prompt OpenCode receives (verbatim, stripped)
tasks/<id>/project/         the fixture; a FRESH copy is made for every run
tasks/<id>/hidden/          check.sh + hidden tests/data (never mounted into the agent namespace)
reference/<id>/             reference solution: changed/new files only (never copied into a run)
check_task.py               checker: copies a project dir, runs hidden/check.sh in offline bwrap, timeout
validate.py                 gate: checker FAILS on the untouched fixture and PASSES on fixture+reference
run_hard_task.py            one task x one alias through real OpenCode against the running controller
hard_inside.py              the in-namespace caller (normal13 inside.py pattern + token/turn accounting)
run_set.py                  all tasks x aliases, resumable, results.jsonl
summarize.py                Markdown table + O-1 verdict
test_hard_tasks.py          model-free tests of the runner
```

## Harness (reused, not new)

`run_hard_task.py` is the protected normal13 controller client (`benchmarks/classic/controller_normal13.py`)
with a different fixture. It imports and reuses: the pinned OpenCode binary and its SHA check
(`normal13/opencode-binary.expected.json`), the same bwrap namespace (`controller_normal13.sandbox_command`:
`--unshare-all`, clearenv, private HOME/XDG, engine/model binds stripped), the same OpenCode config
transformation (local provider only, temperature 0, cache_prompt, thinking off, title agent off), the owner-only
Unix-socket `loopback_bridge.py`, and the unchanged `inside.py` (`capture`, `api`, `Completion`). Like
`controller_normal13.py` it starts no native server and takes no inference lease: the production controller
(`http://127.0.0.1:8092`) owns model load, lease and memory guard; the alias selects the model.

Differences: the prompt is `tasks/<id>/TASK.md`; the wall-clock cap is per run (`--cap-seconds`); OpenCode's
context/input/output limits for the alias are read (read-only) from `~/.config/opencode/opencode.json` so each
model gets its production limits (recorded in summary.json); and the checker runs **after** the agent namespace
exits, on the host side in a second offline bwrap, against a copy of the final `project/`.

Per run, `<output>/summary.json` records: `solved` (checker passed and harness clean), `solved_within_cap`,
`timed_out`, `task_seconds` (agent wall time), `turns` (LLM requests = OpenCode step_finish events),
`tokens` {input, output, reasoning, cache_read, cache_write} summed over turns, `tokens_total`,
`tool_calls` per tool, `opencode_exit`, `complete`, the checker verdict and seconds, the OpenCode SHA and a SHA
of the task directory. `check-output.txt` holds the checker transcript; `agent-stdout.log` the OpenCode events.
`harness_ok == false` means an infrastructure failure (bridge, namespace, controller): not a model failure; re-run.
Do not delete such rows: results.jsonl is append-only. Re-running the same `run_set.py` command re-runs every
(task, alias, rep) without a `harness_ok: true` row (model failures with `harness_ok: true` stay done), and
`summarize.py` counts only the last `harness_ok: true` attempt per key and lists failure rows separately.

Scoring: `solved` judges the final repo state even when the cap was hit (`timed_out` is reported beside it).
For O-1 use `solved`; with `--repeat n` a task counts as solved for a model on a strict majority.

## Commands

Model-free (any time, no lease):
```bash
cd benchmarks/hard-tasks
python3 validate.py --out /tmp/hard-validate.json      # every task: untouched=FAIL, reference=PASS
python3 -m unittest test_hard_tasks -v
```

One task (controller running, QUALITY window):
```bash
python3 run_hard_task.py --task t03-fair-allocation --alias ornith-35b-a3b-local \
  --controller http://127.0.0.1:8092 --cap-seconds 900 --output /abs/fresh/dir
```

The set (resumable; re-running the same command skips finished rows):
```bash
R=${RESULTS_DIR:-results}/hard-tasks-$(date +%Y%m%d)
python3 run_set.py --root $R --alias ornith-35b-a3b-local=900 --alias qwen3.6-35b-a3b-local=900 \
  --alias qwen3.8-flash-next-local=2400
python3 summarize.py --root $R --out $R/HARD-TASKS.md
```

## Adding a task

Copy the layout above. Keep `project/` <= 12 KB (the model has ~6K tokens of free context after OpenCode's
system prompt), no gitignored names (`*.log`, `build/`, ...), hidden tests import the project from `$PROJECT`
and data from `$HIDDEN`, `check.sh` exits 0 only on success, and every hidden assertion must follow from
TASK.md plus the project's docs. Then `validate.py --task <id>` must print OK.
