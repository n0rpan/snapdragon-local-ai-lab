# Benchmark methods, exact models and result scopes

Start with the [per-model before/after tables](BEFORE-AFTER.md). They distinguish finished tasks from throughput screens and show what changed for the core MoE, dense 27B, Z-Image Q4_K/Q8_0 and SSD-backed Flash-Next. Exact publisher repositories, revisions, filenames, byte sizes and all shard SHA256 identities are in [model-artifacts.json](../dependencies/model-artifacts.json). Model weights, raw generated answers, user paths and credentials are excluded.

## Choose the correct evidence level

| Evidence | Measured examples | Suitable claim |
|---|---|---|
| Same-weight runtime pair | Image repack off/on; MTP pool 1/3; Flash IO workers and overlap | Improvement on the specified fixed workload, with stated repeat/power limits |
| Cache/lifecycle pair | Dense resident-prefix reuse; full unload/reload restore | Faster reuse on that prefix and lifecycle; not kernel acceleration |
| Different files/models/quants | Ordinary versus MTP MoE; dense versus MoE; Q4_K versus Q8_0 images | Model/profile selection, including correctness/quality tradeoffs |
| One-response coding screen | Fixed complete 353-token Flash code; two-function portable screen | Bounded code-generation behavior, not a multi-turn agent qualification |
| Full protected agent repair | Three implementation modules, 12 protected tests, finished final response | Complete task result within the recorded deadline and runtime conditions |
| Normal installed workflow | Provider routing, planning/build tools, unload/reload, idle and image handoff | Only the exact lifecycle/routes actually verified; separate from isolated inference |

The final isolated Flash repair completed in 1175.218 s total, with 12/12 protected tests and a completed final response. Its original checker record remains false because it expected the controller spelling `unloaded` while the native server returned `empty`; a new strict source-backed reconciliation and fresh offline oracle corrected that single predicate without rerunning inference. The original and derived hashes, truthful server exit -15, raw peaks and separate admission-rounding review are in [flash-next-final-agent.json](../benchmarks/results/flash-next-final-agent.json). The installed normal Flash planning/image workflow also passed separately in 238.448 s, including idle RAM recovery, switching, Q8 PNG checks and chat exclusion. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json) A failed earlier 1800 s Flash session is never converted into a completed baseline or speedup ratio.

## Reproduction record required for a comparison

Record the exact artifact SHA as well as its publisher/revision/quant. The dense selected file is DevQuasar Q4_K_M, not Unsloth UD-Q4_K_M; the selected MoE uses the MTP repository's different weights. “FP16” in historical dense build labels does not mean the Q4_K_M weights became FP16. Flash UD-IQ3_XXS is a mixed tensor layout, not uniform three-bit storage. Q4/Q8 image labels refer to the diffusion weights; the shared text encoder and VAE are separately pinned.

Also retain engine commit/patch/binary identity, CPU backend/compiler features, context/logical batch/microbatch, decode/batch/IO threads, expert cache/routing/speculation, THP policy, prefix-cache state, image dimensions/steps/seed/CFG/attention/VAE settings, resource quotas and available-memory guard. Count cold load, priming, prompts, generated output, tool calls, retries, final response, independent checks and cleanup according to the row's stated timing boundary. Preserve failures and checker-policy rejections.

Interleave repeat trials under matched power/thermal/cache conditions before generalizing. A battery reserve guard is not charger-controlled power, sampled sensor maxima are not equivalent temperatures, and process RSS is not interchangeable with cgroup charge or unique physical RAM. Flash overlap's aggregate lane IO time is not wall wait: use mode-selected flash-wait counters and retain the compute residual separately. Never add or multiply unrelated component speedups into a total gain.

## Portable two-request screen

This source-preview runner sends the two fixed prompts in [`code-screen-v1.json`](../benchmarks/fixtures/code-screen-v1.json) to an existing local OpenAI-compatible endpoint. It checks interval merging and port parsing against the fixture's independent assertions. It does not start, install, select, unload or otherwise manage a model.

Requirements: Linux, Python 3.9 or newer installed under `/usr`, and Bubblewrap (`bwrap`) with user namespaces available. The software tests use a tiny fake HTTP server and synthetic functions; they do not load models:

```sh
python -m unittest discover -s tests -p test_run_synthetic_screen.py -v
python tools/run_synthetic_screen.py --help
```

Importing the runner, running it without `--run`, and requesting help perform no benchmark work. To execute the screen against your selected local profile:

```sh
python tools/run_synthetic_screen.py --run \
  --endpoint http://127.0.0.1:8092/v1 --model qwen-local \
  --timeout-seconds 900 --max-tokens 1024 --seed 42
```

The only accepted endpoint form is `http://127.0.0.1:PORT/v1`. The client posts directly to `/v1/chat/completions`, ignores proxy environment settings, rejects redirects and never reads credentials or conversation history. For a backend that requires a key, explicitly name an already populated environment variable with `--key-env LOCAL_SCREEN_API_KEY`. The value is neither a command argument nor part of saved requests, summaries or logs. The checker starts with an empty environment.

Run one heavy workload at a time and keep the laptop adequately powered. The existing backend/controller owns its inference lease and resource policy. **The client deliberately does not acquire that lease**, which would deadlock a managed endpoint waiting for the same lease. No model-side CPU/RAM limit, power guard or memory recovery check is supplied by this client. Closing a timed-out HTTP connection is not proof that a backend cancelled generation. Use the backend's normal controls if cleanup is needed.

## Bounds and evidence

Requests are serialized, temperature is fixed at zero, and the seed/output-token limit and request deadline are recorded. Each request has an absolute deadline, a maximum 256 KiB response and a maximum 64 KiB generated source. There are no retries. The client records only allowlisted finite numeric usage/server timing fields; it never copies arbitrary response metadata or error bodies into the summary. Nonstreaming HTTP wall time includes whatever loading, queuing, prompt processing and generation the server performs; it is not time to first token or a separated cold-load measurement.

Each run creates a fresh owner-only directory (0700, files 0600) under the system temporary directory, or an existing parent supplied with `--output-parent`. It retains the two fixed synthetic requests, original answers, extracted source, protected assertions and `summary.json`. Treat that directory as private and choose a persistent parent if you need to keep it; temporary-directory cleanup is operating-system policy. Only the summary is intended for review/export. The CLI reports its path and a pass/fail label, without displaying generated code.

Generated source must consist of plain functions; ordinary single-underscore helper and local names are allowed. Ordinary builtin annotations (`str`, `int`, `list[int]`, `tuple[int, ...]`, unions with `None`, and inert string references) are accepted and stripped before execution. Calls or attribute expressions in annotations, imports, top-level execution, decorators, dynamic evaluation, introspection, dunder names and private attributes are rejected. Common builtins include iterators (`iter`, `next`, `map`, `filter`), numeric helpers, containers and the relevant exceptions. This remains a restricted function screen; valid general Python solutions can exceed its accepted subset. Such checker-policy rejections are reported as `checker_status: unsupported_source`, separately from `assertion_failed`, runtime errors and request/sandbox errors. Do not count an unsupported-source result as a failed assertion or model-quality verdict. Restrictions are additional checks, not a replacement for the sandbox.

**Generated code is executed only inside Bubblewrap.** The sandbox has separate network/PID/user namespaces, no host home or inherited environment, read-only `/usr` and runtime libraries, read-only candidate/assertion/checker mounts, and a private temporary filesystem. The trusted checker sets a two-second CPU limit, 256 MiB address-space limit, zero file-output size, and bounded descriptor/process counts before executing source. The parent imposes a five-second wall deadline and a 16 KiB combined stdio bound, then kills and reaps its own checker group when necessary. Sandbox setup errors fail the check; there is no host-execution fallback.

The summary explicitly marks this as a **two-request screen**, with agent qualification, separate cold-load measurement and RAM recovery verification all false. Passing does not establish a working OpenCode agent, broad coding quality, long-session stability or model superiority. Compare repeated trials with matching backend/model/configuration, power, cache and thermal conditions. Keep failures and count checker restrictions separately from incorrect answers. Complete protected multi-file repairs and normal lifecycle checks remain separate release gates.

The checker accepts one outer Python/py Markdown fence and records `source_format`; it does not relax any algorithm assertions or execute prose. Original answer text is retained privately. The first real MoE screen passed 1/2 cases after this normalization: the other answer accepted Unicode digits. A separate normal OpenCode repair passed the protected port tests in 177.99 s. See [direct API screen](../benchmarks/results/portable-moe-screen.json) and [normal agent result](../benchmarks/results/normal-moe-port-opencode.json). The original pre-normalization failure remains preserved; the correction rechecked saved responses offline without repeating inference.

The separate Flash-Next complete-code screens use one fixed 353-token answer with
independent sandboxed assertions and process/RAM recovery checks. See the
[demand-based startup split](FLASH-NEXT-SPLIT.md) and
[SSD concurrency and overlap comparisons](FLASH-NEXT-SSD-SCHEDULING.md).
Those code screens alone establish bounded code-generation and lifecycle observations. The later complete protected agent repair has its own separate evidence above; broad model quality and normal installed routes require their own checks.
