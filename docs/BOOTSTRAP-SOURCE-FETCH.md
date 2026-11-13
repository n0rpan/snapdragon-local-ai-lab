# First live bootstrap step: source fetch and exact verification

This is a frozen, runnable **source-only** step for root review. Preparation ran nine tiny synthetic tests and help/AST checks. It did not fetch sources, compile, download a model, install anything or start a service. It is not clean-install parity or a fresh binary qualification.

The default engine is Flash: anonymously fetch public BigMoeOnEdge `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` and public llama fork `7ac98f4a8089a36c0581b7c6e8973417a8fa175c`, apply the published demand-mapping patch to the standalone sub-checkout's index, then verify the complete 5,253-blob/raw-source inventory. No private commit, Git author configuration, fork commit, remote configuration, GitHub login or API token is needed. `git fetch --depth 1 --no-tags HTTPS_URL PINNED_SHA` avoids publication/remotes entirely.

The source executor uses an isolated HOME with system/global Git configuration and credential helpers disabled, no terminal prompts and no lazy/LFS payload fetches. It does not invoke CMake or any compiler. Only selected submodules are fetched. All writes stay in the explicit new work directory except acquiring the canonical owner-only inference lock; on a genuinely new user the lock is created once without replacing an existing inode. A mismatched legacy lock is refused.

## Reviewed invocation — not executed by the preparer

From the project checkout, choose a fresh work directory. The root supervisor must register this unit with its external power/status monitoring before launch. Keep one heavy job at a time.

```sh
systemd-run --user --wait --collect \
  --unit=local-ai-bootstrap-sources.service \
  --property=CPUQuota=100% \
  --property=MemoryHigh=768M --property=MemoryMax=1G \
  --property=MemorySwapMax=0 --property=TasksMax=64 \
  --property=RuntimeMaxSec=900 --property=TimeoutStopSec=15 \
  --property=KillMode=control-group --property=SendSIGKILL=yes \
  --property=UMask=0077 --property=LimitCORE=0 \
  --setenv=PYTHONPATH="$PWD/src" \
  --working-directory="$PWD" \
  /usr/bin/python3 -B -m snapdragon_ai_lab.bootstrap_sources \
  --run --engine flash \
  --manifest-sha256 fd5a3bb36a2eaca099eb26b0359ef8c3b32d25ba74535dbf3a61bedea0975223 \
  --work /tmp/local-ai-public-flash-source-v1
```

The unit and caps are verified by the process. The parent deadline is 840 seconds inside the 900-second unit. AC and at least 30% battery are required at start; AC loss, unverified telemetry, battery at/below 20%, sustained observed discharge, less than 2 GiB actual available RAM, own swap or new global page-outs stop the job. Existing global page-ins are recorded and permitted. The parent owns the canonical lease until every child is reaped; cleanup retains it through TERM/KILL. Persistent combined command logs are capped at 16 MiB. Final actual cgroup peak and memory/pids-limit events are recorded; max/OOM/pids-limit events reject success.

A successful result is `source-result.json` beside `source/` and the bounded `commands.log`, with `passed:true`, `source_verified:true`, **`compiled:false, qualified:false`**. An unchanged complete source inventory is only a source receipt. Failed/partial work remains visible; a source retry uses a new directory. Model payload download resume is a different subsequent action.

`--engine classic` and `--engine images` select their public pins and exact patches/submodules from [source-recipes.json](../bootstrap/source-recipes.json). They do not acquire weights, KleidiAI or the classic prebuilt UI in this step. The next build step must prefetch those separately and prevent all uncontrolled configure/build downloads.

## Remaining build inputs and next step

Classic parity requires the **later** FP16+BF16/KleidiAI/repack/OpenMP/OpenSSL recipe, not the earlier MTP binary. The catalog pins KleidiAI v1.32.0 (4,701,556 bytes, SHA256 `a48d373e7c5d721a0aecbf5ddaf0bdf245bc8064e6b84086f13db0e102b553a6`) and the qualified UI archive (3,250,656 bytes, SHA256 `be141c1f3ddb6697e9254e3c97cbec897951c01805764f6d77fee5ae83787bfc`). `BUILD_UI=OFF` alone does not prevent the upstream UI download. The versioned UI URL is a candidate until its guarded transfer matches the expected bytes; there is no latest fallback. Preserve the browser UI feature rather than silently disabling it.

After a passed source receipt, root can review a separate CPU 200%, 4 GiB, zero-swap, one-job, finite build. No build or model action follows automatically. [Full parity finish gate](BOOTSTRAP-PARITY.md).

The prepared default chat model is Ornith. The source-fetch command above explicitly selects the historical Flash source recipe; it does not choose the default chat runtime or enable a model. There is no one-command Rust-engine install in this bootstrap.
