# Guarded fresh build and four native software checks

The first actual public Flash source acquisition [passed](../benchmarks/results/bootstrap-flash-source.json). A compiler run, four software checks and complete model/runtime qualification are separate gates. This page describes the reviewed executors; it does not turn their source-only tests into native acceptance. No command below installs a runtime or enables a profile.

## Dependencies and isolation

Install normal distribution packages for Git, [CMake](https://cmake.org/download/), [Ninja](https://github.com/ninja-build/ninja), [GCC/G++](https://gcc.gnu.org/), Python 3, OpenMP/libgomp, OpenSSL development/runtime files and util-linux `unshare`. Recorded historical versions are GCC 16.1.1, CMake 4.4.4 and Ninja 1.13.2. The explicit Oryon target must be supported. Paths come from the reviewed job's PATH, not a particular user's virtual environment.

One local preflight failed because that laptop's Python environment had no `cmake` launcher in its `bin` directory; the actual CMake lived in the package's `cmake/data/bin` directory. A local PATH override resolved it. **Fresh users need an installed CMake command, not this private environment layout.** A second preflight rejected CMake's normal expansion of `cc`/`c++` to full paths. Executor revision 2 accepts that expansion only when it is the same executable inode as the requested selector resolved from the job PATH, and records effective paths. All other option equality remains exact. Both failures remain failed receipts; neither was a compiler failure or a successful build.

The build copies the passed source tree into a fresh workspace using independent files, audits Git metadata/hardlinks, verifies the exact source again and configures out of tree. It uses Release, one build job and the [explicit source recipes](../bootstrap/source-recipes.json). The Flash recipe preserves repack/KleidiAI OFF and CPU-only streaming. The classic recipe preserves FP16+BF16, KleidiAI/repack/OpenMP/OpenSSL ON; it is not the earlier MTP recipe.

Both configure and compile run under `unshare --user --map-root-user --net`. This denies even CMake `file(DOWNLOAD)` network access; `FETCHCONTENT_FULLY_DISCONNECTED` also stays ON. User/network namespace support is a prerequisite and absence fails closed. No host network setting is changed. Dependencies are acquired explicitly before that namespace, never through an uncontrolled latest fallback.

Classic builds prefetch and SHA-check KleidiAI 1.32.0 and the exact b12189 prebuilt UI. The UI is extracted to the copied source's `tools/ui/dist/index.html` priority path; upstream returns before npm/Hugging Face branches when that path is present. `LLAMA_USE_PREBUILT_UI` remains ON and the embedded browser UI is preserved. Merely setting BUILD_UI OFF would not prevent upstream downloads. [UI metadata pin](../bootstrap/ui-pin.json), [KleidiAI and UI acquisition pins](../bootstrap/source-recipes.json). Archive extraction refuses links/path escapes/special files and bounds member count and expanded size.

## Run a fresh Flash build after source PASS

Use a new build work directory and the actual SHA256 of your passed `source-result.json`. The source-manifest hash below is the frozen reviewed recipe. The build-manifest hash is revision 2, including compiler-selector normalization. Root must register the unit with its external power/status guard before scheduling it; do not overlap a second heavy job.

```sh
systemd-run --user --wait --collect \
  --unit=local-ai-bootstrap-build.service \
  -p CPUQuota=200% -p MemoryHigh=3G -p MemoryMax=4G \
  -p MemorySwapMax=0 -p TasksMax=160 -p RuntimeMaxSec=2700 \
  -p TimeoutStopSec=15 -p KillMode=control-group -p SendSIGKILL=yes \
  -p UMask=0077 -p LimitCORE=0 \
  --setenv=PYTHONPATH="$PWD/src" --setenv=PATH="$PATH" \
  --working-directory="$PWD" \
  /usr/bin/python3 -B -m snapdragon_ai_lab.bootstrap_build --run --engine flash \
  --manifest-sha256 d9a79a85e92eccaf63d4ec2e6c420f44ba29906bfe954daf9c6110998538b79c \
  --source-manifest-sha256 fe5710156dcc34a0cca5af7fa5c40962258d9c41106ca23f04e3b85836a65182 \
  --source-receipt-sha256 YOUR_ACTUAL_SOURCE_RECEIPT_SHA256 \
  --source-work /path/to/passed-source-work \
  --work /path/to/new-build-work
```

The guard requires 8 GiB available RAM at build start and a live 2 GiB reserve, canonical exclusive lease, AC/30% start/20% cutoff, sustained-discharge monitoring and own zero swap. CPU/memory/tasks/deadlines are checked against the dedicated user unit. New global page-outs stop the job; existing global page-ins are recorded. Parent deadline is 2600 seconds inside the 2700-second unit. Combined logs are capped at 16 MiB, all children must be reaped, and actual cgroup peak/events are captured. There is no retry/fallback loop.

A passed `build-result.json` records actual tool versions, effective compiler paths, exact options, dependency identities and new binary SHA256. It remains **`qualified:false`, `models_run:false`, `profiles_enabled:false`**. Build targets include native test executables, but building them does not execute their assertions. Generated version metadata and new toolchains can change binary hashes; do not copy a historical hash into a new profile.

## Run exactly four model-free native checks after build PASS

The next reviewed module is [bootstrap_checks.py](../src/snapdragon_ai_lab/bootstrap_checks.py). It executes only `bmoe_config_test`, `bmoe_params_test`, `bmoe_mapping_release_test` and `bmoe_file_reader_test`. These do not initialize GGUF models. It binds the exact passed build receipt, verifies the CLI/server binary pair and requires explicit hashes for all four fixed test filenames. Test binaries are checked before and after execution. A nonzero exit, skip exit or SKIP/NOT RUN output fails. Full bounded output is inspected, not only a token counter.

After the build unit has stopped successfully, create an owner-only JSON mapping those four filenames to their actual SHA256 values. Record its SHA256 and the actual build-result SHA256. Pass them explicitly:

```sh
systemd-run --user --wait --collect \
  --unit=local-ai-bootstrap-sources.service \
  -p CPUQuota=100% -p MemoryHigh=768M -p MemoryMax=1G \
  -p MemorySwapMax=0 -p TasksMax=64 -p RuntimeMaxSec=900 \
  -p TimeoutStopSec=15 -p KillMode=control-group -p SendSIGKILL=yes \
  -p UMask=0077 -p LimitCORE=0 \
  --setenv=PYTHONPATH="$PWD/src" --working-directory="$PWD" \
  /usr/bin/python3 -B -m snapdragon_ai_lab.bootstrap_checks --run \
  --manifest-sha256 7eccafd2383d35a85bc0a6d38bf094e2e4e978f8abe6960d18c180d36fe6a612 \
  --build-work /path/to/passed-build-work \
  --build-receipt-sha256 YOUR_ACTUAL_BUILD_RECEIPT_SHA256 \
  --test-hashes /path/to/private-test-hashes.json \
  --test-hashes-sha256 YOUR_TEST_HASHES_JSON_SHA256 \
  --work /path/to/new-native-checks-work
```

This deliberately reuses the inactive source-kind unit and its CPU 100%/1 GiB/900-second guard; do not run source acquisition concurrently. The result is `native-checks-result.json`, with per-test exit/hash/time, resource/cleanup facts and no model qualification. The three preparer tests used fake executable bytes and an injected runner only; **actual native gate completion must come from the scheduled job**.

After all four pass, architecture byte-identity tests, real default MoE/Q8 and optional Dense/Flash/Q4 qualification, OpenCode/browser/image workflow, private installation/rollback and the right-side widget still need their own clean-install acceptance. [Complete finish gate](BOOTSTRAP-PARITY.md).
