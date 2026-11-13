# Selected, guarded model acquisition

This executor is prepared and model-free tested; no model bytes were downloaded by the preparer. It is a separate step from source acquisition/build and from runtime qualification. A verified weight file is not an enabled model profile.

Default selection is exactly four artifacts: Ornith MTP MoE, Q8 diffusion, its text encoder and compatible VAE. Their pinned total is 26,355,107,676 bytes. It does not download every model in the repository. `--only ID` narrows that selection; optional flags explicitly add Dense, Q4 images, all four Flash shards or Arete. All optional assets remain unqualified until their complete local acceptance passes. NPU/proprietary bundles, voice, VL and RAG are not silently acquired or enabled.

[Model identities and publisher links](../dependencies/model-artifacts.json), [dependency/license inventory](DEPENDENCIES.md). Each request uses the exact public repository/revision/file/size/SHA, with no SDK token, cookie store, user authentication, latest fallback or model-content logging.

## Reviewed invocation

Schedule only after other heavy jobs finish and the canonical lease is free. Root registers the dedicated unit with its external power/status guard. Choose an owner-only cache directory outside the Git checkout; keep that same directory to resume.

```sh
systemd-run --user --wait --collect \
  --unit=local-ai-bootstrap-fetch.service \
  -p CPUQuota=100% -p MemoryHigh=768M -p MemoryMax=1G \
  -p MemorySwapMax=0 -p TasksMax=64 -p RuntimeMaxSec=3700 \
  -p TimeoutStopSec=15 -p KillMode=control-group -p SendSIGKILL=yes \
  -p UMask=0077 -p LimitCORE=0 \
  --setenv=PYTHONPATH="$PWD/src" --working-directory="$PWD" \
  /usr/bin/python3 -B -m snapdragon_ai_lab.bootstrap_download --run \
  --manifest-sha256 0991981841bdb0a289e65708ea753b9a3428d0f279ac72358b5d72a4ba348ff3 \
  --work /path/to/owned-bootstrap-asset-cache
```

The internal guard requires the dedicated bounded unit, canonical exclusive lease, actual 2 GiB available RAM, own zero swap, verified AC/30% start/20% cutoff and no sustained discharge or new global swap-outs. Existing global swap-ins are recorded and permitted. The 3600-second attempt budget is inside the 3700-second unit. HTTP open/read have a five-second wall watchdog plus socket timeout, with a signal armed only during network waits. No profile, service, panel layout or other model cache is changed.

Disk preflight requires the outstanding selected bytes plus 1 GiB headroom, accounting for matching durable partial checkpoints. Every path component must stay inside the owned cache without symlinks. Existing final files are verified and preserved, never overwritten. Parent-side hashing and bounded JSON work happen in the same scheduled process; file data streams in 1 MiB chunks.

## Resume and optional selection

Rerun the same command after a stopped attempt with the same cache path. Partial data/metadata use owner-only files, URL/size/SHA identity, fsync checkpoints every 32 MiB or five seconds, prefix rehash and truncation of only an unsynced tail. Resumed transfers require a matching HTTP 206/Content-Range; a 200 response is never appended. Completed cached bytes receive a separate full SHA check before atomic create-only publication. Signed redirects stay HTTPS and are not persisted.

Append selections to the same module command, keeping its `--run`, manifest and cache arguments:

```text
--image-model z-image   # opt-in: replaces the default FLUX.2 klein set with Z-Image Turbo Q8 + text encoder + VAE
--include-q4 --only z-image-turbo-q4
--include-flash --only qwen38-flash-next-ud-iq3-xxs
```

The Flash group selects all four exact shards. Without `--only`, an include flag adds that optional selection to the default four assets. These are acquisition choices, not readiness switches. New installation/qualification must generate path-bound verification markers and identity-bound native/model proofs before an optional mode appears usable.

Each attempt writes a private numeric `fetch-attempt-*.json`; each partial has a resumable metadata receipt. They record verified IDs/byte counts/hashes, resource counters and the exact stop code. File-cache reclaim at the enforced 1 GiB cgroup limit is recorded separately from OOM; download-only success rejects OOM/pids-limit events, own swap, live reserve/power failure or cleanup failure. This does not weaken any model/benchmark admission policy. No unlimited retry loop is installed.

Nine source/mock tests cover interrupted prefix recovery, exact Range behavior, incomplete EOF, cached full SHA, existing-file preservation, no-network completion of a full partial, identity changes, safe disk preflight and explicit optional selection. They are not a live transfer receipt. [Source/build/native-check stages](BOOTSTRAP-BUILD.md), [remaining install/parity finish gate](BOOTSTRAP-PARITY.md).
