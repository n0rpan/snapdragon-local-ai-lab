# Complete public source archive

The exporter packages an exact local Git commit into a verified source archive, providing an alternative to the wheel's missing repository-root resources. It includes every regular tracked public file: runtime/prefix/Studio/control source, panel, patches, manifests, examples, docs, tests and notices. It does not include `.git`, working-tree edits, untracked files, caches, weights or private/vendor SDK payloads. A tracked symlink, submodule, prohibited payload or unexpected top-level path refuses the export rather than silently producing an incomplete source release.

From the complete repository, choose the reviewed **full 40-character immutable commit** and a new archive destination whose parent already exists:

```sh
PYTHONPATH=src python3 -B -m snapdragon_ai_lab.source_archive export \
  --repository "$PWD" --commit FULL_REVIEWED_COMMIT \
  --destination /absolute/new-destination/source.tar.gz
```

No remote, checkout, build, download, install, service, model or device action runs. Local Git objects are read with isolated config, lazy fetching disabled, terminal prompting disabled and finite output/deadline limits. Missing promisor objects refuse rather than fetching. The exporter reads committed bytes, so uncommitted exporter/docs changes are not part of an earlier commit. It refuses destination or sidecar replacement and symlinked paths. The archive and adjacent `.json` receipt are fsynced and published without overwriting existing files.

The archive root is `snapdragon-local-ai-lab-<first12commit>`. Its `SOURCE-ARCHIVE.json` records the complete path/mode/size/SHA256/Git-blob inventory and original commit/tree. The exporter independently recomputes the Git tree from those bytes and modes. File execute permissions are retained; Git does not record historical directory ownership or modes. Archive ownership/timestamps are normalized, making repeated exports of the same commit deterministic. Receipts keep `qualified:false` and `end_to_end_parity_passed:false`.

Transfer the pinned archive SHA from the reviewed receipt/release, then verify before extracting:

```sh
PYTHONPATH=src python3 -B -m snapdragon_ai_lab.source_archive verify \
  /absolute/source.tar.gz --sha256 PINNED_ARCHIVE_SHA256
mkdir -m 700 /absolute/fresh-extraction
# After verification, extract only into that new owned directory.
tar --extract --gzip --same-permissions --no-same-owner \
  --file /absolute/source.tar.gz --directory /absolute/fresh-extraction
```

Verification checks the pinned outer checksum, safe regular members, exact complete per-file inventory and Git-tree identity. A capped gzip reader scans physical 512-byte tar headers before `tarfile` parses extended metadata: individual PAX metadata is limited to 64 KiB / 16 MiB aggregate, unsupported GNU metadata refuses, and total decompressed framing/payload is bounded. Ordinary source payloads retain 16 MiB/file, 256 MiB total and 10,000-file limits. An arbitrary untrusted sidecar is not an authenticated release checksum. No Git metadata is required after extraction. From any different working directory, the existing read-only selected plan works with the extracted complete project:

```sh
source_root=/absolute/fresh-extraction/snapdragon-local-ai-lab-COMMIT12
PYTHONPATH="$source_root/src" python3 -B -m snapdragon_ai_lab.bootstrap \
  --project "$source_root" plan --target-home /absolute/not-installed-owner
```

That plan reads public resources only and does not create the target HOME or grant readiness. Source/archive verification is not engine, model, fresh-install or panel qualification. Continue with [selected setup reproducibility](SELECTED-SETUP-REPRODUCIBILITY.md), [bootstrap parity](BOOTSTRAP-PARITY.md) and the separate guarded acquisition/build/install steps. Qualcomm proprietary assets remain external.


The first actual complete public-archive roundtrip passed for commit `8c1cc846e606763fadd75429fde4f58b246291c1`: **271 tracked source files / 4,535,182 source bytes**, compressed to **807,288 bytes**. Pinned checksum verification, exact extracted file bytes/modes/Git-tree identity and absence of Git metadata passed. From a different working directory, the extracted source's existing read-only plan selected the four default assets; both target and caller HOME remained absent. [Numeric receipt and exact hashes](../benchmarks/results/source-archive-roundtrip.json). No engine compilation, download, service, model, device or installation action ran. This closes that source-distribution check for the recorded commit; full fresh-install/runtime parity remains false.
