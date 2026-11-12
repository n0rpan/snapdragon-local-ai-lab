# Releasing

How work moves from the lab's private development repositories to the public ones. Standing rule since 2026-11-11 (owner decision O-29).

## Two kinds of repository

| | Development (`-dev`, private) | Public |
|---|---|---|
| Repositories | one private development repository per public one | [snapdragon-local-ai-lab](https://github.com/n0rpan/snapdragon-local-ai-lab), [snapdragon-engine](https://github.com/n0rpan/snapdragon-engine) |
| History | Full: every lane branch, frozen `root/*` build branches, `archive/*` tags, development diaries, run tickets | One squashed commit per release, tagged `vX.Y` |
| Who pushes | Any lane, any time | Only through a release (below) |
| Gate | Pre-push scan for secrets and private paths | Everything in "Real progress" |

Development history is never pushed to a public repository: no merge, cherry-pick or push of a `-dev` branch, and no `git remote add` of a public URL inside a `-dev` checkout. Public repositories are written only from a fresh snapshot directory.

## Development pushes: free

Push lane work to the `-dev` remotes as often as useful. The pre-push hook scans the pushed objects for keys, tokens, credentials, MAC addresses and serials and refuses the push on a hit. Absolute paths are allowed there; transcripts, real-session prompts, browser data, model weights, caches and core dumps are never committed anywhere.

## Public releases: squashed, on real progress only

A public release happens **only on real progress**, meaning at least one of:

- a **qualified and installed improvement**: it passed its quality gate (see docs/RESULTS.md, "Quality gates") and runs in the installed stack; or
- a **fixed bug**, with a test or receipt that shows the fix.

And all of these hold:

1. **Receipts.** Every new or changed number has a public receipt in `benchmarks/results/public/<id>.json`, exported with `tools/export_receipt.py export` (fail-closed scan).
2. **Docs updated.** README "Stock → today" and docs/RESULTS.md show the new numbers with their conditions and receipt ids; superseded numbers move to the history section, not out of the record.
3. **Scans at 0.** The snapshot builder's scans report 0 secrets, 0 private words and 0 private-path hits on the snapshot tree.
4. **CI green from a fresh clone.** Clone the built snapshot into an empty directory and run the model-free tests there (lab: `PYTHONPATH=src python3 -B -m unittest discover -s tests`; engine: `cargo test --release --workspace --exclude qwen4exp`).
5. **v0.1.2 freeze audit (O-87).** After the qualified chat finals (v0.1.2 is chat-only) are merged into both private dev mains, announce `CODE FREEZE v0.1.2` in the private coordination log, naming both SHAs. Release is blocked until the Fable-led multi-Haiku audit passes on both frozen repositories: bugs/errors, dead code, test theatre, docs and user-facing behavior. Re-audit affected areas after any fixes that change the freeze.

Not a release: work in progress, tuning noise, numbers without a receipt, unqualified candidates, or a flag that is off by default and changes nothing installed. Those stay in `-dev` until they qualify.

**Who decides.** The orchestrator and an independent Opus reviewer agree that the change is real progress and that the checks hold. The owner is informed in the next update with the version, the changelog and the receipt ids, and can veto or revert a release.

## Opt-in defaults (owner decision O-30, 2026-11-11)

Features that trade something the user may not want (desktop responsiveness, prompt data on disk) ship **off by default** in public releases. The owner's machine turns them on in its own profile; that setting is never part of a snapshot.

| Feature | Public default | Owner's machine | How to turn on |
|---|---|---|---|
| Flash cache preference (`oxe_flash.cache_pref`, MemTotal rule, idle desktop pages may go to swap) | off: `auto` | 18 GiB; 20 GiB remains held | profile `cache_pref: "18"` or the panel's Flash cache control |
| Flash disk prefix store | off | on, at the qualified 18 GiB cache tier | profile `oxe_flash.prefix_store.enabled: true`; [qualification and limitations](FLASH-PREFIX-2.md) |
| Studio `oxi` prompt-conditioning cache (`OXI_COND_CACHE`, D12) | off (decided; see note) | on | Studio setting / `OXI_COND_CACHE` |

The v0.1.1 public Studio source makes persistent prompt conditioning opt-in. The
test laptop enables it explicitly. Its earlier 20 GiB desktop-cost screen is
historical; the installed Flash preference is now 18 GiB.

The controller uses [GUARD-2b](GUARD-2.md), including the sustained host-pressure
conditions and immediate native swap/OOM stops. This supersedes the earlier
F127 machine setting; it does not enable PSI-based prefill splitting or throttling.
The prefix-store installation retains its explicitly accepted short-request
timing exceptions, recorded in [FLASH-PREFIX-2](FLASH-PREFIX-2.md).

## Mechanics

- **One commit per release** on public `main`, on top of the previous release tag. Tag `vX.Y` (annotated).
- **Commit message = changelog**: what changed for users, each item with its receipt ids (`Receipts: 063, QC-CTX2`), and the private source commit it was built from.
- The release tool (kept in the private lab repository, `tools/build_public_snapshot.sh`) does the work and never pushes:

  ```sh
  # first release only: a new single-commit repository
  tools/build_public_snapshot.sh REPO REF EXCLUDE_FILE OUT_DIR v0.1.0 CHANGELOG
  # every later release: the next squashed commit against the public repository's last tag
  tools/build_public_snapshot.sh --release vX.Y REPO REF EXCLUDE_FILE PUBLIC_REPO OUT_DIR CHANGELOG
  ```

  Both modes refuse a changelog without receipt ids and write the commit message as: version, changelog, private source commit. `--release` clones the public repository, checks that `vX.Y` is newer than its last tag and that `main` has nothing after that tag, replaces the tree with the scanned snapshot of `REF`, commits it once with the changelog (which must name receipt ids) and tags it. Pushing is a separate step after review: `git -C OUT_DIR push PUBLIC_REPO main vX.Y`.
- `REF` is an annotated `release/vX.Y-source` tag in the `-dev` repository, pushed to the private remote only, so every public release maps to an exact private commit.
- External pull requests to a public repository are reviewed there, then ported into `-dev` by hand; the next release carries them (credited in the changelog).

## What CI proves

The private snapshot builder is the full six-scan release gate, including the local private-word list and SDK scan. Ordinary CI does not recreate that machine-specific deny list or rerun the snapshot builder. Exact-tree CI builds and model-free tests the already scanned snapshot; it does not replace the release scan.
