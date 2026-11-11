# Sandbox source installation and rollback checkpoint

**Implemented: create-only installation of the staged source payload into an explicit sandbox HOME. Not implemented: live-user installation or complete setup parity.** The entry refuses the current HOME and its live hidden configuration trees. Nothing here starts services, registers a panel, merges providers, downloads/copies weights or enables qualification.

The transaction reuses `bootstrap.stage` and its exact file/directory inventory. It verifies the supplied stage-manifest SHA, target-HOME binding and every staged byte/mode before mutation. This checkpoint accepts at most 8 MiB of source resources: runtime modules, disabled MoE/Q8 configuration, inactive OpenCode fragment, unit files, panel assets and notices. Native binaries and model placement remain a later guarded phase.

## Try only in a fresh Work fixture

Run from the source checkout. The outer directory below is created beside that checkout, owned by the current user. It does not change the real `~/.config`, user manager or desktop.

```sh
BOOTSTRAP_SANDBOX="$(mktemp -d -p "$(dirname "$PWD")" bootstrap-install.XXXXXX)"
PYTHONPATH=src python3 -B -m snapdragon_ai_lab.bootstrap stage \
  "$BOOTSTRAP_SANDBOX/stage" --target-home "$BOOTSTRAP_SANDBOX/target-home"
BOOTSTRAP_STAGE_SHA="$(sha256sum "$BOOTSTRAP_SANDBOX/stage/stage-manifest.json" | cut -d ' ' -f 1)"
PYTHONPATH=src python3 -B -m snapdragon_ai_lab.bootstrap_install apply \
  --stage "$BOOTSTRAP_SANDBOX/stage" --target-home "$BOOTSTRAP_SANDBOX/target-home" \
  --journal "$BOOTSTRAP_SANDBOX/install-journal.json" \
  --stage-manifest-sha256 "$BOOTSTRAP_STAGE_SHA" > "$BOOTSTRAP_SANDBOX/install-receipt.json"
```

The target may already contain unrelated provider/layout/sentinel files, which are not read or replaced. Any existing managed file causes refusal before target writes, even if its bytes happen to match. The sole reuse exception is a valid owner-only canonical lock: its inode and bytes are preserved and it is held through the transaction. New directories are private; existing safe directory modes remain unchanged.

Each managed file is written/fsynced to an owned temporary inode, its identity is journaled, then published with a create-only hard link. A concurrent destination or temporary-name collision is preserved. Descriptor-relative directory traversal rejects symlinks; no existing configuration is replaced and no provider/layout merge occurs.

Use the journal SHA from the **saved installation receipt**, rather than approving a newly changed journal by rehashing it:

```sh
BOOTSTRAP_JOURNAL_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["journal_sha256"])' "$BOOTSTRAP_SANDBOX/install-receipt.json")"
PYTHONPATH=src python3 -B -m snapdragon_ai_lab.bootstrap_install rollback \
  --journal "$BOOTSTRAP_SANDBOX/install-journal.json" \
  --journal-sha256 "$BOOTSTRAP_JOURNAL_SHA"
```

Rollback acquires the same canonical inode and removes only recorded files whose inode, owner, mode, mtime, size and SHA still match. Edited/replaced files, changed directory pointers, unknown user files and unrelated configuration are preserved. Only empty directories created by this transaction are removed. The canonical lock is deliberately retained once created, including after failure: deleting/replacing that anchor would undermine shared-inode exclusion. A rollback receipt identifies preserved owned paths. A changed journal or busy canonical lock refuses rollback without deleting files.

Caught failures automatically run the same conservative rollback and retain the private journal. Write-ahead publication identities cover published files, but this checkpoint does not claim completed power-loss/SIGKILL recovery; a crash before a temporary inode is recorded can leave an unknown temporary file for review. No recursive cleanup of unknown files is attempted.

## Tests and remaining finish boundary

Eleven focused tests passed using owned temporary fixtures beside the checkout under Work. They cover the actual staged source payload, existing canonical hardlink/bytes/provider/layout/sentinel preservation, all 15 mutation fault points of a tiny fixture, modified/unknown file preservation, busy-lock/hash refusal, enabled-profile rejection, symlink redirection, final-name races and temporary-name collisions. The fault loop mocks fsync to keep I/O small; the successful full source-payload install/rollback exercises normal filesystem writes/fsync. No live HOME, service, panel, native engine, model or user data was changed.

The module returns `qualified:false`, `services_started:false`, `panel_layout_changed:false`. Unit files still contain `%h`; do not launch a sandbox's units with the real user's manager and assume `%h` means the sandbox. The inactive OpenCode fragment is not a configured provider. Actual host installation, native/model placement, provider merge/launch wiring, right-side widget registration, restart/login behavior and identity-bound model/agent/image/lifecycle acceptance remain open. Fresh Flash proofs also cannot reuse historical private-manifest labels. [Complete parity finish gate](BOOTSTRAP-PARITY.md).
