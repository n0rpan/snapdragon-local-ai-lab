# Optional Omarchy control panel preview

The packaged `snapdragon.local-ai` panel is installed and verified on the test laptop. It replaced the personal prototype in place, preserving right-bar index 1 without a duplicate entry or shell restart. It shows selected model, backend state, available RAM, compute-busy state and image activity; actions open OpenCode/chat/Image Studio and control lazy loading/unloading.

It calls the existing `~/.local/bin/local-ai` command. This development preview does not yet install the full model runtime, services or weights. The installer places the packaged planning launcher beside the existing regular control command and backs up any previous launcher. A failed panel installation restores it. Install the control command as a regular file, because its planning action resolves this sibling after resolving its own path. Status polling stays local; the installer starts no models and downloads nothing.

Preview installation on an existing configured system:

```sh
PYTHONPATH=src python3 -m snapdragon_ai_lab install-plugin --dry-run
```

When the preview is appropriate for your setup, omit `--dry-run` to install. The installer requires Omarchy and an executable local-ai command, detects Snapdragon X through architecture/device-tree identifiers, backs up the existing layout/plugin, and places the panel in the always-visible right section. A failed layout command restores the prior files. It refuses to replace a symlinked plugin directory.

`--allow-unsupported` is an explicit contributor override for other hardware. The currently tested machine is a Yoga Slim 7x; family detection is not proof that another device's GPU/NPU or firmware behaves the same way.

To migrate an existing personal clone without adding a second entry:

```sh
PYTHONPATH=src python3 -m snapdragon_ai_lab install-plugin --replace-plugin <old-plugin-id> --dry-run
# After reviewing the preflight, repeat without --dry-run.
```

The replacement ID must be a distinct dotted plugin ID and an existing user plugin directory. Migration requires an explicit `bar.layout` containing that ID exactly once and no public-plugin entry or duplicate IDs. It preserves the entry's settings, its index when already in `right`, and every unrelated configuration value. An entry in the tray or another section moves to `right`, immediately after `omarchy.tray` (or first if there is no tray toggle). No model is loaded, and no shell restart is needed unless the panel code changed (see below).

Before changing anything, migration validates the layout and paths; `--dry-run` writes nothing. Symlinked installation roots, plugin roots, backup roots, shell files and target directories are refused. A timestamped `plugin-backups/` directory retains the original `shell.json`, any previous public plugin under `plugin/`, and the replaced clone under `replaced-plugin/`. On failure the installer restores the shell and both plugin directories. For manual rollback after a successful migration, remove the new public plugin, restore `plugin/` to `plugins/snapdragon.local-ai` if it exists, restore `replaced-plugin/` to its original ID, then restore `shell.json` last. Avoid concurrent layout edits during installation.

**Updating the panel (v0.1.3).** A first install, or an update that changes only the texts, needs no shell restart. When an update changes the panel code (`.qml`/`.js`), the running shell keeps the old code with the new texts until `omarchy restart shell` runs. `install-plugin` reports this as `restart_required`, with `restart_command` and `next_step`. Add `--restart-shell` to run the restart right after a successful install (the bar disappears for a moment). Nothing restarts by default, and `--dry-run` never restarts.

Without `--replace-plugin`, the existing installation behavior is unchanged: assets are installed and `omarchy bar put snapdragon.local-ai --section right` places the panel. Use explicit migration when replacing a personal clone.

The plugin uses the current user's HOME rather than a fixed username. It is a local control interface, and does not collect prompts, images or telemetry. Its dependencies and runtime setup will be documented before a complete release.

2026-11-10: Ornith shown as the production default (no beta labels); Flash-Next opens the ordinary OpenCode build agent; chooser/storage data for a later redesign is in [UI-DATA](UI-DATA.md).
