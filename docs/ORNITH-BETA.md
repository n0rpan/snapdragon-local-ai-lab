# Ornith opt-in profile

Ornith 1.5 35B A3B APEX-MTP Compact passed the matched two-code/five-functional screen and a separate normal OpenCode repair with all 13 protected cases, six completed tools, one completed test command and a complete final response. Both included native cleanup, RAM recovery and lease release. The normal task took 128.25 seconds from load through cleanup. No new paired reference agent was run, so this is not a comparative agent-speed result. The structured final JSON screen used the declared native API schema contract; earlier natural-format failures remain preserved. [Results and limits](ORNITH-MOE-COMPARISON.md).

The source adds an **optional profile**, not a new default. Qwen stays selected, with its existing settings and prefix cache. Ornith uses the same classic engine, its own embedded template, 16K context, threads 4/batch threads 10, batch/ubatch 256, F16 KV, flash attention off, CPU repack, native MTP draft 2 and thinking disabled. Its native expert count and routing are untouched. The beta explicitly disables the external PrefixStore because the protected task did not enable --slots or slot-save-path. Ordinary in-session prompt reuse remains available.

The panel's beta OpenCode launch uses temperature 0, prompt reuse and thinking false, with input/output metadata 12,288/2,048. Proxy requests to the beta text-generation routes receive those defaults only when fields are absent. Explicit client sampling, token limits and thinking choices survive and are user experiments beyond the fixed qualification. No forced JSON grammar is added to ordinary beta chat or OpenCode requests. Other models' request policy remains unchanged.

## Prepare an explicit release

The [portable example](../examples/ornith-beta-profile.pending.json) is disabled. The normal bootstrap neither selects nor downloads Ornith. Wrapper code remains MIT; model weights and publisher terms remain separate. The exact model is in [the artifact catalog](../dependencies/model-artifacts.json): 17,437,861,152 bytes, full SHA a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df.

[stage_ornith_beta.py](../tools/stage_ornith_beta.py) accepts explicitly hashed actual screen, inventory and protected-task receipts. The caller must explicitly repeat the reviewed task SHA. It validates all mandatory gates and current file stat, then writes only a **fresh private stage**: augmented runtime.json, exact rollback-runtime.json, a model-only OpenCode fragment, runtime source and an identity manifest. It does not install, restart, fetch, load or rehash model weights. It refuses live .config/.local destinations. Another machine or rebuilt engine needs its own qualification; copying this machine's receipt hashes does not qualify it.

```sh
python3 -B tools/stage_ornith_beta.py "$NEW_PRIVATE_STAGE" \
 --runtime "$CURRENT_RUNTIME_JSON" --runtime-sha256 "$CURRENT_RUNTIME_SHA" \
 --agent "$ACTUAL_NORMAL13_HOST_RESULT" --agent-sha256 "$NORMAL13_SHA" \
 --screen "$ACTUAL_ORNITH_V3_SCREEN" --screen-sha256 "$SCREEN_SHA" \
 --inventory "$ACTUAL_COMPLETE_INVENTORY" --inventory-sha256 "$INVENTORY_SHA" \
 --root-reviewed-agent-sha256 "$NORMAL13_SHA"
```

All paths must be absolute. Review the staged diff before a separate deployment. Runtime catalog checks read only model stat; they never hash payloads or wake a model. Beta load verifies the bounded classic executable hash under the existing inference lease. The classic engine's THP policy is unchanged.

## Deployment and rollback gates

1. Wait for every model, image, download or native test job to terminate. Confirm idle frontends, available RAM, AC and battery reserve. Take the existing canonical lease for the configuration transaction without replacing its inode.
2. Verify staged source/receipt hashes and the exact runtime preimage. Back up current runtime config, user unit, CLI/launcher, OpenCode file and user-owned panel. Preserve the qualified prior runtime release.
3. Place the staged runtime files in a fresh immutable runtime release, update only its WorkingDirectory/ExecStart references, and install the reviewed CLI/launcher. Keep CPU 800%, High 24 GiB/Max 25 GiB, swap 0 and other existing unit settings. Merge only the fragment's Ornith model entry into the existing local provider; preserve other providers, credentials, agents, permissions and global model selection.
4. Use the [existing panel installer and backup path](OMARCHY-PLUGIN.md). For a personal clone, use its explicit replacement option; do not create a duplicate widget or edit packaged /usr/share/omarchy files. User plugin files hot-reload. No shell reset is required.
5. Restart only the reviewed chat controller after copying the approved config, then confirm metadata lists both Qwen and Ornith while the selected default remains Qwen and no model is loaded. Status polling must remain nonwaking. Release the transaction lease before inference so the controller can acquire it normally.
6. Separately validate an explicit Ornith load/chat/tool response, app/OpenCode model routing, switch back to Qwen, exclusive image handoff, unload and the real idle timer. Record actual process reap, own swap/OOM/PID events, resource limits, host reserve and RAM recovery. Keep the beta offered only if those installed lifecycle checks pass. Finish with the original Qwen selected.

Rollback: under the same idle/lease discipline, restore exact backed-up unit/config/CLI/OpenCode/panel files and the prior runtime release. Reload user units and restart the prior controller. Preserve stage evidence, the lease inode, model weights and user history/images. Removing downloaded models is not part of rollback.

## Device identity after reboot

The first release deliberately matches device/inode/size/mtime from its actual inventory and task. A mount namespace or Btrfs reboot can change st_dev, making the beta unavailable until metadata is rebound. This is not a reason to hash 17 GB during UI polling or rerun the protected task automatically.

Record the independently verified filesystem UUID in the private deployment evidence. If only the device number later changes, perform a bounded, lease-protected inventory reproof: verify the same filesystem UUID, inode, size and mtime, and identical complete GGUF metadata/tensor/tokenizer/template descriptors. Then explicitly review a metadata-only rebind, preserving the prior full-file SHA and normal-task receipts. If stable filesystem identity or descriptor equality cannot be established, keep the beta disabled. There is currently no automatic doctor/rebind command; do not globally ignore device mismatches.

## Test laptop installed checkpoint

The reviewed beta is installed beside Qwen. The Local AI panel offers **Use Ornith**; the OpenCode launcher follows that selection, while ordinary terminal OpenCode still defaults to Qwen. The installed controller passed lazy selection, a real chat answer and structured tool call, its unchanged two-minute idle unload, shared-lease exclusion, Qwen reload/answer and explicit unload. Both native workers were reaped and RAM recovered; Qwen was selected again and no model was left loaded. [Installed lifecycle evidence](../benchmarks/results/ornith-beta-installed.json).

The shared-lock check simulates an image worker holding the existing lease; it did not generate an image. The earlier protected normal13 task supplies model/agent qualification; the panel-launched OpenCode task was not repeated during this lifecycle. Reboot/device-number reproof and another machine's qualification remain separate.
