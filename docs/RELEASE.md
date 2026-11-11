# Release checklist

The historical Flash baseline passed its protected repair and is installed with the runtime, settings, controls and UI. Its separate normal planning/image-handoff run passed in 238.448 s. [Historical installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json). The current Flash optimization goal remains open; these earlier checks do not qualify the new candidates.

- [x] Verify selected batch settings and cross-session prefix caching on isolated real OpenCode. Normal installed endpoints subsequently passed the separate lifecycle gate. The user stopped the remaining affinity screen after flat completed controls.
- [x] Document prefill prompt/context differences, measured microbatch improvement, cache behavior and power limits. Exclusive thermal attribution remains unproven.
- [x] Complete one joint KleidiAI/BF16/Q4_0 qualification and finished-task trial. Raw prefill improved; overall agent result did not justify promotion. Dense MTP and 32K are deferred by the user's queue-reduction instruction.
- [x] Verify shared settings on the MoE and compare both models on the harder repair under equal limits. MoE passed cold/reloaded; both dense controls timed out and failed one independent test.
- [x] Test dense plan/review plus MoE build, counting all swaps, reloads and retries. It passed in 1242.99 s but lost to the 213.70 s reloaded MoE; automatic split stays disabled.
- [x] Validate Q8/Q4 native 1024 and draft 256 through actual Studio; limited visual review passed. Q8 presets installed, normal handoff PNG matched the staged result exactly.
- [x] Select the shared CPU coding build and image CPU build, with short published patches and source provenance. Installation and normal-workflow verification passed on the test laptop.
- [x] Verify normal on-demand loading, switching, real 121.23-second idle release, cache clearing, image exclusion/handoff and RAM recovery. Benchmark reserve guards remain active for Flash.
- [x] Produce one selected-profile quality/time table with fixture and power limits.
- [x] Finish the [numeric disabled-path summary](FAILED-PATHS.md) with exact measurements, source hashes, fixed GPU-cleanup status and comparison limits.
- [x] Provide [manual source-preview runtime setup](RUNTIME-SETUP.md), portable disabled-by-default configurations, user units and an OpenCode v2 sample; offline parser checks pass.
- [x] Complete the [image content-identity and source-license inventory](DEPENDENCIES.md): serialized local Q4/encoder/VAE checks match pinned publisher hashes; Q8 retains prior full-hash evidence. Historical VAE download origin is not claimed.
- [x] Provide and execute the portable two-request screen against the normal MoE endpoint, with independent offline checks. It records 1/2 correct one-shot answers; the separate normal OpenCode port repair passed in 177.99 s. A model failure is retained rather than treated as a runner installation failure. The manual guide does not claim an automated complete runtime installer.
- [x] Package the Snapdragon-aware panel and migrate the prototype in place, preserving right index 1, with backups and no desktop restart.
- [x] Group 30 AI trees under Work/local-ai; preserve canonical lock inode, active services outside Work, all experiments and hardware handoffs. All 104 post-move checks passed; the source Git worktree and four SDK/header links were repaired.
- [x] Resolve image source identity: stable-diffusion.cpp `5a1ba5d895bec4e07deab80d6d791721cd59d9c2`, ggml `135ce1e0e3f2b0693d3de93d4f6b5fa0a1947820`; exact six-file patch replay passed. A bit-identical independent binary rebuild is not claimed.
- [x] Review license notices, result provenance and the public-file inventory. Local source and JSON/link checks passed; the final inventory is retained alongside the wheel validation; the wheel contains only the intended source/resources. This is a source-preview check, not Flash qualification or public publishing.
- [x] Prepare the complete local project and result summary for owner review. Public repository creation/publishing is not authorized or required for this local release.

North Mini Code, IQ4_XS and a small NPU sidecar remain optional and do not block completion of the local optimization goal.


## Runtime source packaging checkpoint

The package now includes the tested controller/prefix-store, control CLI and Image Studio source under `snapdragon_ai_lab/runtime`. Importing the runtime package starts no services. These are source snapshots, not a self-installing production release; no model weights, proprietary runtime, generated images or chat state are included. User units and chosen model/profile configuration are installed on the test laptop. Classic lifecycle acceptance passed; the newly installed Flash planning/image handoff also passed its separate complete normal-workflow receipt. Portable automatic installation of that full runtime remains unfinished.

The runtime accepts explicit paths (`LOCAL_LLM_CONFIG`, `LOCAL_IMAGE_CONFIG_DIR`, `LOCAL_AI_LEASE`). Every participating engine must use the same lock inode during migration. Default legacy paths are retained only for compatibility with the development setup. The local staging helper prepares installed-path units without writing system/user configuration or restarting services. Do not treat the existence of these sources as proof of a finished installer.

Image source identity and reconstruction are documented in [SOURCE-PROVENANCE.md](SOURCE-PROVENANCE.md). The verifier has passed against the tested snapshot; it does not replace runtime/image validation.

## Additional Flash-Next request

- [x] Pin sources/quantization, build native Oryon CPU CLI/server, verify compiler features and native settings, and check empty HTTP/auth behavior. No model inference is implied.
- [x] Complete verified four-shard download and guarded bounded-RAM smoke; all three exact checks and RAM recovery passed.
- [x] Preserve the earlier deadline-failed repair as historical evidence; its correct artifact is not a completed task time.
- [x] Qualify the winning demand-mapping/IO4/overlap full-agent repair: 1175.218 s total, 12/12 independent tests, complete final response, protected files unchanged, cleanup and RAM recovery passed. Preserve the raw false checker and strict source-backed native `empty`/controller `unloaded` reconciliation without an inference rerun. [Curated receipt](../benchmarks/results/flash-next-final-agent.json)
- [x] Install the admitted 21.5 GiB Flash profile, dynamic 2048-output alias, read-only planning launcher, Dense/MoE per-launch controls and Flash panel UI.
- [x] Complete the installed normal planning/image workflow in 238.448 s, including real idle/RAM recovery, MoE/Flash switching, Q8 PNG validation, chat exclusion and final unloaded/idle/lease-free state. Preserve the earlier Studio rejection; the passing rerun followed only the reviewed 1-to-15-second unload-timeout fix. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json)

Phone and Apple demonstrations are research context, not completion evidence for this Snapdragon laptop.


## Current Flash optimization extension — unfinished

The current plan retains the original targets: at least 1.5× complete protected-task speedup against 1175.218 s and at least 2× fixed-workload decode throughput, with all 512 experts/native top 10, unchanged independent correctness checks, 21.5 GiB admission and the user's updated 2 GiB actual host reserve. Nothing is pushed. The earlier checklist records the installed baseline rather than completion of this extension.

- [x] Create the rollback baseline and finish the requested attribution report before tuning. See [profile](FLASH-NEXT-PROFILE.md).
- [ ] Finish the finite supported short-screen queue and choose the final arithmetic, storage and cache configuration. IO8 has a qualifying full-task observation; cache 13 GiB and selective Q8 retain their own scopes. Wider threads, placement and predictive prefetch retain their failures. Coalescing is below its whole-request advance gate. Dense repack support and genuine io_uring backend qualification remain open.
- [ ] Close speculation with measured, admissible results or a specific evidenced unsupported outcome. MTP head integrity and constructor/workspace measurements pass; complete MTP inference admission and token/state/rollback correctness do not. N-gram speculation has separate support and rollback gates.
- [ ] Finish the requested small QAIRT Qwen3-4B helper, including ordinary-user access, task quality, idle release and co-residency RAM. It is required by this extension; large-model NPU mapping remains excluded.
- [ ] Close the conditional [Vulkan prefill gate](FLASH-GPU-PREFILL-GATE.md) with its exact support and measurement scope. Prior 27B results do not qualify Flash.
- [ ] After worthwhile short screens are resolved, validate the final configuration on the complete protected task with all 12 independent tests, final response, unchanged protected files, power/RAM gates and reliable memory release. Preserve matched and historical comparison scopes; do not multiply component ratios.
- [ ] Deploy only qualified survivors and verify normal OpenCode/planning, model switching, real idle release, image exclusion/handoff, the always-visible panel and rollback. Verify the shareable fresh-install path and dependencies against the same selected configuration.

The broader model-quality and image follow-ups remain tracked in the plan. A successful tiny fixture, constructor measurement, source review or installer dry-run is insufficient for any unchecked item above.
