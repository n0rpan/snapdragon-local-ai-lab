# Optional SCMI memlat boot experiment

The Yoga Slim 7x Gen 9 (83ED / 14Q8X9, BIOS NHCN62WW) has the kernel-side SCMI generic extension and Qualcomm memlat devfreq driver built in (`7.3.0-40-omarchy`). Its stock device tree exposes SCMI protocol 0x13 but not vendor protocol 0x80. A minimal DT overlay adds only `protocol@80 { reg = <0x80>; }`; the driver provides the DDR, LLCC and DDR-QOS configuration.

This is firmware-dependent and remains an opt-in boot experiment. Upstream reference: [patch 151 at the source commit](https://github.com/ramonjesusmiranda-debug/sp11-el2/blob/8f1726d49e57de97ac0a349a89ae894206f0fc12/kernel-patches/151-memlat-arm64-dts-qcom-hamoa-Enable-LLCC-DDR-DDR-QOS.patch) and [upstream RFC v8](https://patch.msgid.link/20260724-rfc_v8_scmi_memlat-v1-0-cb732bcff1f4@oss.qualcomm.com). The patch is not merged upstream. See the [overlay and boot requirements](../benchmarks/hardware/memlat/README.md).

## Current evidence

On this laptop, an independently reviewed alternate UKI booted on 2026-11-09. Firmware accepted protocol 0x80, the built-in driver initialized three performance domains, and remote-governed `ddr`, `llcc` and `ddr-qos` devfreq devices appeared. The owner explicitly approved making this tested entry the persistent default after the post-resume bandwidth check; that change was applied at 16:43 UTC. Stock Omarchy remains a named fallback. This does not make MEMLAT a default for other installations.

A 2 GiB, 32-pass NEON streaming-read fixture passed all checksums. With performance CPU governor and 3.4176 GHz, the largest median was 115.85 GB/s (six spread threads), versus a pre-boot maximum of 50.31 GB/s. In the matched CPU-only Ornith Compact MTP screen (4 decode / 10 batch threads, 4096 context, three fixed 128-token prompts and 64-token outputs), median decode rose from 12.7317 to 17.1822 tok/s (+35%), prefill from 94.7439 to 107.6971 tok/s (+14%), and request time fell from 6.3015 to 4.8364 seconds. All three outputs were byte-identical.

These are bounded measurements, not a general hardware or firmware qualification. Results depend on firmware, workload, CPU placement and thermal/power conditions. Higher performance may increase heat and power use. Completed thread sweeps reached 97.7–98.0 °C; a 30-second idle observation cooled from 45.6 to 39.9 °C and DDR firmware votes fell as low as 200 MHz. A subsequent 601-second Ornith decode run changed from 29.87 to 29.72 tok/s in the first/last windows (−0.5%), with a whole-unit peak of 97.7 °C. No package-power counter is exposed. A matched idle-power comparison and s2idle recovery remain unverified. Full measurements and methods are in [ROOFLINE.md](ROOFLINE.md). See [Omarchy issue #18254](https://github.com/omacom/omarchy/issues/18254).

### Deep suspend and resume, 2026-11-09

The owner performed two **deep** suspend cycles at 16:32:09–16:32:25 and 16:32:53–16:33:09 UTC. `/sys/power/mem_sleep` was `s2idle [deep]`; this was not an s2idle test. Uptime and boot ID continued. DDR/LLCC/DDR-QOS devfreq devices survived both cycles. The kernel journal from 16:31:25 UTC through the post-resume check shows both deep entry/exit pairs and no SCMI/memlat error. It does show ath12k Wi-Fi DMA-allocation retries during resume; this test does not establish their cause or general Wi-Fi stability. The raw journal stays private because it contains device identifiers.

The same 2 GiB NEON read fixture (32 passes, six threads spread over CPUs `0,4,8,1,5,9`) passed three checksums after resume: **108.97 / 108.27 / 108.63 GB/s**, median **108.63 GB/s**. This is 6.2% below the earlier 115.85 GB/s peak and above the predeclared 90%-of-baseline pass floor. DDR reached its 4.224 GHz vote (8448 MT/s), LLCC 1.066 GHz and DDR-QOS 100. This short check establishes recovery, not long-term suspend reliability. Numeric receipt: `benchmarks/roofline/post-resume-read.jsonl` (lab notebook, not in the public snapshot).

### Approved default and exact rollback on this machine

`/boot/limine.conf:4` changed from `default_entry: 2` to:

```text
default_entry: Local AI memlat test (7.3.0-40)
```

Limine supports named entry paths; unlike numeric indices, these survive snapshot-menu insertions. Following the owner's approval of automatic rebuilds for daily edge updates, line 4 now selects `Local AI memlat (7.3.0-40-omarchy)`. That automatic entry points to `/boot/EFI/LocalAI/omarchy-memlat-7.3.0-40-omarchy-affb2cce7ddb.efi`, SHA256 `affb2cce7ddbf5e7cf89733e70defc4ad038189c3de0a539b77125274991f5fe`, byte-identical to the tested image. The original tested entry/image and `Omarchy/linux-omarchy` stock fallback are retained. No reboot was requested or performed for either default change.

Root-only before/after backups and the owned-file manifest are in `/var/lib/local-ai-memlat/20261109T164300Z/`. To revert the selection without touching either image:

```sh
sudo /usr/local/libexec/local-ai-memlat-boot fallback
```

This writes `default_entry: Omarchy/linux-omarchy` to the same line after checking that the stock leaf, image and Limine image hash agree. Alternatively choose **Omarchy → linux-omarchy** in the boot menu. The menu timeout remains one second. The two methods select the unchanged stock boot, without modifying firmware or rebuilding anything.

### Automatic verified rebuilds for daily edge updates

The owner approved automatic rebuilds so daily edge kernel updates retain MEMLAT when verification succeeds. The old kernel image must not remain the default after matching modules disappear. Installed safeguards are root-owned copies of the repository files:

- [tools/memlat_boot.py](../tools/memlat_boot.py) → `/usr/local/libexec/local-ai-memlat-boot`, with the qualified stock/MEMLAT hashes, module directory and named entries in `/etc/local-ai/memlat-boot.json`.
- [ALPM pre-transaction hook](../packaging/memlat/05-local-ai-memlat-fallback.hook) → `/etc/pacman.d/hooks/05-local-ai-memlat-fallback.hook`. Kernel, headers, bootloader and kernel/DTB-path transactions select stock **before** files are replaced and mark that the approved MEMLAT default needs rebuilding. `AbortOnFail` stops the transaction if safe selection cannot be established.
- [ALPM post-transaction hook](../packaging/memlat/99-z-local-ai-memlat-rebuild.hook) → `/etc/pacman.d/hooks/99-z-local-ai-memlat-rebuild.hook`, ordered after stock initramfs/UKI generation. This rebuilds against the new stock UKI/DTB, installs a separate content-addressed image, updates only the owned automatic menu entry, and reselects it on success.
- [boot-tool pre hook](../packaging/memlat/20-local-ai-memlat-fallback) → `/etc/boot/hooks/pre.d/20-local-ai-memlat-fallback`, and [post hook](../packaging/memlat/85-local-ai-memlat-rebuild) → `/etc/boot/hooks/post.d/85-local-ai-memlat-rebuild`. Manual `limine-entry-tool` rewrites get the same fallback/rebuild handling under its existing mutex. The post hook runs before config-hash enrollment.

Rebuild prerequisites are matching kernel modules/headers with the built-in MEMLAT driver, disabled Secure Boot, and a stock Yoga DTB whose complete bytes exactly match one embedded `.dtbauto` section. The installed [builder](../tools/memlat.py) and [overlay](../benchmarks/hardware/memlat/yoga-memlat.dtso) live at `/usr/local/libexec/local_ai_memlat.py` and `/usr/local/share/local-ai-memlat/yoga-memlat.dtso`. The builder requires the patch to fit the existing unsigned PE section, and verification checks that exactly that DTB payload changes; kernel, initrd, command line and other device trees are preserved. New images use the kernel version and image hash in their filenames; stock images are never overwritten. The stock menu entry's outer `cmdline` is copied as well: it differs from the embedded UKI command line on this Yoga, and the automatic entry must preserve the actual tested boot arguments. Command-line text stays in the private root policy/menu, not public receipts.

If any rebuild check fails, stock remains the default, the failed rebuild remains pending, and `notify-send` sends the owner a critical desktop notification (with console/package log output if the desktop bus is unavailable). A failure after image staging also leaves stock selected. No firmware variables or stock images are changed. Another explicit user default is preserved when no MEMLAT update is pending. Manual `fallback` cancels a pending automatic rebuild. Old images are retained for rollback; they need their matching modules if selected manually.

The current-kernel dry run passed and produced **the exact SHA256 of the already tested image**. A real rebuild/install then selected the content-addressed automatic entry; `check` passed and a second post-hook-style call did no work. Twenty-four model-free MEMLAT tests cover dry-run immutability, repeated rebuilds, update failure + notification, preservation of the original entry and stock outer command line, empty command lines, boot-generator formatting changes, rejection of semantic edits, missing modules, changed hashes, menu renumbering and concurrent-edit refusal. See the [dry-run receipt](../benchmarks/hardware/memlat/autorebuild-current.json). A genuinely newer kernel and its firmware/runtime behavior remain untested until the next update; successful construction is not a runtime stability guarantee. After updates, use `doctor` to verify the domains and keep stock available if boot or inference regresses.

```sh
sudo /usr/local/libexec/local-ai-memlat-boot rebuild --dry-run
sudo /usr/local/libexec/local-ai-memlat-boot check
```

Direct manual overwrites that bypass both package and boot tools must first run `fallback`; the hooks cannot protect arbitrary writes outside those tools. To uninstall automation, run `fallback`, remove only the two ALPM hooks, two boot-tool hooks, installed boot helper/builder/overlay and `/etc/local-ai/memlat-boot.json`. Keep stock and backups. Removing experimental images/entries is separate; never remove the selected image.

## Build, install and remove

Run the helper from the repository root. `doctor` is read-only. Build against the exact installed UKI payload; the helper checks that it matches the selected stock Yoga DTB/DTS before producing a separate image:

```sh
python3 tools/memlat.py doctor
python3 tools/memlat.py build --uki /path/to/current.efi --output /path/to/memlat-test.efi
```

By default, `build` preprocesses and compiles the installed Yoga DTS with `cpp` and `dtc -@`, then applies the overlay with `fdtoverlay`. You can pass `--stock-dtb VERIFIED` or `--stock-dts DTS` to select the source explicitly, `--kernel-build /usr/lib/modules/$(uname -r)/build` to select matching kernel tools, or explicit `--cpp`, `--dtc` and `--fdtoverlay` paths. Never overwrite the standard DTB or UKI. Kernel updates do not refresh the separate experimental image: rebuild and review against the new stock image, and keep the modules matching whichever kernel you boot available.

Installation writes a separate image and Limine entry plus an ownership manifest. It requires Secure Boot disabled, existing parent directories/config, and fresh destination-image/manifest paths. A valid explicit numeric Limine default is required so the helper can prove it preserves the existing selection. Review the image and entry before choosing the experimental entry for boot:

```sh
python3 tools/memlat.py install --image /path/to/memlat-test.efi \
  --destination /path/to/EFI/memlat-test.efi \
  --config /path/to/limine.conf \
  --manifest /path/to/memlat-test.json \
  --efi-path 'boot():/EFI/memlat-test.efi'
```

The helper does not alter the persistent default, select a one-shot boot or reboot. Removal is limited to the exact image and entry recorded in the manifest; it refuses modified artifacts and preserves unrelated Limine edits:

```sh
python3 tools/memlat.py uninstall --manifest /path/to/memlat-test.json
```

If firmware rejects protocol 0x80 or the devices do not bind, boot the normal Omarchy entry and remove the experimental artifacts with `uninstall`.
