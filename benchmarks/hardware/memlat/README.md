# Yoga SCMI memlat experiment

The overlay adds the protocol node from [RFC v8 patch 151](https://github.com/ramonjesusmiranda-debug/sp11-el2/blob/8f1726d49e57de97ac0a349a89ae894206f0fc12/kernel-patches/151-memlat-arm64-dts-qcom-hamoa-Enable-LLCC-DDR-DDR-QOS.patch). It repeats the existing parent address/size cells. It adds no LLCC/DDR resources: the matching driver supplies those.

Requires the matching built-in QCOM SCMI generic extension and memlat driver, Hamoa SCMI mailbox/shared memory, and CPUCP firmware support. Driver presence alone does not establish firmware compatibility. This is an optional boot experiment, not an installer default or a live overlay operation.

Build with the matching kernel's `scripts/dtc/dtc -@ -I dts -O dtb`; apply offline with `scripts/dtc/fdtoverlay` to the verified stock Yoga DTB. Preserve existing phandles. Compare canonical trees and require only the protocol node and symbol to be added. Never overwrite the standard DTB/UKI or change the persistent default. A separate copied UKI and Limine entry were independently reviewed on this laptop; Secure Boot was disabled. Ask the owner before rebooting.

After boot, verify protocol acceptance, memlat binding, and DDR/LLCC/DDR_QOS devfreq devices before benchmarking. If rejected, preserve the error and return to the untouched stock entry. See [ROOFLINE.md](../../../docs/ROOFLINE.md) for current measurements and limitations.
