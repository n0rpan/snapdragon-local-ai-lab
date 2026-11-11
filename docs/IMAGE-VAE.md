# Installed VAE affinity fix and selectable convolution

The test laptop now runs the qualified VAE-1 Studio and image engine. The fast 512×512 FLUX.2 klein Q8 route uses chunks32/32, f32 attention and int8 VAE convolution. Other sizes and edits retain sd.cpp. The change fixes an affinity bug: the scheduler pinned its creator to one CPU, and later VAE workers inherited that mask. The engine now restores creator affinity on pool drop and drops the DiT/pool before starting VAE work.

[Portable measurements and artifact identities](../benchmarks/results/public/VAE-1.json) preserve the qualification and installed smoke. The f32 output after the fix is pixel-identical to the old f32 output. All nine direct VAE measurements observed 12  allowed CPUs.

| VAE mode | Mean cold time including load, 3 seeds | Quality |
| --- | ---: | --- |
| f32 | 3.034 s | reference |
| Winograd | 2.304 s | pixel PSNR92.17–92.46 dB vs f32 |
| int8 | 1.196 s | pixel PSNR53.90–54.99 dB vs f32 |

The predeclared choice rule selects int8: it passes all three seeds and saves more than 0.5 s beyond the eligible lossless choice. Public source defaults to f32; this laptop explicitly selects int8. Public persistent prompt conditioning remains opt-in/OFF; this laptop explicitly enables it.

End-to-end Studio qualification used A42, B42, B43, C44. A was sd.cpp 81.475 s; cold-conditioning B requests were 31.078/30.885 s; matching-conditioning C was 24.790 s. The B quality scores 31.466/32.068 dB and C 34.053 dB exceed the frozen 27.812 dB gate. These timings are a **screen**, not a new repeated identical-seed speed comparison. The earlier [IMG-SHIP-1](IMAGE-SHIP.md) matched comparison remains separate evidence; its 51.53 s result cannot be treated as the control for this run.

Installed fresh-prompt smoke completed in 24.892 s on oxi, with no fallback, a 7.660 GiB cgroup peak under 22 GiB, zero model swap and no OOM. It is one smoke, not a speed claim. The standard guards and canonical inference lease passed. The frozen Studio privacy filter omits native free-text logs; the twelve-CPU evidence comes from direct VAE receipts, not invented per-request log lines.

Source identities: engine 7120b01, binary`fc267f50a3e05feeb4ec641abeee3275a29b2fb04c30c02af676fe5e3e5c0069`; qualified Studio 1abf34b, artifact`90a18e6c3ce7078f119fb1bd38552dae221a9b850eeaa939503c910f3bdc9ec1`. Lab source merge 124f94b preserves public cache-OFF semantics and passed 118 model-free Studio tests; it is distinct from the installed frozen artifact.

The machine-specific guarded rollback restores its prior Studio/engine and explicit cache setting. The engine selector also retains sd.cpp fallback. No chat model, memory guard or default changed with this installation. Larger-size fast routing remains unqualified pending its own resolution-specific quality gate.
