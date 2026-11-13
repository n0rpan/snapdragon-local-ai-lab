# Dependency identities and license inventory

This source preview's original wrappers/tools are MIT (copyright Oscar Nordström (n0rpan)). Code taken from or modifying another project keeps that project's licence and notice; see [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES). External runtimes, their dependencies and model weights retain their own licenses. The [machine-readable inventory](../dependencies/image-assets-and-licenses.json) separates publisher catalog identity, local download receipts, prior local hash verification and unverified candidate identity. No model payload was read or hashed while the agent was running. After all heavy jobs ended and both frontends were idle, the three outstanding assets were hashed serially under the canonical exclusive lease; no model download or inference occurred.

## Host build tools and external sources

A fresh installation needs Git, Python 3, a C/C++ compiler with OpenMP/libgomp, CMake and Ninja. The classic chat build also uses OpenSSL development files. The guarded executors need a systemd user manager and `unshare` user/network namespaces. The Omarchy panel additionally needs the supported Omarchy shell; it does not supply a desktop environment.

The independent synthetic code checks also require [bubblewrap for aarch64](https://archlinuxarm.org/packages/aarch64/bubblewrap) and usable user/network namespaces. Software transport tests bind local ephemeral loopback ports and the THP fixture uses process-level memory controls. Run them in an environment that permits those operations; do not weaken the assertions to accommodate a restricted tool sandbox.

For Arch Linux ARM, the official package catalogs link [base-devel](https://archlinuxarm.org/packages/aarch64/base-devel), [CMake](https://archlinuxarm.org/packages/aarch64/cmake), [Ninja](https://archlinuxarm.org/packages/aarch64/ninja) and [OpenSSL](https://archlinuxarm.org/packages/aarch64/openssl). These are acquisition references, not frozen package versions or evidence that every current repository version is qualified. The executors do not silently install or upgrade host packages.

Before scheduling a build, check tool discovery in the same PATH that its unit will use:

```sh
command -v git python3 cc c++ cmake ninja systemd-run unshare
```

The observed toolchain is GCC/G++ 16.1.1, CMake 4.4.4 and Ninja 1.13.2. Explicit Oryon compiler flags must be supported by the selected compiler. Record actual versions and resulting binary hashes, then qualify the new build. A private virtual environment on the development laptop is not an installation prerequisite; if using a virtual environment, make its actual tool executables discoverable explicitly.

Use the exact source/patch/build inputs in the [Flash reconstruction guide](FLASH-REBUILD.md), [classic and image source provenance](SOURCE-PROVENANCE.md), [runtime setup](lab-notes/RUNTIME-SETUP.md) and [bootstrap source catalog](../bootstrap/source-recipes.json). The catalog separately pins KleidiAI v1.32.0 and the classic browser UI archive; archive size/SHA verification is required before configure. Model repositories, immutable revisions, filenames and checksums are in the [model manifest](../dependencies/model-artifacts.json). Preserve upstream notices and model licenses. Qualcomm runtime and compiled model bundles remain separately acquired external dependencies; see the NPU helper guide (lab note omitted from this release).

## Image assets

| Local asset | Publisher / pinned revision | Bytes | Publisher SHA256 | Local evidence |
| --- | --- | ---: | --- | --- |
| FLUX.2 klein 4B Q8 (**default** image model) | [leejet/FLUX.2-klein-4B-GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/tree/3b1f5a9dc3abb32238b053aeb3d823c30afdacbd) | 4300629440 | `0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28` | Apache-2.0 on the 4B model card (checked 2026-11-12). Same file as Studio's klein path. The 9B variant is not in the default set. |
| Qwen3-4B Q8_0 text encoder (klein) | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/22c9fc8a8c7700b76a1789366280a6a5a1ad1120) | 4280405792 | `eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d` | Apache-2.0 (repository card). Same file Studio's klein path uses. |
| FLUX.2 VAE for klein | [Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/tree/5b3abca5c0dc1ecd4eaf3340868d09063fc1c965) | 336211292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` | Apache-2.0 (repository card). |
| Z-Image Turbo Q4 (opt-in, `--image-model z-image`) | [leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/c61c0e422dc8b541b7548cf33a4ef8302b0f8085) | 3864250304 | `14b375ab4f226bc5378f68f37e899ef3c2242b8541e61e2bc1aff40976086fbd` | Existing HF download receipt plus subsequent full local SHA256 and size verification match the pinned publisher artifact. |
| Z-Image Turbo Q8 | Same pinned catalog | 6577440704 | `df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1` | Prior setup-worker full-SHA verification and ready receipt; current size matches. Not rehashed here. |
| Image text encoder Qwen3-4B-Instruct-2507 Q4_K_M | [Unsloth catalog](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/a06e946bb6b655725eafa393f4a9745d460374c9) | 2497281120 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` | Existing HF download receipt plus subsequent full local SHA256 and size verification match the pinned publisher artifact. |
| Compatible VAE, `split_files/vae/ae.safetensors` | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo/tree/6fc90a3b1b653e935a0d175e260736de25b84df5) | 335304388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` | Local `vae/vae.safetensors` now matches the full SHA256 and size of this artifact. Historical download origin remains unestablished. |

The pinned image engine's [Z-Image instructions](https://github.com/leejet/stable-diffusion.cpp/blob/5a1ba5d895bec4e07deab80d6d791721cd59d9c2/docs/z_image.md) identify the Qwen text encoder and FLUX.1-schnell VAE. The official [FLUX.1-schnell catalog](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/741f7c3ce8b383c54771c7003378a50191e9efe9) reports `ae.safetensors` at the same size but masks its gated LFS object hash to an unauthenticated metadata request. The public Comfy catalog supplies a concrete compatible artifact. The subsequent local full checksum confirms content equality; this does not prove which host originally supplied the file. No asset was replaced.

Each listed publisher card currently declares Apache-2.0. These card fields describe those pinned publisher artifacts; they do not relicense unrelated weights. Retain the actual downloaded model's applicable license/notice and access terms separately from this project's MIT wrapper.

### Local content verification completed after inference

The [checksum receipt](../dependencies/image-asset-checksums.json) records the completed one-pass Q4/text-encoder/VAE check: all expected byte counts and SHA256 values matched. The operation ran only after heavy units were terminal and frontends idle, holding the canonical exclusive lease throughout. Files retained the same regular-file type, size, inode and modification time during hashing. The three reads completed in 6.01 seconds; that is verification timing, not inference performance evidence. Q8 retains its separate prior full-hash receipt and was not redundantly rehashed.

This closes the local image-content identity gap. A different installation must perform its own checks against the pinned manifest before enabling validation gates. Content equality identifies the compatible VAE bytes, not their historical download origin. No generated image, upload, prompt or chat was inspected. No model file was changed, substituted or downloaded.

## Source licenses actually inspected

| Component | Inspected text / status |
| --- | --- |
| Original wrapper/tools | [LICENSE](../LICENSE), MIT. |
| llama.cpp patch | Upstream MIT notice retained in [third_party/llama.cpp-LICENSE](../third_party/llama.cpp-LICENSE); [source pin/patch provenance](../patches/llama-source-provenance.json). |
| stable-diffusion.cpp patch | Upstream MIT notice retained in [third_party/stable-diffusion.cpp-LICENSE](../third_party/stable-diffusion.cpp-LICENSE); [source audit](SOURCE-PROVENANCE.md). Its pinned leejet ggml also has an MIT LICENSE. |
| Image engine source dependencies | Inspected libwebp COPYING (BSD three-clause text) plus separate PATENTS grant; libwebm LICENSE.TXT (BSD three-clause); darts_clone notice (three-clause-style text); utf8proc MIT/data notices; stb_image embeds MIT/public-domain alternatives. These source trees are not copied into this public wrapper repository. |
| KleidiAI v1.32.0 | Fetched source includes Apache-2.0 and BSD-3-Clause license texts; README SPDX is Apache-2.0. Preserve actual per-file notices when assembling an engine package. |
| Experimental BigMoeOnEdge | **Apache-2.0, not MIT**, at [pin 930b2c9ebd8c](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/LICENSE). Its source/binary is an external experimental dependency, not included or relicensed here. |
| BigMoe third parties | Its [THIRD_PARTY_LICENSES](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/THIRD_PARTY_LICENSES) inventories MIT llama/ggml, cpp-httplib and nlohmann/json; Monocypher uses the BSD-2-Clause choice from its dual BSD/CC0 terms. Optional UI/mobile/font/CUDA components have additional notices; their presence in that upstream list does not mean our CPU server includes every component. |
| vendor runtime/SDK/model bundles | Proprietary/external terms; no redistribution under this repository's MIT license. |

License-text hashes and precise scope notes are in the JSON inventory. No full engine source or compiled engines are bundled in this repository. Before any future binary package, inventory the **actual linked/distributed** components, preserve all relevant upstream license texts/notices and review model terms. This factual source inventory is not a complete binary bill of materials or a publication approval.
