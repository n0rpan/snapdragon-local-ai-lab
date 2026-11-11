# Image engine source identity

The tested CPU repack build comes from [stable-diffusion.cpp commit 5a1ba5d](https://github.com/leejet/stable-diffusion.cpp/tree/5a1ba5d895bec4e07deab80d6d791721cd59d9c2) and its pinned [leejet/ggml commit 135ce1e0](https://github.com/leejet/ggml/tree/135ce1e0e3f2b0693d3de93d4f6b5fa0a1947820). This is the ggml fork named by the parent gitlink, not a guessed contemporary ggml-org revision.

[Patch metadata](../patches/manifest.json) and [source audit](../patches/source-provenance.json) record the six exact preimage/postimage hashes. Replaying the published patch on pristine files reproduces all six snapshot files byte for byte. All 2353 tracked ggml blobs match. The root tree has 754 tracked blobs; its only other difference is declared Windows batch-file line-ending normalization. libwebp and libwebm source trees were also checked; the only media difference is a Windows batch file's line endings.

Reconstruct source in a separate directory:

```sh
git clone https://github.com/leejet/stable-diffusion.cpp.git sd-source
git -C sd-source checkout 5a1ba5d895bec4e07deab80d6d791721cd59d9c2
git -C sd-source submodule update --init --recursive
git -C sd-source apply /absolute/path/to/snapdragon-local-ai-lab/patches/sd-cpu-repack.patch
```

The read-only verifier accepts a pristine Git checkout, patched snapshot and patch path:

```sh
python tools/verify_image_source.py /path/to/pristine /path/to/patched /absolute/path/to/patches/sd-cpu-repack.patch
```

It checks pinned identities, tracked code, patch replay and allowed non-compiled line-ending differences. It performs no build or inference and exits unsuccessfully on a mismatch. Inputs remain unchanged; replay occurs in a temporary directory.

The tested build used GCC 16.1.1, CMake 4.4.4, static linking, CPU-only backends, OpenMP and CPU repacking. KleidiAI was off for this image build. Its architecture flags were:

```text
-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME
```

Preprocessor inspection confirms FP16 scalar/vector/FML, BF16 scalar/vector and I8MM support with these flags. Model weights are separately obtained dependencies. `SD_CPU_REPACK_EXPERIMENT=1` enables the loader patch; without it the patch is inactive.

The measured and deployed binary has SHA256 `e59a180c339266ff8b29e31c3ed5ae6b68ddb05ea1de9e199bdbf9ecea44f606`. Source identity is resolved; a bit-for-bit independent binary rebuild is **not** claimed. The tested snapshot omitted Git metadata, so a normal checkout can embed different version strings. Dependencies, toolchain and build environment must also be locked before claiming binary reproducibility. Native Q8 quality and performance validation remains separate from this source audit.
