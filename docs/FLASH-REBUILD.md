# Rebuild the qualified Flash engine from public source

**Source replay is verified. A fresh binary and clean installation remain unqualified.** The installed planning engine was built from the public sources below plus one demand-mapping change. Its historical server SHA256 is `8588e6b619f3292a412504f070e67b994162ad706ae1dfc5e09359a24f341773`; this is provenance, not an expected hash for every new toolchain. No upstream/private source commit needs publishing to reconstruct the code.

## Public source chain

| Component | Public repository | Revision |
| --- | --- | --- |
| Parent | [Helldez/BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) | `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` |
| llama fork | [Helldez/llama.cpp](https://github.com/Helldez/llama.cpp) | `7ac98f4a8089a36c0581b7c6e8973417a8fa175c` |
| Local source delta | [flash-demand-mapping.patch](../patches/flash-demand-mapping.patch) | SHA256 `50eab1d270721d5e0d1215ad858324502308bb6be855af6b73108457b3d539cc` |

Both public commit metadata endpoints returned HTTP 200 with the expected revisions and trees. The llama fork already includes the expert-ready hook. Apply the mapping patch in a separate standalone fork, then put that local fork beneath the new parent's llama gitlink. The only code change is `init_mappings(true, ...)` → `init_mappings(false, ...)` in `src/llama-model.cpp`; there is no other parent source delta. The source patch preserves routing, tensor storage, all experts and arithmetic.

The patched llama tree is `4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef`. Replaying the historical parent gitlink change also reproduced its exact qualified tree. [Portable provenance](../dependencies/flash-source-provenance.json) records this evidence. Historical private commit IDs are not acquisition inputs: a local reconstruction commit can have a new identity while containing the same source.

[The source inventory](../dependencies/flash-source-inventory.json) expands 5,253 tracked parent/llama blobs with modes and Git object IDs. Its flattened content digest is `142a3bb2d27bc75761b51d018a541cb299258ef57ca71f17e257005324147941`. Git/build metadata and gitlink commit identity are excluded. The verifier checks the index, working tree, actual blob bytes, modes and symlink text without following symlinks. Ten explicitly listed Windows scripts permit only the CRLF checkout normalization declared by the pinned parent's `.gitattributes`; Linux compile inputs receive no such normalization.

## Reconstruct in a new directory

Choose new source/build directories and the location of this complete repository or installed resource bundle. Source acquisition and the later compile must run in separately scheduled, guarded host windows. These commands do not install or start a service.

```sh
FLASH_WORK=/absolute/path/to/new/flash-rebuild
FLASH_LAB=/absolute/path/to/snapdragon-local-ai-lab
mkdir -p "$FLASH_WORK"

# Public HTTPS only: no inherited credential helper, headers, hooks or identity.
flash_git() {
  env -u GIT_ASKPASS -u SSH_ASKPASS -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= -c core.askPass= -c http.extraHeader= \
        -c core.hooksPath=/dev/null "$@"
}
flash_git init "$FLASH_WORK/parent"
flash_git -C "$FLASH_WORK/parent" remote add origin https://github.com/Helldez/BigMoeOnEdge.git
flash_git -C "$FLASH_WORK/parent" fetch --depth 1 origin 930b2c9ebd8c2cface6566bd9f11075ac07adfe0
flash_git -C "$FLASH_WORK/parent" checkout --detach FETCH_HEAD
flash_git init "$FLASH_WORK/llama-fork"
flash_git -C "$FLASH_WORK/llama-fork" remote add origin https://github.com/Helldez/llama.cpp.git
flash_git -C "$FLASH_WORK/llama-fork" fetch --depth 1 origin 7ac98f4a8089a36c0581b7c6e8973417a8fa175c
flash_git -C "$FLASH_WORK/llama-fork" checkout --detach FETCH_HEAD
flash_git -C "$FLASH_WORK/llama-fork" apply --index "$FLASH_LAB/patches/flash-demand-mapping.patch"
test "$(flash_git -C "$FLASH_WORK/llama-fork" write-tree)" = 4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef

# A local BUILD-ONLY fork commit; no global Git config, upstream attribution or push.
flash_git -C "$FLASH_WORK/llama-fork" -c user.name='Local source reconstruction' \
  -c user.email='source-reconstruction@invalid' -c commit.gpgsign=false \
  commit --no-gpg-sign -m 'build: reconstruct qualified demand mapping'
flash_git clone --no-hardlinks "$FLASH_WORK/llama-fork" "$FLASH_WORK/parent/third_party/llama.cpp"
flash_git -C "$FLASH_WORK/parent" add third_party/llama.cpp
python3 "$FLASH_LAB/tools/verify-flash-source.py" "$FLASH_WORK/parent" \
  --inventory "$FLASH_LAB/dependencies/flash-source-inventory.json"
```

The anonymous fetches request complete shallow source trees without partial-clone filters. The read-only verifier disables lazy fetching. It also locates its inventory automatically in the repository layout or beneath `<prefix>/share/snapdragon-local-ai-lab/dependencies`; `--inventory` supports other installed resource layouts. No private source directory, old build archive or user Git configuration is required.

The optional parent webview gitlink is recorded but unused with desktop disabled. llama has no nested submodules at this pin. Common, ggml, Jinja, mtmd, JSON and cpp-httplib are pinned inside its tree; Monocypher is inside the parent tree. Do not substitute latest dependencies. No npm frontend or prebuilt UI asset is required for this bmoe loopback API build: upstream llama's server/tools/app targets are disabled, and bmoe's own asset step only stages tracked catalog/skills plus an optional already-built frontend.

## Build with the qualified CPU configuration

Historical tools were GCC/G++ 16.1.1, CMake 4.4.4 and Ninja 1.13.2 on Linux aarch64. Prerequisites include Python 3, Git, system C/C++ development libraries and OpenMP/libgomp. [The build plan](../dependencies/flash-build-plan.json) contains portable argument vectors and the effective recorded options. Project libraries are static; a fully static executable independent of libc/C++/OpenMP is not claimed.

The host supervisor must own the canonical inference lease through compiler exit, child reaping and recovery. Schedule one compile job, CPU quota 200%, MemoryMax 4 GiB, swap max 0, TasksMax 160 and a finite 45-minute deadline. Require external power, the reviewed battery guards, at least 8 GiB MemAvailable at compile start and the actual 2 GiB reserve throughout. A bare command below does not supply those protections; run it inside that supervisor. Do not open a separate child lock to replace the inherited lease.

```sh
FLASH_FLAGS='-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME'
cmake -S "$FLASH_WORK/parent" -B "$FLASH_WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++ \
  -DCMAKE_C_FLAGS="$FLASH_FLAGS" -DCMAKE_CXX_FLAGS="$FLASH_FLAGS" \
  -DBMOE_BUILD_CLI=ON -DBMOE_BUILD_SERVER=ON -DBMOE_BUILD_DESKTOP=OFF \
  -DBMOE_BUILD_TESTS=ON -DBMOE_BUILD_TOOLS=OFF -DBMOE_BACKEND_MODULES=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_RPC=ON \
  -DGGML_VULKAN=OFF -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_LLGUIDANCE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$FLASH_WORK/build" --parallel 1 --target \
  bmoe-cli bmoe-server bmoe_config_test bmoe_params_test \
  bmoe_mapping_release_test bmoe_file_reader_test bmoe_moe_gates
```

Repack remains off because the streamer rebinds native tensor pointers. RPC is compiled for the historical host-test configuration, not exposed as a runtime endpoint. GPU/NPU backends, backend modules and KleidiAI remain off. Generated `common/build-info.cpp`, `ggml-version.h` and other configure outputs must come from these source templates; do not copy private generated files. New local commit/count/dirty/compiler strings can change embedded versions and binary hashes.

Run the four model-free config/params/mapping-release/file-reader checks after building. Building `bmoe_moe_gates` does not run or pass its model-based byte-identity gates. Schedule those and fresh actual model/profile qualification separately: correct completed functional/tool/agent tasks, unchanged full expert routing, actual RAM/swap/power guards, EOF/reaping/unload and recovery. Only then may the new executable/profile become ready/selectable. See [bootstrap parity](BOOTSTRAP-PARITY.md) for the clean-install finish gate.

## Separate later changes and notices

The production reconstruction ends after demand mapping. Attribution and summary-counter patches are diagnostic layers with separate inventories; frequency-cache policy, 13/14 GiB cache trials, selective Q8 and any backend work remain separate candidates requiring their own measured gates. They are not silently applied here.

BigMoeOnEdge is Apache-2.0. llama.cpp is MIT with bundled dependency notices. Preserve upstream license files and per-file notices when fetching, building and distributing; the MIT wrapper does not relicense their code. This mapping patch removes no notice. No Qualcomm libraries, model weights or engine binaries are shipped by this reconstruction package.
