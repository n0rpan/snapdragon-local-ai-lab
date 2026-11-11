# Which SSD/RAM optimizations transfer?

The placement principles transfer to other models; the useful implementation depends on whether the model is dense or MoE, fits RAM, and can use a fast packed CPU layout. Keep the current resident, repacked coding profiles. Flash-Next's selective SSD reads solve a larger-than-RAM problem and are not an established speed upgrade for those profiles.

This is a read-only applicability audit dated 2026-11-08. No model, build, benchmark, download or runtime change was made. Only this new note was written. The repository controller/prefix source matched deployed runtime snapshot `699298b35556db473f08` at audit time.

## Current coding profiles

The selected `qwen-local` profile is **Qwen3.6-35B-A3B MoE**, with a 22,663,387,424-byte MTP UD-Q4_K_M file. The selectable dense profile is **Qwen3.8-27B**, with a 16,547,400,064-byte Q4_K_M file. Both use the qualified CPU engine SHA `1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1`: llama.cpp `954de248ac6907dd05dc19f0f511ab18f2807a54` plus the small-batch-pool patch. The audited build enables CPU repack, OpenMP, KleidiAI v1.32.0 and ARM DOTPROD/I8MM/FP16/BF16, with GPU off. These are recorded identities, not new payload checksum runs. [Profiles, weights and build](RUNTIME-SETUP.md), [source provenance](../patches/llama-source-provenance.json).

Both profiles already use repacking, 16K context, F16 KV, four decode/ten batch threads, batch/microbatch 256, one slot and no warmup inference. MoE retains MTP with two draft tokens and `LLAMA_CPU_SMALL_BATCH_LIMIT=3`; dense retains no speculation and limit 1. The patch keeps small speculative verification batches on the generation thread pool. The controller merges the selected profile over its top-level defaults, so inspecting only the top-level dense fields would miss MoE's MTP settings. [Pool patch](../patches/llama-small-batch-pool.patch), [profile selection](../src/snapdragon_ai_lab/runtime/controller.py).

KleidiAI being compiled in does not mean every expert uses it: this pinned implementation accepts `MUL_MAT` and rejects other operations at its entry gate. Actual dispatch also depends on tensor type, shape and buffer. Preserve native/repacked MoE expert handling; do not attribute its whole-model performance to KleidiAI alone. [Pinned KleidiAI gate](https://github.com/ggml-org/llama.cpp/blob/954de248ac6907dd05dc19f0f511ab18f2807a54/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp#L1793).

## Transfer boundaries

| Mechanism | Current resident coding profiles | Where it can help elsewhere |
|---|---|---|
| Exact expert SSD cache | Adds miss I/O and cache/hook overhead to weights that currently fit; can also require giving up repack | Larger MoE models, larger quants or tighter RAM budgets, if selected experts are contiguous/addressable and the actual CPU kernel accepts their raw layout |
| Dense/shared weights resident, sparse tables lazy | Ordinary packed CPU buffers are already useful; there is no established Flash-sized n-gram table to move in these profiles | Models with genuinely sparse lookup tables or routed experts. Dense layers still need their full weights for each token |
| Suppress eager startup population | A plausible load-time/memory-pressure lead; not yet qualified for these resident profiles | Especially oversized mapped files when a later policy intentionally copies only a subset or reads sparsely |
| Sparse-bank THP control | Do not copy Flash's process-wide THP setting automatically | Sparse anonymous expert-bank windows where huge pages amplify physical residency; contiguous resident arrays have different tradeoffs |
| More expert I/O workers or wider streaming prefill | No expected warm-decode benefit when expert weights are resident | Workloads actually blocked on expert misses and able to reuse loaded experts across enough prompt rows |
| MTP and prompt-prefix reuse | Already applied to MoE and the classic controller respectively | Compatible models/backends with verified draft rollback and cache identity; additional state must fit RAM |

Exact caching does not itself drop experts or change weights. Pruning, substitute routing and lower-bit quantization are separate accuracy choices. The deployed mixed GGUF labels also do not establish every tensor's type. The raw IQ2/IQ3 multirow prototype therefore does not automatically accelerate these Q4-labelled resident models; their packed kernels and MTP path must be preserved. [Existing transfer/qualification results](TRANSFERRED-IDEAS.md), [streaming memory contract](STREAMING-MEMORY.md).

For a dense model there is no expert-selection saving: dense FFN weights participate each token. SSD-backed dense execution can extend capacity, but repeatedly fetching those weights is a capacity/latency tradeoff, not the same locality opportunity as caching routed experts.

## Startup mapping is a separate question

The current controller starts a model only when needed, and `--no-warmup` skips a dummy inference. Neither disables file population. The pinned common defaults choose load mode AUTO; on this CPU path the loader uses mmap, calls `init_mappings(true, ...)`, and Linux applies `MAP_POPULATE` when prefetch is requested without lazy ranges. It also issues WILLNEED advice. [Model load](https://github.com/ggml-org/llama.cpp/blob/954de248ac6907dd05dc19f0f511ab18f2807a54/src/llama-model.cpp#L1988), [mapping](https://github.com/ggml-org/llama.cpp/blob/954de248ac6907dd05dc19f0f511ab18f2807a54/src/llama-mmap.cpp#L482).

For ordinary native buffers the loader can retain mapped weight pages. For an already allocated specialized buffer, it calls `ggml_backend_tensor_set`, allowing the selected backend to copy/pack weights into its own storage. Thus mmap, anonymous CPU copies and repacking are different stages, not mutually interchangeable flags. A populated source mapping can contribute transient file-backed pressure alongside packed allocations; the loader later releases unused mapping fragments. [Buffer selection](https://github.com/ggml-org/llama.cpp/blob/954de248ac6907dd05dc19f0f511ab18f2807a54/src/llama-model.cpp#L1260), [mapped load/copy](https://github.com/ggml-org/llama.cpp/blob/954de248ac6907dd05dc19f0f511ab18f2807a54/src/llama-model-loader.cpp#L1713).

Flash's isolated mapping change suppresses that initial population before BMoE applies its selective policies. The same idea could reduce startup double-touch/page-cache pressure for a resident repack build, but every weight that needs packing still has to be read. It could also move faults into loading/first inference or lose useful sequential prefetch. Compare cold load, warm page-cache reload, first useful response and peak RAM before considering it for classic profiles. A warm steady-generation benchmark alone cannot settle that question. No such classic comparison was run for this note.

## Planning-profile memory assumptions

The classic controller's admission estimate is model-file bytes plus 3 GiB; it is a guard, not a proof of every context/repack/MTP peak. Prefix storage has a 3 GiB **disk** cap and stores only leading system/tool instructions, not a second expert bank or a 3 GiB RAM allocation. [Admission/launch](../src/snapdragon_ai_lab/runtime/controller.py), [prefix policy](../src/snapdragon_ai_lab/runtime/prefix_plan.py).

A Flash planning profile needs its own measured admission record: resident dense/shared weights, physical expert-cache residency and rounding, sparse mapped pages, context/recurrent state, work buffers, and any speculative head all count. An expert-cache budget is not a process limit; an 83 GB mapped file is not 83 GB resident RAM either. Retain the validated context/cache/precision, canonical exclusive lease, unload-before-switch behavior and host headroom until the complete agent task qualifies. Do not silently replace the fast resident MoE default, enable MTP for Flash, increase context or co-load both models merely because a profile can be displayed. [Memory accounting](STREAMING-MEMORY.md), [existing runtime lifecycle](RUNTIME-SETUP.md).

The concrete transferable lead is startup/memory accounting while preserving packed resident execution. SSD expert streaming remains useful for models that exceed the memory budget; adopting it for a model that fits needs a demonstrated finished-task benefit.
