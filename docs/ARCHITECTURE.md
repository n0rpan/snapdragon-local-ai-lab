# Architecture

How the pieces fit, and why each design suits this chip. Numbers are cited from [RESULTS.md](RESULTS.md) with their evidence tags.

```
 clients (OpenCode, pi, dsh, any OpenAI SDK)
          │  http://127.0.0.1:8092/v1   (loopback only)
          ▼
 ┌────────────────────────── controller (this repo, Python) ──────────────────────────┐
 │ OpenAI API + aliases · request queue (one at a time) · model selection / lazy load │
 │ idle unload · prefix-store policy · memory admission + guard · inference lease     │
 └──────────────┬───────────────────────┬──────────────────────────┬──────────────────┘
                ▼                       ▼                          ▼
        oxe-server --engine oxe   oxe-server --engine flash   llama.cpp / sd.cpp (patched)
        Ornith, Qwen3.6 (35B MoE) Flash-Next (125B MoE)       27B dense fallback, images
        weights mmapped in RAM    dense in RAM, experts on SSD
                                  ▲
                                  │ O_DIRECT reads, IO core 8
                                NVMe (expert sidecar file)
```

## 1. The chip, in one paragraph

Twelve identical Oryon cores in three clusters of four, each cluster sharing an L2. NEON with `i8mm` (int8 matrix multiply, `SMMLA`) and dot-product instructions; no SVE, no SME. LPDDR5X delivers about 108 GB/s of streaming reads to eight packed cores (115.85 GB/s to six spread cores) once the memory-frequency votes work. NVMe direct reads reach 3.7 GB/s raw at queue depth 5 and plateau near 4.6 GB/s at high depth; through LUKS encryption 2.4 GB/s at depth 5, with ~0.4 core-seconds of decryption per GB on the IO cores. Where the expert file lives matters. 32 GB of RAM. Under sustained all-core load the clusters throttle unevenly: cluster 0 holds 3.4 GHz, cluster 2 drops to about 2.5 GHz at ~95 °C.

A 35B MoE that routes ~3B parameters per token reads 1.8–2.6 GB of weights per token. **Decode is bandwidth-bound; prefill is int8-kernel-bound.** Most choices below follow from that.

## 2. `oxe`: the MoE engine (Ornith, Qwen3.6)

Repository: snapdragon-engine, crates `kernels-sys`, `gguf`, `sched`, `model`, `tokenizer`, `chat`, `server`.

- **Narrow by design.** It supports the `qwen35moe` architecture family (hybrid attention + gated delta-net recurrent layers, routed + shared experts, MTP head) and nothing else. A general graph executor walks ~3,900 nodes per token with a barrier each; `oxe` lays out a **static plan** at load and crosses about 170–210 barriers per token. Nothing is allocated, locked or logged in the per-token loop.
- **Threads.** A persistent pinned pool with a sense-reversing spin barrier. Decode on 8 packed threads (cores 0–7): more threads add synchronisation and no bandwidth. Prefill on all 12: it is compute-bound.
- **Kernels.** ggml's CPU quant kernels are vendored (`vendor/ggml-subset`, MIT) behind a hand-written FFI and repacked at load into layouts `SMMLA` consumes directly. Bit-exactness against native-layout llama.cpp is the reference.
- **Batched prefill.** 256 tokens per layer pass; dense projections become GEMMs. A kernel bitmask (`mask 1021` today) selects tile shapes, e.g. dense Q4_K in 8-token × 4-row tiles, which keeps the decoded weight block in L1 while 8 tokens use it. "Groups" run several ubatches through a layer together so routed experts see more rows per pass (the unpack cost per expert is worth about 3 token rows).
- **Power-of-two aliasing.** At n_ctx 16,384/32,768 the attention score columns all fell into one L1 set (19.8 vs 56 GFMA/s). Scratch strides are padded. This only showed at production shapes, so every benchmark here uses them.
- **Tiled attention.** Several query tokens per pass over each KV block: the KV cache (20.5 MB per 1K positions) is read from DRAM fewer times. Matters most on long prefixes, i.e. agent sessions.
- **MTP + verify lanes.** A multi-token-prediction head drafts one token (Qwen3.6's own head; for Ornith a head grafted from Qwen3.6, since Ornith's own head accepts too little); one verify pass checks it. On a MoE the verify reads the union of the experts both tokens route to, so the second token used to cost 0.74 of a plain step. The verify-lane kernels bring that to 0.39, which makes depth 1 a win at every measured context (−12% to −26% decode time, identical tokens). Depth ≥2 still loses.
- **Tool calls.** The `chat` crate renders the model's chat template (minijinja, llama.cpp-compatible) and parses Qwen3-Coder XML tool calls. Streamed and non-streamed paths give the same result, and Python-style `True`/`False`/`None` in a boolean parameter is coerced instead of dropping the call (a real bug found in production).

## 3. Flash: SSD streaming for a model 2.7× larger than RAM

Repository: snapdragon-engine, crates `qwen4exp` (Flash graph, expert cache, IO) and `server` with `--engine flash`. Model: Qwen3.8-Flash-Next, 125B main weights, 512 experts per layer, top 10 routed, UD-IQ3_XXS, 83.6 GB.

- **What stays where.** Dense weights (about 4 GB) and the router are resident; they are read every token. Experts live in a repacked sidecar file on NVMe and move through a RAM **expert cache** of 14–20 GiB.
- **Cache policy.** LFU with per-layer quotas (−5.5…−6.2% token time vs Clock). Routing is skewed, so a 16–21 GiB cache hits about 91%. Online policies are already ~2.1× the Belady optimum; the remaining gap is *prediction*, not eviction (see [help wanted 3](HELP-WANTED.md#3-flash-expert-prediction-at-small-caches)).
- **IO.** O_DIRECT reads at high queue depth from a dedicated IO thread on core 8; compute stays on cores 0–7 so the IO core's cluster has no compute threads (any compute thread there raises read latency ~27%). Misses for a layer are issued at once and overlap with compute.
- **Decode** is one dependency chain per token; the remaining time is expert compute plus the latency of misses, not scheduling (an own executor bought −0.18%).
- **Prefill is layer-major.** A whole batch goes through one layer while that layer's experts are resident, instead of each token dragging every layer through the cache. Experts are processed in waves; **order 2** loads residents first, then misses, each sorted by the last prompt token that uses them, so the decode right after a short prompt finds its hot experts still cached (−4.95% decode vs the plain order, −37% repeat-pass reads).
- **Long-prompt guard.** A multi-token pass checks `MemAvailable` per chunk and stops cleanly before the host floor.
- **Numerics.** The fast path (8-thread graph, fixed-chunk attention, shared expert first) is reorder-class: mean KLD 6.5e-4 against the exact path, within the measured noise band; an `--flash-exact` path remains as the reference.

## 4. `oxi`: the image engine (experimental)

Repository: snapdragon-engine, image branch, crates `int8-gemm`, `image-dit`, `image-flow`, `image-vae`. A Rust port of FLUX.2 klein (DiT + VAE) with an f32 reference path and an int8 `SMMLA` fast path (per-row / per-token chunk scales, fixed accumulation order). The reference is within its own 1-ulp noise of stable-diffusion.cpp. The int8 linears are 4–5× faster per block; with int8 attention as well the image fails the quality gate (22.75 vs 27.81 dB), with f32 attention it passes on three seeds at 512 px (31.5–34.1 dB). It is not timed end to end or deployed yet, so production images still use patched stable-diffusion.cpp.

## 5. The controller

Source: `src/snapdragon_ai_lab/runtime/` (`controller.py`, `openai_api.py`, `memory_policy.py`, `prefix_store.py`, `oxe_runtime.py`, `request_ledger.py`, …). Runs as a systemd user service.

- **One port, OpenAI wire format.** `/v1/models`, `/v1/chat/completions` (streaming, tools, usage), model aliases (`local-default`, `local-heavy`), litellm-style `provider/` prefixes. Details: [OPENAI-API.md](OPENAI-API.md).
- **One model at a time, one request at a time.** A FIFO queue holds further requests up to 120 s, then answers 503 with `Retry-After`. A request for another model unloads the current one. Models load on first use and unload after 600 s idle.
- **Inference lease.** Every engine process, benchmark and test that loads real weights holds one `flock` lease. Two heavy jobs never share the 32 GB.
- **Harness compatibility.** Request normalisation for OpenCode, pi and dsh (e.g. Flash requests are brought into its qualified range: temperature 0, ≤2,048 output tokens, thinking off; each change is counted in `/_local/status`).

## 6. Caching and sessions

All exact (lossless): they restore state, they never approximate it.

| Layer | What it keeps | Where | Effect |
|---|---|---|---|
| Prefix store | the leading system prompt + tool schemas, keyed by model/build/template/schema identity | disk, owner-only, size-bounded | first prompt of a session 30.6 → 1.2 s [S] |
| Parked sessions | a whole conversation's KV + recurrent state when a side request (title, subagent) displaces it | RAM, 2 GiB | 32K session resumed in 42 ms [M] |
| Turn checkpoints | state at message boundaries, so an edited or re-rendered history resumes from the last matching turn | RAM, 8 slots | −17.8% new prompt tokens in a recorded session [M] |
| Session cache | turn checkpoints written to disk at idle unload | disk | 2,304 / 2,304 tokens reused after reload [M] |

Why this matters more than kernels: state for these models is ~61 MiB + 20.5 KB per token and copies at tens of GB/s, while recomputing a 10K-token conversation takes about a minute. One limit: the hybrid recurrent layers cannot be position-shifted, so llama.cpp-style `--cache-reuse` (KV shifting) cannot work here; reuse is by exact prefix only.

## 7. Memory admission

Swap is off for the model service, so running out of RAM means a guard stop, never thrashing.

- **Scopes.** Classic models run in the inference service with `MemoryMax=28G`, `MemorySwapMax=0`; Flash runs in its own 29 GiB scope. Desktop pages may swap to an NVMe swapfile with a small zswap pool.
- **Guard.** One guard per engine process stops it below the host floor (1.5 GiB for classic profiles, 1.0 GiB for Flash), on corroborated sustained memory-pressure stalls, or on any swap/OOM event of the service. Wrappers call the guard; they never carry a second copy with another threshold.
- **Flash admission is measured.** A cache of C GiB is admitted only if `MemAvailable` at launch covers C + the measured load overhead + measured model growth during a long session (3.5 GiB) + measured harness growth beside it (0.8 GiB) + the 1 GiB floor. Today that is 24.1 / 26.1 / 28.1 / 30.1 GiB free for the 14 / 16 / 18 / 20 GiB tiers [D].
- **Tiers.** Production runs a fixed 14 GiB tier (qualified, 22.28 GiB cgroup peak, 3.64 GiB host minimum on the protected task). The engine also has `--flash-cache-gib auto` (largest tier that fits, else refuse with exit 74); a "preferred tier" flag that falls back to the largest fitting tier and reports why is built but not installed. Larger tiers are faster per token but need a quieter desktop: 20 GiB needs 30.1 of 30.7 GiB free.

## 8. Production discipline

Every change declares its class (lossless / reorder / lossy) before it is measured, is frozen as a binary with its SHA-256 before the run, and is installed only after: a protected OpenCode task (13/13 checks), a post-install regression through the installed service, and a one-command rollback written and checked first. See [RESULTS.md §2](RESULTS.md#2-method).
