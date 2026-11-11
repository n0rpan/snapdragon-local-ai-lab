# Pthread caller-affinity regression

Experimental; not installed. Pinned base 954de24 plus the installed runtime patch stack, `GGML_OPENMP=OFF`.

An unpaused pool can be called after another pool or caller changed the dispatch thread's affinity. Its workers retain their masks, but kickoff previously restored the caller mask only when resuming a paused pool. The observed 12-thread batch had workers on 0–10 and its caller still on 7, with 11 unused; two busy workers shared 7.

`patches/llama-pthread-main-affinity.patch` restores the main-worker affinity on every pthread dispatch. This does not change the OpenMP branch. The regression test alternates two unpaused pools, deliberately changes caller affinity, then checks the selected CPU mask and exact matrix output. Compile it against the candidate's `ggml-cpu`, `ggml-base`, KleidiAI and vendor-hash static archives, linking pthread/dl/m. It prints `ALL PASSED`. CPUs 0–2 must be available.

The 4K KLD against the original model reference is 0 with 100% top-1. That proves numerical qualification at the tested shape; it does not establish a speed win. Batch 12 prefill still loses to OpenMP, so the candidate remains experimental pending 8/8 and real-controller tests. [Diagnosis and numbers](../../benchmarks/results/ornith-pthread-affinity-diagnosis.json).
