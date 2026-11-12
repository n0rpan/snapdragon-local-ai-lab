# Contributing

Thanks for looking. This is a lab: we try ideas, measure them honestly and keep the failures. Contributions that come with a measurement, or that make measuring easier, are the most useful. Good places to start: [docs/HELP-WANTED.md](docs/HELP-WANTED.md).

## Two repositories

- **snapdragon-local-ai-lab** (this one): controller, OpenAI-compatible API, harness integrations, benchmark and qualification tools, results and docs. Python.
- **snapdragon-engine**: the Rust engines (`oxe` MoE, Flash SSD streaming, `oxi` images) and their vendored kernels.

Open issues where the code lives; if unsure, open it here.

## Ground rules for performance work

1. **Correctness first.** Declare the change's class before measuring: *lossless* (bit-identical / identical tokens), *reorder* (same maths, different float order: within measured noise) or *lossy* (opt-in only, with a quality metric and a task check). See [docs/RESULTS.md §2](docs/RESULTS.md#2-method).
2. **Interleave arms.** A B B A in one block, warm-up dropped, medians. These laptops throttle within ~30 s of all-core load, so sequential runs are not comparable.
3. **One change per arm.** An arm that changes two things proves neither.
4. **3% or it is noise** for a decision. Smaller lossless wins are still welcome; say so and they go into the combined "small-gains" arm.
5. **Production shapes.** Test at real context sizes (16K/32K), real head counts and real thread counts. Power-of-two aliasing and thread scaling only show there.
6. **Decode after prefill.** A prefill-side change must also time the decode right after it.
7. **Report the conditions:** device, SoC SKU, RAM, kernel, engine commit, model SHA-256, threads and placement, context, cache size (Flash), temperature, power source (AC/battery).
8. **Profiles are not evidence.** A profile ranks candidates; an A/B decides.
9. **Keep failures.** A clean negative result with its numbers is a contribution; add it to the "what did not work" table.

## Privacy

Share only what you meant to make public. Use `python3 -I tools/export_receipt.py export …` for receipts: it keeps numbers, hashes, argv and hardware and refuses to write anything that looks like a home path, user or host name, serial number, MAC address, e-mail or key. Never attach private prompts, model answers, generated images, chat transcripts, credentials or device serials. No model weights or vendor (Qualcomm) binaries in the repository.

## Code

- Lab: `./tools/run_lab_tests.sh` must pass. It gives the suite a temporary HOME and XDG directories; do not point tests at an installed runtime or its inference lease. To select tests, pass unittest arguments, for example `./tools/run_lab_tests.sh discover -s tests -p test_000_isolation.py`.
- Engine: `cargo test --release` for the crates you touch; Flash changes also `flash-m3 selftest`. Build failures must fail the script (`set -o pipefail`).
- Keep changes small and behind a flag when they change numerics or scheduling; default off until measured.
- Tests and helpers get a timeout. Anything that may use more than ~1 GiB runs under a memory limit.
- Generated or AI-assisted code is fine here; you are responsible for understanding and testing it. Upstream projects have their own rules (llama.cpp requires disclosure and human-written PR text); follow them when you send a patch upstream.

## Licence

Contributions are accepted under the MIT licence of this repository. A patch offered to an upstream project is contributed under that project's licence by its author. Keep upstream copyright and licence notices intact, and record the source revision of anything vendored or patched in `patches/manifest.json` or the engine's `PROVENANCE.md`.
