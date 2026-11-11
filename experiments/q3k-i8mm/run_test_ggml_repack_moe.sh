#!/bin/sh
# Builds and runs test_ggml_repack_moe.cpp against a build of the patched tree that has ONLY the ggml-cpu target
# built (libggml-base.a + libggml-cpu.a, plus libkleidiai.a if the build has GGML_CPU_KLEIDIAI=ON). No models, no timing.
#   LLAMA_SRC     patched llama.cpp checkout (required)
#   LLAMA_BUILD   its build dir (default $LLAMA_SRC/build-claude)
#   TEST_THREADS  graph thread counts to run (default "1 4 8")
#   TEST_CPUS     optional taskset CPU list, e.g. 11 to keep every run on one core (results do not depend on it)
#
# Pass criteria: every run prints ALL PASSED, and every output hash is identical across GGML_CPU_MOE_GEMM=0/1/exact
# and across all thread counts (the MoE gemm must be bitwise identical to the gemv path for every tested type).
set -eu
LLAMA_SRC=${LLAMA_SRC:?set LLAMA_SRC to the patched llama.cpp checkout}
LLAMA_BUILD=${LLAMA_BUILD:-$LLAMA_SRC/build-claude}
TEST_THREADS=${TEST_THREADS:-"1 4 8"}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
KLEIDI=""
if [ -f "$LLAMA_BUILD/_deps/kleidiai-build/libkleidiai.a" ]; then KLEIDI="$LLAMA_BUILD/_deps/kleidiai-build/libkleidiai.a"; fi
nice -n 19 g++ -O2 -std=gnu++17 -I"$LLAMA_SRC/ggml/include" -o "$OUT/test_ggml_repack_moe" "$HERE/test_ggml_repack_moe.cpp" \
    "$LLAMA_BUILD/ggml/src/libggml-cpu.a" "$LLAMA_BUILD/ggml/src/libggml-base.a" $KLEIDI -fopenmp -lm -lpthread
RUN="nice -n 19"
if [ -n "${TEST_CPUS:-}" ]; then RUN="taskset -c $TEST_CPUS nice -n 19"; fi
fail=0
for t in $TEST_THREADS; do
    for m in 0 1 exact; do
        if ! Q3K_TEST_THREADS=$t GGML_CPU_MOE_GEMM=$m OMP_WAIT_POLICY=passive $RUN "$OUT/test_ggml_repack_moe" "$@" > "$OUT/out-$m-t$t.txt"; then
            echo "FAILED: GGML_CPU_MOE_GEMM=$m threads=$t"; fail=1
        fi
        # output lines without the header (mode, thread count)
        grep -v -e '^GGML_CPU_MOE_GEMM=' -e '^graph_threads=' "$OUT/out-$m-t$t.txt" > "$OUT/body-$m-t$t.txt"
    done
done
first_t=$(echo $TEST_THREADS | cut -d' ' -f1)
cat "$OUT/out-1-t$first_t.txt"
echo "---- output hashes must be identical across GGML_CPU_MOE_GEMM=0/1/exact and threads ($TEST_THREADS) ----"
for f in "$OUT"/body-*.txt; do
    if ! diff -q "$OUT/body-0-t$first_t.txt" "$f" > /dev/null; then
        echo "DIFFERS: $(basename "$f") vs body-0-t$first_t.txt"; diff "$OUT/body-0-t$first_t.txt" "$f" | head -20; fail=1
    fi
done
echo "---- GGML_CPU_REPACK_Q3K=0 ----"
GGML_CPU_REPACK_Q3K=0 $RUN "$OUT/test_ggml_repack_moe" Q3_K > "$OUT/q3k0.txt" || true
grep -E 'repacked=|mul_mat|PASSED|FAILED' "$OUT/q3k0.txt"
grep -q 'repacked=NO' "$OUT/q3k0.txt" || { echo "FAILED: GGML_CPU_REPACK_Q3K=0 still repacks Q3_K"; fail=1; }
rm -rf "$OUT"
if [ $fail -eq 0 ]; then echo "RUNNER PASSED"; else echo "RUNNER FAILED"; fi
exit $fail
