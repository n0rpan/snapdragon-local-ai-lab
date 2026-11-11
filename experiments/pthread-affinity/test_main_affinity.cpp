// Regression: a reusable, unpaused pool must restore its caller's placement.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <vector>

static void pin(int cpu) {
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
}
static void check(int cpu) {
    cpu_set_t mask; assert(sched_getaffinity(0, sizeof(mask), &mask) == 0);
    if (CPU_COUNT(&mask) != 1 || !CPU_ISSET(cpu, &mask)) {
        std::fprintf(stderr, "caller affinity not restored: expected CPU %d\n", cpu);
        std::abort();
    }
}
static ggml_threadpool * pool(int n) {
    auto params = ggml_threadpool_params_default(n);
    params.poll = 0; params.strict_cpu = true; params.paused = false;
    for (int i = 0; i < n; ++i) params.cpumask[i] = true;
    return ggml_threadpool_new(&params);
}
int main() {
    cpu_set_t original; assert(sched_getaffinity(0, sizeof(original), &original) == 0);
    assert(CPU_ISSET(0, &original) && CPU_ISSET(1, &original) && CPU_ISSET(2, &original));
    auto * ctx = ggml_init({16 * 1024 * 1024, nullptr, false}); assert(ctx);
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 4);
    ggml_set_f32(a, 1.0f); ggml_set_f32(b, 1.0f);
    auto * c = ggml_mul_mat(ctx, a, b);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, c);
    auto * batch = pool(3); auto * decode = pool(2);
    for (int i = 0; i < 3; ++i) {
        // Neither pool is paused. Caller starts on the OTHER pool's main CPU.
        for (int arm = 0; arm < 2; ++arm) {
            const int n = arm ? 2 : 3; pin(arm ? 2 : 1);
            auto plan = ggml_graph_plan(graph, n, arm ? decode : batch);
            std::vector<unsigned char> scratch(plan.work_size);
            plan.work_data = scratch.data();
            assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            check(n - 1);
            assert(ggml_get_f32_1d(c, 0) == 512.0f);
        }
    }
    ggml_threadpool_free(decode); ggml_threadpool_free(batch); ggml_free(ctx);
    assert(sched_setaffinity(0, sizeof(original), &original) == 0);
    std::puts("ALL PASSED: active-pool caller affinity, alternating pools, exact output");
}
