// SPDX-License-Identifier: MIT
// Streaming read ceiling: allocation/fill/thread startup are outside the timer.
#define _GNU_SOURCE
#include <arm_neon.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pthread_barrier_t ready, start, done;
static int abort_read;

struct worker {
    const uint64_t *data;
    size_t begin, end;
    uint64_t checksum;
    unsigned passes;
    int cpu, bind_error;
} __attribute__((aligned(64)));

static void die(const char *message) {
    fprintf(stderr, "memory_read: %s\n", message);
    exit(1);
}

static unsigned number(const char *text, unsigned maximum) {
    if (!*text || *text == '-' || *text == '+') die("invalid unsigned argument");
    errno = 0;
    char *end;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || *end || !value || value > maximum) die("argument outside range");
    return (unsigned)value;
}

static int cpu_list(const char *text, int *cpus) {
    cpu_set_t seen;
    CPU_ZERO(&seen);
    int count = 0;
    const char *at = text;
    while (*at) {
        if (count == 12 || *at < '0' || *at > '9') die("invalid CPU list");
        errno = 0;
        char *end;
        unsigned long cpu = strtoul(at, &end, 10);
        if (errno || cpu >= CPU_SETSIZE || (*end && *end != ',') || CPU_ISSET(cpu, &seen))
            die("invalid or duplicate CPU");
        CPU_SET(cpu, &seen);
        cpus[count++] = (int)cpu;
        if (!*end) break;
        at = end + 1;
        if (!*at) die("trailing CPU-list comma");
    }
    return count;
}

static uint64_t expected_prefix(size_t words) {
    uint64_t remainder = words % 256;
    return (uint64_t)(words / 256) * 32640 + remainder * (remainder - (remainder != 0)) / 2;
}

static uint64_t read_chunk(const uint64_t *data, size_t words) {
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    uint64x2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    size_t at = 0;
    for (; words - at >= 16; at += 16) {
        a0 = vaddq_u64(a0, vld1q_u64(data + at));
        a1 = vaddq_u64(a1, vld1q_u64(data + at + 2));
        a2 = vaddq_u64(a2, vld1q_u64(data + at + 4));
        a3 = vaddq_u64(a3, vld1q_u64(data + at + 6));
        a4 = vaddq_u64(a4, vld1q_u64(data + at + 8));
        a5 = vaddq_u64(a5, vld1q_u64(data + at + 10));
        a6 = vaddq_u64(a6, vld1q_u64(data + at + 12));
        a7 = vaddq_u64(a7, vld1q_u64(data + at + 14));
    }
    uint64x2_t left = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
    uint64x2_t right = vaddq_u64(vaddq_u64(a4, a5), vaddq_u64(a6, a7));
    uint64_t sum = vaddvq_u64(vaddq_u64(left, right));
    for (; at < words; ++at) sum += data[at];
    return sum;
}

static void barrier(pthread_barrier_t *value) {
    int result = pthread_barrier_wait(value);
    if (result && result != PTHREAD_BARRIER_SERIAL_THREAD) die("barrier wait failed");
}

static void *run(void *argument) {
    struct worker *worker = argument;
    cpu_set_t set, observed;
    CPU_ZERO(&set);
    CPU_SET(worker->cpu, &set);
    worker->bind_error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (!worker->bind_error) {
        worker->bind_error = pthread_getaffinity_np(pthread_self(), sizeof(observed), &observed);
        if (!worker->bind_error && (CPU_COUNT(&observed) != 1 || !CPU_ISSET(worker->cpu, &observed)))
            worker->bind_error = EINVAL;
    }
    barrier(&ready);
    barrier(&start);
    if (!abort_read) {
        uint64_t total = 0;
        for (unsigned pass = 0; pass < worker->passes; ++pass) {
            // Forces fresh memory loads each pass; the checksum is consumed below.
            __asm__ __volatile__("" ::: "memory");
            total += read_chunk(worker->data + worker->begin, worker->end - worker->begin);
        }
        worker->checksum = total;
    }
    barrier(&done);
    return NULL;
}

int main(int argc, char **argv) {
    unsigned threads = 0, mib = 2048, passes = 32;
    const char *list = NULL;
    for (int at = 1; at < argc; at += 2) {
        if (at + 1 == argc) die("usage: --threads 2|4|6|8|12 --cpus ORDERED_LIST [--mib 1024..2048] [--passes 1..256]");
        if (!strcmp(argv[at], "--threads")) threads = number(argv[at + 1], 12);
        else if (!strcmp(argv[at], "--mib")) mib = number(argv[at + 1], 2048);
        else if (!strcmp(argv[at], "--passes")) passes = number(argv[at + 1], 256);
        else if (!strcmp(argv[at], "--cpus")) list = argv[at + 1];
        else die("unknown option");
    }
    if (!(threads == 2 || threads == 4 || threads == 6 || threads == 8 || threads == 12) || !list || mib < 1024)
        die("explicit thread/CPU placement and 1..2 GiB array required");
    int cpus[12];
    if (cpu_list(list, cpus) != (int)threads) die("CPU count must equal thread count");
    size_t bytes = (size_t)mib * 1024 * 1024, words = bytes / sizeof(uint64_t);
    uint64_t *data;
    if (posix_memalign((void **)&data, 64, bytes)) die("allocation failed");
    for (size_t at = 0; at < words; ++at) data[at] = at & 255;
    struct worker workers[12] = {0};
    pthread_t ids[12];
    if (pthread_barrier_init(&ready, NULL, threads + 1) ||
        pthread_barrier_init(&start, NULL, threads + 1) ||
        pthread_barrier_init(&done, NULL, threads + 1)) die("barrier initialization failed");
    for (unsigned at = 0; at < threads; ++at) {
        workers[at] = (struct worker){.data = data, .begin = words * at / threads,
            .end = words * (at + 1) / threads, .passes = passes, .cpu = cpus[at]};
        if (pthread_create(&ids[at], NULL, run, &workers[at])) die("pthread_create failed");
    }
    barrier(&ready);
    for (unsigned at = 0; at < threads; ++at) {
        if (workers[at].bind_error) {
            fprintf(stderr, "memory_read: CPU %d affinity failed: %s\n", cpus[at], strerror(workers[at].bind_error));
            abort_read = 1;
        }
    }
    struct timespec before, after;
    if (clock_gettime(CLOCK_MONOTONIC, &before)) die("clock_gettime failed");
    barrier(&start);
    barrier(&done);
    if (clock_gettime(CLOCK_MONOTONIC, &after)) die("clock_gettime failed");
    uint64_t checksum = 0;
    for (unsigned at = 0; at < threads; ++at) {
        if (pthread_join(ids[at], NULL)) die("pthread_join failed");
        uint64_t expected = (expected_prefix(workers[at].end) - expected_prefix(workers[at].begin)) * passes;
        if (!abort_read && workers[at].checksum != expected) die("per-thread checksum mismatch");
        checksum += workers[at].checksum;
    }
    uint64_t expected = expected_prefix(words) * passes;
    double seconds = (after.tv_sec - before.tv_sec) + (after.tv_nsec - before.tv_nsec) * 1e-9;
    free(data);
    pthread_barrier_destroy(&ready);
    pthread_barrier_destroy(&start);
    pthread_barrier_destroy(&done);
    if (abort_read) return 1;
    if (checksum != expected || seconds <= 0) die("checksum or timer failed");
    uint64_t read_bytes = (uint64_t)bytes * passes;
    printf("{\"ok\":true,\"benchmark\":\"neon_memory_read\",\"array_bytes\":%zu,\"threads\":%u,\"passes\":%u,"
           "\"bytes\":%" PRIu64 ",\"seconds\":%.9f,\"gb_s\":%.6f,\"checksum\":%" PRIu64 ","
           "\"expected_checksum\":%" PRIu64 ",\"cpus\":[", bytes, threads, passes, read_bytes, seconds,
           read_bytes / seconds / 1e9, checksum, expected);
    for (unsigned at = 0; at < threads; ++at) printf("%s%d", at ? "," : "", cpus[at]);
    puts("]}");
    return 0;
}
