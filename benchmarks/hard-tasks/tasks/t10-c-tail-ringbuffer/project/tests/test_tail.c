#include <stdio.h>
#include <string.h>
#include "ring.h"
#include "strbuf.h"

static int failures;

#define CHECK(cond) do { \
	if (!(cond)) { \
		fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); \
		failures++; \
	} \
} while (0)

static char got[1024];

static void collect(const char *line, size_t len, void *ctx) {
	(void)ctx;
	strncat(got, line, len);
	strcat(got, "|");
}

static const char *ring_dump(ring *r) {
	got[0] = '\0';
	ring_iter(r, collect, NULL);
	return got;
}

int main(void) {
	ring r;

	ring_init(&r, 3);
	ring_push(&r, "a", 1);
	ring_push(&r, "b", 1);
	CHECK(strcmp(ring_dump(&r), "a|b|") == 0);
	ring_push(&r, "c", 1);
	ring_push(&r, "d", 1);
	ring_push(&r, "e", 1);
	CHECK(strcmp(ring_dump(&r), "c|d|e|") == 0);
	ring_free(&r);

	ring_init(&r, 0);
	ring_push(&r, "x", 1);
	CHECK(strcmp(ring_dump(&r), "") == 0);
	ring_free(&r);

	ring_init(&r, 2);
	ring_push(&r, "", 0);
	ring_push(&r, "xyz", 2);
	CHECK(strcmp(ring_dump(&r), "|xy|") == 0);
	ring_free(&r);

	strbuf sb;
	sb_init(&sb);
	sb_append(&sb, "abc", 3);
	CHECK(sb.len == 3 && strcmp(sb_cstr(&sb), "abc") == 0);
	sb.len = 0;
	for (int i = 0; i < 16; i++)
		sb_append(&sb, "x", 1);
	CHECK(strlen(sb_cstr(&sb)) == 16);
	sb_free(&sb);

	if (failures) {
		fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	puts("test_tail: all checks passed");
	return 0;
}
