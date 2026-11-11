#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ring.h"
#include "strbuf.h"

static int bad_count(void) {
	fputs("tail: invalid line count\n", stderr);
	return 2;
}

/* Parses a non-negative decimal integer. Returns 0 on success. */
static int parse_count(const char *s, size_t *out) {
	char *end;
	errno = 0;
	long v = strtol(s, &end, 10);
	if (end == s || *end != '\0' || v < 0 || errno == ERANGE)
		return -1;
	*out = (size_t)v;
	return 0;
}

static void emit(const char *line, size_t len, void *ctx) {
	(void)ctx;
	fwrite(line, 1, len, stdout);
	putchar('\n');
}

int main(int argc, char **argv) {
	size_t n = 10;
	if (argc == 3 && strcmp(argv[1], "-n") == 0) {
		if (parse_count(argv[2], &n) != 0)
			return bad_count();
	} else if (argc != 1) {
		return bad_count();
	}

	ring r;
	strbuf line;
	ring_init(&r, n);
	sb_init(&line);

	int c;
	while ((c = getchar()) != EOF) {
		if (c == '\n') {
			ring_push(&r, sb_cstr(&line), line.len);
			line.len = 0;
		} else {
			char ch = (char)c;
			sb_append(&line, &ch, 1);
		}
	}
	if (line.len > 0) /* final line without a trailing newline */
		ring_push(&r, sb_cstr(&line), line.len);

	ring_iter(&r, emit, NULL);
	ring_free(&r);
	sb_free(&line);
	return 0;
}
