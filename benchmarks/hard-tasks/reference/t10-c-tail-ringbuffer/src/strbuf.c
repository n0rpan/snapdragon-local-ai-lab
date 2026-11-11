#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strbuf.h"

static void oom(void) {
	fputs("tail: out of memory\n", stderr);
	exit(1);
}

void sb_init(strbuf *sb) {
	sb->data = NULL;
	sb->len = 0;
	sb->cap = 0;
}

void sb_append(strbuf *sb, const char *s, size_t n) {
	if (n == 0)
		return;
	if (sb->len + n + 1 > sb->cap) { /* +1: always keep room for the NUL */
		size_t cap = sb->cap ? sb->cap : 16;
		while (cap < sb->len + n + 1)
			cap *= 2;
		char *p = realloc(sb->data, cap);
		if (!p)
			oom();
		sb->data = p;
		sb->cap = cap;
	}
	memcpy(sb->data + sb->len, s, n);
	sb->len += n;
}

const char *sb_cstr(strbuf *sb) {
	if (!sb->data)
		return "";
	sb->data[sb->len] = '\0';
	return sb->data;
}

void sb_free(strbuf *sb) {
	free(sb->data);
	sb_init(sb);
}
