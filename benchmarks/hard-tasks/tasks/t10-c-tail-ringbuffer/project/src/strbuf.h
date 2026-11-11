#ifndef STRBUF_H
#define STRBUF_H
#include <stddef.h>

/* Growable byte buffer. sb_cstr() returns a NUL-terminated view of the bytes. */
typedef struct {
	char *data;
	size_t len;
	size_t cap;
} strbuf;

void sb_init(strbuf *sb);
void sb_append(strbuf *sb, const char *s, size_t n);
const char *sb_cstr(strbuf *sb);
void sb_free(strbuf *sb);

#endif
