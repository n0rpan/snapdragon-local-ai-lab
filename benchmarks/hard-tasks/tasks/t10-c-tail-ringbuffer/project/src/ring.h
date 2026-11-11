#ifndef RING_H
#define RING_H
#include <stddef.h>

/* Keeps the last `cap` lines (copies). cap 0 keeps nothing. */
typedef struct {
	char **items;
	size_t *lens;
	size_t cap;
	size_t count; /* lines stored, <= cap */
	size_t head;  /* slot the next push writes to */
} ring;

typedef void (*ring_fn)(const char *line, size_t len, void *ctx);

void ring_init(ring *r, size_t cap);
void ring_push(ring *r, const char *line, size_t len);
void ring_iter(const ring *r, ring_fn fn, void *ctx); /* oldest -> newest */
void ring_free(ring *r);

#endif
