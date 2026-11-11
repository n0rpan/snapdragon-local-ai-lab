#include <stdlib.h>
#include <string.h>
#include "ring.h"

void ring_init(ring *r, size_t cap) {
	r->items = calloc(cap, sizeof *r->items);
	r->lens = calloc(cap, sizeof *r->lens);
	r->cap = cap;
	r->count = 0;
	r->head = 0;
}

void ring_push(ring *r, const char *line, size_t len) {
	if (r->cap == 0)
		return;
	char *copy = malloc(len + 1);
	memcpy(copy, line, len);
	copy[len] = '\0';
	free(r->items[r->head]); /* overwrites the oldest line once full */
	r->items[r->head] = copy;
	r->lens[r->head] = len;
	r->head = (r->head + 1) % r->cap;
	if (r->count < r->cap)
		r->count++;
}

void ring_iter(const ring *r, ring_fn fn, void *ctx) {
	for (size_t i = 0; i < r->count; i++)
		fn(r->items[i], r->lens[i], ctx);
}

void ring_free(ring *r) {
	for (size_t i = 0; i < r->cap; i++)
		free(r->items[i]);
	free(r->items);
	free(r->lens);
	r->items = NULL;
	r->lens = NULL;
	r->cap = 0;
	r->count = 0;
	r->head = 0;
}
