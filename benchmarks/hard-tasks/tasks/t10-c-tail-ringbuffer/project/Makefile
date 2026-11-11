CC = gcc
CFLAGS = -std=c11 -Wall -O1 -g
LIB = src/strbuf.c src/ring.c

tail: src/tail.c $(LIB) src/strbuf.h src/ring.h
	$(CC) $(CFLAGS) src/tail.c $(LIB) -o tail

tests/test_tail: tests/test_tail.c $(LIB) src/strbuf.h src/ring.h
	$(CC) $(CFLAGS) -Isrc tests/test_tail.c $(LIB) -o tests/test_tail

test: tail tests/test_tail
	./tests/test_tail
	sh tests/run.sh

clean:
	rm -f tail tests/test_tail

.PHONY: test clean
