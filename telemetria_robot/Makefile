CC=gcc
CFLAGS=-pthread -Wall -Wextra

all: server

server: server.c
	$(CC) $(CFLAGS) -o server server.c

clean:
	rm -f server
