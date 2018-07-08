CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -g -fPIC
LFLAGS = -shared -fPIC
OBJS = vudu.o
TARGET = vudurt.so

all: $(TARGET) test

$(TARGET): $(OBJS)
	$(CC) $(LFLAGS) -o $@ $^

vudu.o: vudu.c vudu.h
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm $(TARGET) $(OBJS) test

test: test.c $(TARGET)
	$(CC) -o $@ test.c $(TARGET)

run:
	LD_LIBRARY_PATH=. ./test
