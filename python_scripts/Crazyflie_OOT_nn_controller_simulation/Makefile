CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -I. -DNN_DUMP_ONCE
LDFLAGS ?=
LIBS ?= -lm
PYTHON ?= python3
RM ?= rm -f

BIN := nn_controller_simulation
CSVFILES := $(wildcard *_log.csv)
CFILES := $(wildcard *.c)
OBJS := $(CFILES:.c=.o)

.PHONY: all build run plots clean

all: run plots

build: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(BIN)
	./$(BIN)

plots: pwm_plots.py
	$(PYTHON) pwm_plots.py

clean:
	$(RM) $(OBJS) $(BIN) $(CSVFILES)