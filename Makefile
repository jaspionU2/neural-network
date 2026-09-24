PROJ_NAME=unity_build
BUILD_DIR=build
SRC=src/unity_build.c
TARGET=$(BUILD_DIR)/$(PROJ_NAME)

CC=gcc
CFLAGS=-Wall -Wextra -Werror -g -fdiagnostics-color=always -I./include
LDFLAGS=-lm

ALL_DEPS := $(shell find src include -type f \( -name '*.c' -o -name '*.h' \) | sort)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) $(ALL_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean