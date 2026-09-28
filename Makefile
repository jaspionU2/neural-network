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

setup:
	python -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

clean:
	rm -rf $(BUILD_DIR)

clean-python:
	rm	-rf	__pycache__	.pytest_cache
	rm	-rf	.venv

.PHONY: all clean