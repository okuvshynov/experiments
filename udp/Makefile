# UDP Latency Benchmarks Makefile
#
# Usage:
#   make              - Build all targets
#   make clean        - Remove build directory
#   make client       - Build only clients
#   make server       - Build only servers
#   make aggr         - Build only link aggregation versions

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

BUILD_DIR = build
SRC_DIR = .

# Source files
CLIENT_SRCS = udpclient.c udpclient_opt.c udpclient_aggr.c
SERVER_SRCS = udpserver.c udpserver_aggr.c
LINUX_SRCS = client_opt_linux.c server_linux.c

# Targets
CLIENT_BINS = $(addprefix $(BUILD_DIR)/, $(CLIENT_SRCS:.c=))
SERVER_BINS = $(addprefix $(BUILD_DIR)/, $(SERVER_SRCS:.c=))
LINUX_BINS = $(addprefix $(BUILD_DIR)/, $(LINUX_SRCS:.c=))

# Platform detection
UNAME_S := $(shell uname -s)

# All targets (exclude Linux-specific on macOS)
ifeq ($(UNAME_S),Linux)
    ALL_TARGETS = $(CLIENT_BINS) $(SERVER_BINS) $(LINUX_BINS)
else
    ALL_TARGETS = $(CLIENT_BINS) $(SERVER_BINS)
endif

.PHONY: all clean client server aggr linux help

all: $(ALL_TARGETS)
	@echo "Build complete. Binaries in $(BUILD_DIR)/"

client: $(CLIENT_BINS)

server: $(SERVER_BINS)

aggr: $(BUILD_DIR)/udpclient_aggr $(BUILD_DIR)/udpserver_aggr

linux: $(LINUX_BINS)

# Generic build rule
$(BUILD_DIR)/%: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling $< -> $@"
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

help:
	@echo "UDP Latency Benchmarks"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all binaries (default)"
	@echo "  client  - Build client binaries only"
	@echo "  server  - Build server binaries only"
	@echo "  aggr    - Build link aggregation versions only"
	@echo "  linux   - Build Linux-specific versions only"
	@echo "  clean   - Remove build directory"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Binaries will be placed in: $(BUILD_DIR)/"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make clean && make"
	@echo "  make aggr"
