.PHONY: all clean help macos linux-arm64 linux-amd64 linux-arm32

# Build directory
BUILD_DIR := bin

# Default target
all: macos linux-arm64

# Build for macOS
macos:
	@echo "Building for macOS ARM64..."
	@mkdir -p $(BUILD_DIR)
	GOOS=darwin GOARCH=arm64 go build -o $(BUILD_DIR)/bioproxy-darwin-arm64

# Build for Linux ARM64 (Raspberry Pi 3/4/5, Pi Zero 2 W)
linux-arm64:
	@echo "Building for Linux ARM64 (Raspberry Pi)..."
	@mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=arm64 go build -o $(BUILD_DIR)/bioproxy-linux-arm64

# Build for Linux AMD64 (standard Linux servers)
linux-amd64:
	@echo "Building for Linux AMD64..."
	@mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=amd64 go build -o $(BUILD_DIR)/bioproxy-linux-amd64

# Build for Linux ARM 32-bit (older Raspberry Pi models, Pi Zero)
linux-arm32:
	@echo "Building for Linux ARM 32-bit..."
	@mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=arm GOARM=7 go build -o $(BUILD_DIR)/bioproxy-linux-arm32

# Build all platforms
all-platforms: macos linux-arm64 linux-amd64 linux-arm32
	@echo "Built for all platforms:"
	@ls -lh $(BUILD_DIR)/bioproxy-*

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# Show help
help:
	@echo "Available targets:"
	@echo "  make              - Build for macOS and Linux ARM64 (default)"
	@echo "  make macos        - Build for macOS ARM64"
	@echo "  make linux-arm64  - Build for Raspberry Pi (ARM64)"
	@echo "  make linux-amd64  - Build for Linux x86_64"
	@echo "  make linux-arm32  - Build for older Raspberry Pi (ARM 32-bit)"
	@echo "  make all-platforms - Build for all platforms"
	@echo "  make clean        - Remove build directory"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Binaries are built in: $(BUILD_DIR)/"
