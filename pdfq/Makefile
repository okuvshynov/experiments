# Makefile for pdfq - Simple PDF Viewer for macOS

SWIFT = swiftc
SWIFT_FLAGS = -O -framework Cocoa -framework PDFKit
SOURCE = pdfq.swift
BINARY = pdfq
INSTALL_DIR = /usr/local/bin

.PHONY: all build clean install uninstall

all: build

build:
	@echo "Building pdfq..."
	$(SWIFT) $(SWIFT_FLAGS) $(SOURCE) -o $(BINARY)
	@echo "Build complete! Binary: ./$(BINARY)"

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BINARY)
	@echo "Clean complete."

install: build
	@echo "Installing pdfq to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	cp $(BINARY) $(INSTALL_DIR)/$(BINARY)
	@echo "Installation complete! You can now run: pdfq <path/to/file.pdf>"

uninstall:
	@echo "Uninstalling pdfq from $(INSTALL_DIR)..."
	rm -f $(INSTALL_DIR)/$(BINARY)
	@echo "Uninstall complete."

help:
	@echo "pdfq - Simple PDF Viewer for macOS"
	@echo ""
	@echo "Targets:"
	@echo "  build     - Build the pdfq binary (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Build and install to $(INSTALL_DIR)"
	@echo "  uninstall - Remove from $(INSTALL_DIR)"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make build"
	@echo "  ./pdfq <path/to/file.pdf>"
