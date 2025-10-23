# pdfq - Simple PDF Viewer for macOS

A lightweight PDF viewer for macOS that logs text selections and annotations to stdout.

## Features

- View PDF documents using Apple's PDFKit framework
- Logs text selections to stdout in real-time
- Logs annotation creation events
- Simple command-line interface
- Continuous scrolling mode

## Requirements

- macOS (tested on modern versions)
- Swift compiler (comes with Xcode Command Line Tools)

## Building

```bash
make build
```

This will compile the Swift source and create a `pdfq` binary in the current directory.

## Installation

To install system-wide:

```bash
make install
```

This installs the binary to `/usr/local/bin/pdfq`.

## Usage

```bash
./pdfq path/to/your.pdf
```

Or if installed:

```bash
pdfq path/to/your.pdf
```

### Output Format

The application logs the following events to stdout:

**Text Selection:**
```
TEXT_SELECTION: <selected text>
```

**Annotation Creation:**
```
ANNOTATION_CREATED: Type=<type>, Contents="<contents>", Page=<page>, Bounds=<bounds>
```

## Examples

```bash
# View a PDF
./pdfq ~/Documents/sample.pdf

# View and capture logs to a file
./pdfq ~/Documents/sample.pdf > events.log
```

## Uninstalling

```bash
make uninstall
```

## Cleaning Build Artifacts

```bash
make clean
```

## Adding Annotations

The viewer includes several ways to add annotations:

### Keyboard Shortcuts
- **Cmd+T** - Add Text Annotation (yellow text box)
- **Cmd+N** - Add Note Annotation (sticky note icon)
- **Cmd+H** - Add Highlight (select text first, then press Cmd+H)
- **Cmd+R** - Add Circle Annotation (red circle shape)

### Menu Bar
Use the "Annotations" menu to access all annotation tools.

### Tips
- For highlights: First select text with your mouse, then press Cmd+H
- Text annotations can be edited by double-clicking them
- All annotations are logged to stdout immediately when created

## How It Works

The application:
1. Loads the specified PDF file using PDFKit
2. Creates a native macOS window with a PDF viewer
3. Sets up a menu bar with annotation tools and keyboard shortcuts
4. Registers notification observers for:
   - Text selection changes (`PDFViewSelectionChanged`)
   - Annotation modifications
5. Logs events to stdout as they occur

## License

This is a simple example application for demonstration purposes.
