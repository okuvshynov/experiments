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

**Comment Added:**
```
ANNOTATION_CREATED: Page=<page>, SelectedText="<text>", Comment="<your comment>"
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

## Adding Comments to Selected Text

The viewer allows you to add comments to any selected text:

1. **Select text** in the PDF with your mouse
2. **Press Cmd+K** (or use menu: Comment → Add Comment to Selection)
3. **Enter your comment** in the dialog that appears
4. The selected text will be highlighted and your comment will be attached

All comments are logged to stdout immediately when created, showing:
- The page number
- The selected text
- Your comment

## How It Works

The application:
1. Loads the specified PDF file using PDFKit
2. Creates a native macOS window with a PDF viewer
3. Sets up a menu bar with a simple comment tool (Cmd+K)
4. Registers notification observers for text selection changes
5. When you add a comment:
   - Creates a highlight annotation at the selected text position
   - Attaches your comment to that annotation
   - Logs the event to stdout with page, selected text, and comment
6. All events are logged to stdout in real-time

## License

This is a simple example application for demonstration purposes.
