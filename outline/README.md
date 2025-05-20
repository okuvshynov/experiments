# Outline

A language-agnostic code outline generator that extracts classes, methods, functions, and top-level variables from source code.

## Features

- Supports multiple programming languages (JavaScript, TypeScript, Python, Ruby, Go, Java, C, C++)
- Language auto-detection based on file extension or content
- Reads from files or stdin
- Text or JSON output formats

## Installation

```bash
# Install dependencies
npm install

# Make the CLI executable
chmod +x bin/outline.js

# Optionally, install globally
npm install -g .
```

## Usage

```bash
# Process a file
./bin/outline.js path/to/source/file.js

# Process stdin
cat path/to/source/file.js | ./bin/outline.js

# Specify language explicitly
./bin/outline.js --language python path/to/source/file.py

# Output as JSON
./bin/outline.js --format json path/to/source/file.js
```

## Supported Languages

- JavaScript
- TypeScript
- Python
- Ruby
- Go
- Java
- C
- C++

## How It Works

Outline uses Tree-sitter to parse source code into an abstract syntax tree (AST), then traverses the tree to extract structural elements like classes, methods, and functions.