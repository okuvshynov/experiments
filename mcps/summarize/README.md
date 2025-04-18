# Local LLM Summarization MCP Server

## Problem

When working with large projects in Claude Code, Claude Desktop app, or other coding agents that use proprietary models:

- Summarizing an entire project with Claude models can quickly hit usage limits
- Working without summaries requires extensive exploration, often hitting limits anyway

## Solution

This project creates a local LLM-based summarization tool that:

1. Uses a local LLM to summarize files and directories
2. Implements this as an MCP server tool for Claude Code
3. Allows Claude to request summaries by path rather than processing entire file contents

Benefits:
- Claude doesn't need to read the entire project (potentially millions of tokens)
- Summaries are generated cheaper with local models
- Powerful models like Claude Sonnet 3.7 can still be used to identify what to summarize and generate final documentation

## Setup Instructions

### 1. Install Claude Code
Follow the [official documentation](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)

### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Run OpenAI-compatible Local LLM Server
This project was tested with [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) using various 3B-7B-13B models:

```bash
./llama.cpp/build/bin/llama-server -m path/to/your/local/model.gguf -n 8192 -c 65536 --host 0.0.0.0
```

### 4. Clone This Repository
```bash
git clone https://github.com/okuvshynov/experiments.git
```

### 5. Configure Claude Code to Use the MCP Server
```bash
claude mcp add summarize -s user -- uv "--directory" /Absolute/path/to/experiments/mcps/summarize run summarize.py
```
Note: The `-s` flag adds this globally rather than per project

### 6. Define Custom Claude Command for Your Project

Example with [cubestat](https://github.com/okuvshynov/cubestat):

```bash
git clone https://github.com/okuvshynov/cubestat.git
mkdir -p cubestat/.claude/commands && cp experiments/mcps/commands/initref.md cubestat/.claude/commands
```

## Usage

Once set up, you can use the `initref` command from within Claude Code to:
1. Pass files for summarization to your local LLM
2. Use the summaries to generate documentation, saving on tokens/limits

Example result: [cubestat documentation commit](https://github.com/okuvshynov/cubestat/commit/cd482d964ccbc2449d56e99a87f0172df412d489)

