# BioProxy - LLM Message Injection Proxy

A lightweight, universal proxy for OpenAI-compatible LLM servers that allows dynamic message template injection based on message prefixes.

## Features

- ✅ **Universal compatibility** - Works with any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, Ollama, etc.)
- ✅ **Dynamic template loading** - Templates reload from file on every request (no restart needed)
- ✅ **Prefix-based routing** - Map prefixes like `/context` to different template files
- ✅ **Zero dependencies** - Uses only Go standard library
- ✅ **Lightweight** - Single binary, ~2MB compiled
- ✅ **Transparent** - Proxies all other requests unchanged

## How It Works

```
Browser/UI → BioProxy (port 8081) → LLM Server (port 8080)
                    ↓
              Template Injection
              (on /v1/chat/completions)
```

When you send a message starting with a configured prefix (e.g., `/context hello`):
1. Proxy detects the prefix
2. Reads the corresponding template file (fresh on each request)
3. Replaces `__the__user__message__` placeholder with your actual message
4. Sends modified request to LLM server

## Quick Start

### 1. Build

```bash
go build -o bioproxy
```

### 2. Configure

Edit `config.json` to map prefixes to template files:

```json
{
  "prefixes": {
    "/context ": "context.txt",
    "/mini ": "mini.txt"
  }
}
```

### 3. Create Templates

Create template files with `__the__user__message__` placeholder:

**context.txt:**
```
Here's useful context for the conversation:

[Your large context here]

User's question: __the__user__message__
```

**mini.txt:**
```
Answer concisely: __the__user__message__
```

### 4. Run

```bash
./bioproxy -backend http://localhost:8080 -port 8081
```

### 5. Use

Point your LLM client/UI to `http://localhost:8081` and send messages with prefixes:

- `/context What is the capital of France?` → Uses context.txt template
- `/mini What is 2+2?` → Uses mini.txt template
- `Regular message` → Passes through unchanged

## Configuration

### Command-Line Flags

- `-backend` - Backend LLM server URL (default: `http://localhost:8080`)
- `-port` - Proxy server port (default: `8081`)
- `-config` - Config file path (default: `config.json`)
- `-placeholder` - Template placeholder (default: `__the__user__message__`)

### Example

```bash
./bioproxy \
  -backend http://localhost:11434 \
  -port 9000 \
  -config my-config.json \
  -placeholder "{USER_MESSAGE}"
```

## Template Files

Templates are plain text files with a placeholder for the user's message. They reload on every request, so you can edit them without restarting the proxy.

**Example template with context:**
```
System: You are a helpful coding assistant with access to the following codebase documentation:

[Large documentation here...]

User query: __the__user__message__

Please answer based on the documentation above.
```

## Compatible LLM Servers

Works with any server implementing OpenAI's `/v1/chat/completions` endpoint:

- ✅ llama.cpp server
- ✅ vLLM
- ✅ LM Studio
- ✅ Ollama (use `/v1/chat/completions` endpoint)
- ✅ Text Generation Inference (TGI)
- ✅ LocalAI
- ✅ Koboldcpp
- ✅ OpenAI API (for testing)

## Use Cases

- **Dynamic context injection** - Load different contexts without changing system prompts
- **Prompt management** - Switch between different prompting strategies with prefixes
- **A/B testing** - Test different prompt templates easily
- **Multi-domain assistants** - Use different personas/contexts per prefix
- **Development workflow** - Separate "detailed" vs "concise" modes

## Architecture

```
┌─────────────┐
│  Web UI     │
│  (Browser)  │
└──────┬──────┘
       │ http://localhost:8081
       ↓
┌─────────────┐
│  BioProxy   │──→ Reads template files
│  (Go)       │
└──────┬──────┘
       │ Modified request
       ↓
┌─────────────┐
│ LLM Server  │
│ (llama.cpp) │
└─────────────┘
```

## Development

The proxy is intentionally simple and hackable:

- `main.go` - ~200 lines, well-commented
- Standard library only
- Easy to extend (add logging, metrics, auth, etc.)

## Advanced Usage

### Custom Placeholder

Use a different placeholder pattern:

```bash
./bioproxy -placeholder "{{MESSAGE}}"
```

Then in templates:
```
Your context here... {{MESSAGE}}
```

### Multiple Backends

Run multiple proxy instances for different backends:

```bash
# Proxy for llama.cpp
./bioproxy -backend http://localhost:8080 -port 8081

# Proxy for vLLM
./bioproxy -backend http://localhost:8000 -port 8082 -config vllm-config.json
```

### Dynamic Context Files

Since templates reload on every request, you can use external tools to update them:

```bash
# Update context file from external source
./fetch-latest-docs.sh > context.txt

# Next request automatically uses new content
```

## License

MIT

## Contributing

This is a simple learning project. Feel free to fork and customize for your needs!
