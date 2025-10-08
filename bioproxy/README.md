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
make
```

This builds binaries for macOS and Linux ARM64 in the `bin/` directory.

For other platforms:
```bash
make all-platforms  # Build for all supported platforms
make linux-amd64    # Linux x86_64
make help           # Show all options
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

Create template files with `<{message}>` placeholder for user input:

**context.txt:**
```
Here's useful context for the conversation:

[Your large context here]

User's question: <{message}>
```

**mini.txt:**
```
Answer concisely: <{message}>
```

### 4. Run

```bash
# macOS
bin/bioproxy-darwin-arm64 -backend http://localhost:8080 -port 8081

# Linux (Raspberry Pi)
bin/bioproxy-linux-arm64 -backend http://192.168.1.X:8080 -port 8081
```

**Note:** If using `.local` hostnames on Linux, see troubleshooting section below.

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
- `-warmup-interval` - Interval between warmup checks (default: `30s`, set to `0` to disable)
- `-log-requests` - Log full requests to temp files for debugging (default: `false`)

### Example

```bash
./bioproxy \
  -backend http://localhost:11434 \
  -port 9000 \
  -config my-config.json
```

## Template Files

Templates use the unified `<{...}>` syntax for all dynamic content:
- `<{message}>` - Replaced with the user's message
- `<{filepath}>` - Replaced with content from the specified file

Templates reload on every request, so you can edit them without restarting the proxy.

**Example template with context:**
```
System: You are a helpful coding assistant with access to the following codebase documentation:

[Large documentation here...]

User query: <{message}>

Please answer based on the documentation above.
```

### Dynamic File Inclusion

Templates can include content from other files using `<{filepath}>` syntax. This is useful for injecting dynamic context that's updated by external processes.

**Example template with dynamic includes:**
```
Here's the latest context:

<{/tmp/context_dynamic.txt}>

User's question: <{message}>
```

**How it works:**
1. Template files are read on each request
2. All `<{...}>` patterns in the **original template** are processed:
   - `<{message}>` is replaced with the user's message
   - `<{filepath}>` is replaced with the content of that file (read fresh)
3. **Important**: Patterns are only detected in the original template, not in substituted content
   - This prevents recursive expansion
   - If a file contains `<{message}>`, it won't be replaced
   - If the user message contains `<{/tmp/file}>`, it won't be replaced

**Use cases:**
- Background job writes latest git commits to `/tmp/recent_commits.txt`
- Cron job exports database state to `/tmp/db_state.txt`
- File watcher updates `/tmp/codebase_index.txt` when code changes
- Template includes these files to provide fresh context

**Multiple includes:**
```
Project status: <{/tmp/status.txt}>
Recent commits: <{/tmp/commits.txt}>
Open issues: <{/tmp/issues.txt}>

Question: <{message}>
```

All file paths are read fresh on every request, so external processes can update them without restarting the proxy.

### Automatic Warmup

BioProxy can automatically send warmup requests when template files change, keeping the LLM's KV cache primed with fresh context.

**How it works:**
1. After you use a template (e.g., `/context`), the proxy tracks that template and all its included files
2. Every `-warmup-interval` (default 30s), the warmup loop:
   - Checks if any files have changed (using SHA256 hashes)
   - If changed AND no main request is active → sends warmup request with:
     - Template with all includes processed
     - `<{message}>` replaced with empty string
     - `n_predict=0` parameter (just process prompt, no generation)
3. Simple and predictable: only the warmup loop and main requests, never two warmups running

**Benefits:**
- First real query after context update is faster (prompt already in KV cache)
- Useful for frequently-updated dynamic contexts
- No manual intervention needed

**Example:**
```bash
# Enable with 30s interval (default)
bin/bioproxy-darwin-arm64 -backend http://localhost:8080 -port 8081

# Custom interval
bin/bioproxy-darwin-arm64 -warmup-interval 1m

# Disable warmup
bin/bioproxy-darwin-arm64 -warmup-interval 0
```

**Example:**
```
T=0:   User sends: /context What is Go?
       → Proxy tracks: context.txt + /tmp/dynamic.txt
       → Stores SHA256 hashes

T=5:   /tmp/dynamic.txt changes

T=30:  Warmup loop checks
       → Files changed: yes
       → Main request active: no
       → Sends warmup with latest content
       → Updates stored hashes

T=60:  Warmup loop checks
       → Files changed: no
       → Skips warmup

# If file changes during main request:
T=90:  User sends another question (takes 20s to process)
T=95:  /tmp/dynamic.txt changes
T=100: Warmup loop checks
       → Files changed: yes
       → Main request active: yes
       → Skips warmup (will catch it on next tick)
T=120: Warmup loop checks
       → Files changed: yes (from T=95)
       → Main request active: no
       → Sends warmup
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

### Request Debugging

Enable full request logging to see exactly what's being sent to the LLM after all template processing:

```bash
bin/bioproxy-darwin-arm64 -log-requests -backend http://localhost:8080 -port 8081
```

**Output:**
```
2025/10/05 12:34:56 Injected template context.txt for prefix /context
2025/10/05 12:34:56 Request logged to: /tmp/bioproxy-request-1234567890.json
```

**The temp file contains the full request with:**
- All template processing applied
- File includes resolved
- Pretty-printed JSON for easy reading

**Example logged request:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Here's useful context...\n\n[large context]\n\nUser's question: What is Go?"
    }
  ],
  "n_predict": 0,
  "temperature": 0.7
}
```

**Tip:** Temp files are created in your system's temp directory (usually `/tmp` on Unix, `%TEMP%` on Windows) and are not automatically deleted, so you can inspect them after the request completes.

## Troubleshooting

### mDNS `.local` Hostname Resolution on Linux

**Problem:** `dial tcp: lookup studio.local: no such host`

**Cause:** Go's HTTP client doesn't resolve `.local` mDNS hostnames by default

**Solutions:**

1. **Use IP address (simplest):**
   ```bash
   # Find the IP
   ping -c 1 studio.local

   # Use IP instead
   ./bioproxy -backend http://192.168.1.X:8081 -port 8080
   ```

2. **Add to /etc/hosts:**
   ```bash
   echo "192.168.1.X studio.local" | sudo tee -a /etc/hosts
   ```

3. **Install mDNS support (Raspberry Pi/Debian):**
   ```bash
   sudo apt-get install libnss-mdns
   sudo systemctl restart avahi-daemon
   ```

### Template File Not Found

**Problem:** `Warning: Failed to read template context.txt`

**Solutions:**
- Check file paths in `config.json` are correct (relative to where you run the binary)
- Use absolute paths if needed: `"/home/user/templates/context.txt"`

### Included File Errors

**Problem:** `[Error reading /tmp/context.txt: no such file]` appears in prompt

**Cause:** Template references `<{/tmp/context.txt}>` but file doesn't exist

**Solutions:**
- Create the file before sending requests
- Use absolute paths that exist
- Template will continue working, just shows error message

## License

MIT

## Contributing

This is a simple learning project. Feel free to fork and customize for your needs!

See [CLAUDE.md](CLAUDE.md) for development notes and implementation details.
