# BioProxy - Development Notes

## Project Origin

This project was created as a learning exercise for Go, with the goal of building a practical tool for enhancing LLM workflows. The original requirement was to dynamically inject context into LLM prompts without modifying the llama.cpp server code.

## Original Problem Statement

When using llama.cpp server with its web UI:
- Need to inject dynamic context from files into LLM conversations
- Context files change frequently (updated by external processes)
- Don't want to restart server or manually copy-paste context
- Want different "modes" triggered by message prefixes (e.g., `/context`, `/mini`)

## Design Decisions

### Why a Proxy?

**Alternatives considered:**
1. **Modify llama.cpp server** - Requires C++ knowledge, hard to maintain across updates
2. **Use existing LLM gateways** (LiteLLM, Portkey) - Too heavy, require Python, more complex than needed
3. **Custom lightweight proxy** - ✅ Chosen: Simple, portable, learning opportunity

**Proxy advantages:**
- Works with ANY OpenAI-compatible server (llama.cpp, vLLM, Ollama, etc.)
- Single binary, no runtime dependencies
- Easy to understand and modify (~270 LoC)
- Great Go learning project

### Why Go?

- Built-in HTTP server/client (no dependencies needed)
- Cross-compilation is trivial
- Single binary deployment
- Fast and efficient for proxying
- Good learning experience

### Architecture Choices

**Request flow:**
```
Browser → Proxy (intercept /v1/chat/completions) → LLM Server
           ↓
       1. Read template
       2. Process <{file}> includes
       3. Replace __the__user__message__
       4. Forward modified request
```

**Why intercept only `/v1/chat/completions`:**
- Web UI makes requests to same origin
- Proxy forwards all other requests (static files, health checks, etc.) unchanged
- Transparent to the client

**File reading strategy:**
- Templates read fresh on every request (no caching)
- Allows external processes to update files without proxy restart
- Slight performance cost acceptable for flexibility

### Dynamic File Inclusion Feature

**Syntax:** `<{filepath}>`

**Why this syntax:**
- Unlikely to conflict with actual message content
- Easy to regex match: `<\{([^}]+)\}>`
- Visually distinct from markdown/code syntax
- Similar to template languages

**Implementation:**
1. Read template file
2. Find all `<{filepath}>` patterns
3. Replace each with content of the file (read fresh)
4. Then replace `__the__user__message__` with actual user message
5. Send to LLM

**Error handling:**
- If included file doesn't exist, insert error message instead of failing
- Logs warnings but continues processing
- Graceful degradation

## Implementation Notes

### Cross-Platform Builds

Uses Makefile with Go's built-in cross-compilation:
- `GOOS=linux GOARCH=arm64` for Raspberry Pi
- `GOOS=darwin GOARCH=arm64` for macOS
- Binaries output to `bin/` directory with platform suffixes

### mDNS Resolution Issue

**Problem:** Go's HTTP client doesn't resolve `.local` mDNS hostnames by default

**Why:** Go uses its own DNS resolver, not system libraries (which support mDNS via Avahi)

**Solutions:**
1. Use IP addresses instead of `.local` names
2. Add entries to `/etc/hosts`
3. Install `libnss-mdns` on Linux systems

### Template Files

**Simple templates (no includes):**
```
Answer concisely: __the__user__message__
```

**With dynamic includes:**
```
Latest git commits:
<{/tmp/recent_commits.txt}>

Question: __the__user__message__
```

**Multiple includes:**
```
Project: <{/tmp/project.txt}>
Commits: <{/tmp/commits.txt}>
Question: __the__user__message__
```

## What Went Well

1. **Go standard library is excellent** - Zero dependencies needed
2. **httputil.ReverseProxy** - Built-in reverse proxy made implementation trivial
3. **Cross-compilation** - Single command to build for different platforms
4. **Clean architecture** - Simple, easy to understand and extend
5. **Learning experience** - Good introduction to Go for practical use

## Issues Encountered

### 1. Web UI Port Configuration
**Issue:** Initially thought web UI could be configured to use different endpoint

**Reality:** Web UI makes relative requests to same origin, so proxy must serve on the port user accesses

**Solution:** Proxy forwards ALL requests, only intercepting `/v1/chat/completions`

### 2. mDNS Hostname Resolution
**Issue:** `studio.local` didn't resolve in Go HTTP client

**Why:** Go's DNS resolver doesn't use system mDNS by default

**Solution:** Document the issue and provide workarounds (use IPs, /etc/hosts, or install libnss-mdns)

### 3. JSON Handling
**Issue:** Need to preserve all request fields when modifying messages

**Solution:** Parse into `map[string]interface{}`, extract messages separately, then merge back

## Future Enhancement Ideas

- [ ] Configuration hot-reload (watch config.json for changes)
- [ ] Metrics/logging (requests processed, templates used, etc.)
- [ ] Authentication passthrough
- [ ] Template variables (e.g., `<{$timestamp}>`, `<{$user}>`)
- [ ] Regex-based prefix matching
- [ ] Multiple message modification (not just first user message)
- [ ] Template composition/inheritance
- [ ] WebSocket support for streaming

## Files Overview

```
bioproxy/
├── main.go           # Core proxy implementation (~270 lines)
├── config.json       # Prefix → template mappings
├── Makefile          # Cross-platform build targets
├── .gitignore        # Excludes bin/ and build artifacts
├── README.md         # User documentation
├── CLAUDE.md         # This file - development notes
├── context.txt       # Example template with static context
├── mini.txt          # Example template for concise answers
└── bin/              # Build output (gitignored)
    ├── bioproxy-darwin-arm64
    └── bioproxy-linux-arm64
```

## Dependencies

**Runtime:** None (Go standard library only)

**Build:** Go 1.21+ (uses generics)

**Imports:**
- `encoding/json` - Parse/marshal request/response
- `net/http` - HTTP server
- `net/http/httputil` - Reverse proxy
- `regexp` - Template file inclusion pattern matching
- `flag` - Command-line arguments
- Standard utilities (io, os, strings, etc.)

## Testing Notes

**Manual testing performed:**
1. ✅ Prefix detection and template injection
2. ✅ Dynamic file inclusion with `<{filepath}>`
3. ✅ Multiple file includes in one template
4. ✅ Error handling (missing files)
5. ✅ Cross-compilation for ARM64 Linux
6. ✅ Deployment to Raspberry Pi

**Not yet tested:**
- Streaming responses
- Large file includes (>1MB)
- Concurrent requests
- Edge cases (malformed JSON, etc.)

## Deployment

**Development:**
```bash
make
bin/bioproxy-darwin-arm64 -backend http://localhost:8080 -port 8081
```

**Production (Raspberry Pi):**
```bash
make linux-arm64
scp bin/bioproxy-linux-arm64 pi@raspberrypi:/home/pi/bioproxy
ssh pi@raspberrypi
./bioproxy -backend http://192.168.1.X:8081 -port 8080
```

**Systemd service (future):**
Could create a systemd unit file for auto-start on boot.

## Code Quality

- Well-commented
- Single responsibility functions
- Error handling at each step
- Logging for debugging
- No global mutable state (beyond config)
- Type-safe JSON handling

## Learning Outcomes

**Go language:**
- HTTP server/client patterns
- Reverse proxy implementation
- JSON marshaling with interfaces
- Regex usage
- Cross-compilation
- Flag parsing

**System design:**
- Proxy architecture patterns
- Template systems
- Dynamic content injection
- Configuration management
- Error handling strategies

## Related Resources

- [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
- [Go net/http documentation](https://pkg.go.dev/net/http)
- [LiteLLM proxy](https://github.com/BerriAI/litellm) - Alternative solution
