# HTTP Proxy with JSON Explorer

A simple HTTP proxy server that logs requests/responses and provides an interactive web interface for exploring JSON content.

## Features

- HTTP proxy that forwards requests to a target server
- Logs all requests and responses with full details
- Web-based explorer interface for viewing logged requests
- Interactive JSON viewer for request/response bodies
- Search functionality to filter logs
- Real-time updates (auto-refresh every 5 seconds)

## Installation

```bash
npm install
```

## Usage

### Starting the proxy

```bash
# Default: proxy on port 3000, explorer on port 3001, target http://example.com
npm start

# With custom configuration
TARGET_URL=https://api.example.com PORT=8080 EXPLORER_PORT=8081 npm start
```

### Environment Variables

- `PORT` - Proxy server port (default: 3000)
- `EXPLORER_PORT` - Explorer web interface port (default: 3001)
- `TARGET_URL` - Target server to proxy requests to (default: http://example.com)

### Using the proxy

1. Configure your application to use `http://localhost:3000` as the HTTP proxy
2. All requests will be forwarded to the target URL
3. Open `http://localhost:3001` in your browser to view the explorer interface

### Explorer Interface

- **Search**: Filter requests by URL, method, or content
- **Refresh**: Manually refresh the log list
- **Clear All**: Remove all logged requests
- **Click on a request**: View full details including headers and JSON bodies

## Example

To test with a simple JSON API:

```bash
# Start proxy targeting JSONPlaceholder API
TARGET_URL=https://jsonplaceholder.typicode.com npm start

# In another terminal, make a request through the proxy
curl http://localhost:3000/posts/1

# View the logged request at http://localhost:3001
```

## Notes

- The proxy stores up to 1000 recent requests in memory
- Request/response bodies are limited to 10MB
- JSON content is automatically parsed and formatted in the explorer