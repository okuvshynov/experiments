const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const bodyParser = require('body-parser');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const EXPLORER_PORT = process.env.EXPLORER_PORT || 3001;

const requestLog = [];

function captureRequestBody(proxyReq, req, res) {
  if (req.body) {
    const bodyData = JSON.stringify(req.body);
    proxyReq.setHeader('Content-Type', 'application/json');
    proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
    proxyReq.write(bodyData);
  }
}

function onProxyReq(proxyReq, req, res) {
  const requestId = uuidv4();
  req.requestId = requestId;
  
  // Debug logging
  console.log('[DEBUG] onProxyReq called');
  console.log('[DEBUG] Original request URL:', req.url);
  console.log('[DEBUG] proxyReq.path:', proxyReq.path);
  console.log('[DEBUG] proxyReq.host:', proxyReq.host);
  console.log('[DEBUG] proxyReq.hostname:', proxyReq.hostname);
  console.log('[DEBUG] proxyReq.port:', proxyReq.port);
  console.log('[DEBUG] proxyReq.protocol:', proxyReq.protocol);
  console.log('[DEBUG] proxyReq.agent:', proxyReq.agent);
  console.log('[DEBUG] Host header:', proxyReq.getHeader('host'));
  
  // Construct target URL - try to get it from the actual request
  let targetUrl = TARGET_URL + proxyReq.path;
  
  const logEntry = {
    id: requestId,
    timestamp: new Date().toISOString(),
    method: req.method,
    url: req.url,
    headers: req.headers,
    body: req.body,
    targetUrl: targetUrl,
    response: null
  };
  
  requestLog.push(logEntry);
  
  if (requestLog.length > 1000) {
    requestLog.shift();
  }
  
  // Handle request body
  if (req.body) {
    captureRequestBody(proxyReq, req, res);
  }
}

function onProxyRes(proxyRes, req, res) {
  console.log('[DEBUG] onProxyRes called');
  console.log('[DEBUG] Response status:', proxyRes.statusCode);
  console.log('[DEBUG] Response headers:', proxyRes.headers);
  
  let responseData = '';
  
  const originalWrite = res.write;
  const originalEnd = res.end;
  
  res.write = function(chunk) {
    responseData += chunk;
    originalWrite.apply(res, arguments);
  };
  
  res.end = function(chunk) {
    if (chunk) {
      responseData += chunk;
    }
    
    const logEntry = requestLog.find(entry => entry.id === req.requestId);
    if (logEntry) {
      logEntry.response = {
        status: proxyRes.statusCode,
        statusMessage: proxyRes.statusMessage,
        headers: proxyRes.headers,
        body: tryParseJSON(responseData)
      };
    }
    
    originalEnd.apply(res, arguments);
  };
}

function tryParseJSON(data) {
  try {
    return JSON.parse(data);
  } catch (e) {
    return data;
  }
}

// Add request logging middleware
app.use((req, res, next) => {
  console.log('[DEBUG] Incoming request:', req.method, req.url);
  console.log('[DEBUG] Request headers:', req.headers);
  next();
});

app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
app.use(bodyParser.text({ type: 'text/*' }));

const TARGET_URL = process.env.TARGET_URL || 'http://example.com';
console.log('[DEBUG] Creating proxy with target:', TARGET_URL);

const proxyMiddleware = createProxyMiddleware({
  target: TARGET_URL,
  changeOrigin: true,
  onProxyReq: onProxyReq,
  onProxyRes: onProxyRes,
  selfHandleResponse: false,
  logLevel: 'debug',
  onError: (err, req, res) => {
    console.error('[PROXY ERROR]', err.message);
    console.error('[PROXY ERROR] Target:', TARGET_URL);
    console.error('[PROXY ERROR] Request URL:', req.url);
    res.status(500).json({ error: 'Proxy error', message: err.message });
  }
});

app.use('/', proxyMiddleware);

const explorerApp = express();
explorerApp.use(cors());
explorerApp.use(express.static(path.join(__dirname, '..', 'public')));

explorerApp.get('/api/logs', (req, res) => {
  const { limit = 100, offset = 0, search = '' } = req.query;
  
  let filteredLogs = requestLog;
  
  if (search) {
    filteredLogs = requestLog.filter(log => {
      const searchLower = search.toLowerCase();
      return (
        log.url.toLowerCase().includes(searchLower) ||
        log.method.toLowerCase().includes(searchLower) ||
        JSON.stringify(log.body).toLowerCase().includes(searchLower) ||
        (log.response && JSON.stringify(log.response.body).toLowerCase().includes(searchLower))
      );
    });
  }
  
  const paginatedLogs = filteredLogs
    .slice(-limit - offset)
    .slice(-limit)
    .reverse();
  
  res.json({
    logs: paginatedLogs,
    total: filteredLogs.length
  });
});

explorerApp.get('/api/logs/:id', (req, res) => {
  const log = requestLog.find(entry => entry.id === req.params.id);
  if (log) {
    res.json(log);
  } else {
    res.status(404).json({ error: 'Log entry not found' });
  }
});

explorerApp.delete('/api/logs', (req, res) => {
  requestLog.length = 0;
  res.json({ message: 'All logs cleared' });
});

app.listen(PORT, () => {
  console.log(`Proxy server running on port ${PORT}`);
  console.log(`Target URL: ${process.env.TARGET_URL || 'http://example.com'}`);
});

explorerApp.listen(EXPLORER_PORT, () => {
  console.log(`Explorer interface running on port ${EXPLORER_PORT}`);
  console.log(`Access the explorer at http://localhost:${EXPLORER_PORT}`);
});