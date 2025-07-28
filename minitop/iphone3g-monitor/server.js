const express = require('express');
const os = require('os-utils');
const osInfo = require('os');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve static files
app.use(express.static('public'));

// CPU usage endpoint
app.get('/api/cpu', (req, res) => {
    os.cpuUsage((usage) => {
        res.json({ 
            cpu: Math.round(usage * 100),
            timestamp: Date.now()
        });
    });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
    
    // Get network interfaces
    const interfaces = osInfo.networkInterfaces();
    console.log('\nAccess from your iPhone 3G at:');
    
    Object.keys(interfaces).forEach(name => {
        interfaces[name].forEach(iface => {
            if (iface.family === 'IPv4' && !iface.internal) {
                console.log(`  http://${iface.address}:${PORT}`);
            }
        });
    });
});