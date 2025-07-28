(function() {
    // Prevent scrolling/bouncing
    document.addEventListener('touchmove', function(e) {
        e.preventDefault();
    }, false);
    
    var chartElement = document.getElementById('chart');
    var cpuValueElement = document.getElementById('cpu-value');
    var dataPoints = [];
    var maxDataPoints = 80; // Number of bars that fit in the chart width
    
    // Create initial bars
    for (var i = 0; i < maxDataPoints; i++) {
        var bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.height = '0%';
        chartElement.appendChild(bar);
        dataPoints.push(0);
    }
    
    function updateChart(cpuPercent) {
        // Update data
        dataPoints.shift();
        dataPoints.push(cpuPercent);
        
        // Update bars
        var bars = chartElement.getElementsByClassName('bar');
        for (var i = 0; i < bars.length; i++) {
            bars[i].style.height = dataPoints[i] + '%';
        }
        
        // Update text
        cpuValueElement.textContent = cpuPercent + '%';
    }
    
    function fetchCPU() {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/cpu', true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                try {
                    var data = JSON.parse(xhr.responseText);
                    updateChart(data.cpu);
                } catch (e) {
                    console.error('Error parsing JSON:', e);
                }
            }
        };
        xhr.send();
    }
    
    // Initial fetch
    fetchCPU();
    
    // Auto-refresh every 2 seconds
    setInterval(fetchCPU, 2000);
})();