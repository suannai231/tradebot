// Trading Bot Dashboard JavaScript
console.log('Dashboard JavaScript loaded successfully!');

class TradingDashboard {
    constructor() {
        this.ws = null;
        this.priceChart = null;
        this.priceData = new Map(); // Store price data for each symbol
        this.maxDataPoints = 50;
        this.selectedSymbol = null;
        this.activityLogItems = [];
        this.maxActivityItems = 100;
        
        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupChart();
        this.loadInitialData();
        this.setupEventListeners();
        this.startDataRefresh();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        console.log('Attempting WebSocket connection to', wsUrl);
        
        this.connectWebSocket(wsUrl);
    }

    connectWebSocket(url) {
        try {
            this.ws = new WebSocket(url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(url), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'connection_established':
                console.log('Dashboard connection established');
                break;
                
            case 'price_update':
                this.handlePriceUpdate(message.data);
                break;
                
            case 'trading_signal':
                this.handleTradingSignal(message.data);
                break;
                
            case 'ping':
                // Respond to ping to keep connection alive
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({type: 'pong'}));
                }
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handlePriceUpdate(data) {
        const { symbol, price, timestamp, volume } = data;
        
        // Store price data
        if (!this.priceData.has(symbol)) {
            this.priceData.set(symbol, []);
        }
        
        const symbolData = this.priceData.get(symbol);
        symbolData.push({
            time: new Date(timestamp),
            price: price,
            volume: volume
        });
        
        // Keep only recent data points
        if (symbolData.length > this.maxDataPoints) {
            symbolData.shift();
        }
        
        // Update chart if this symbol is selected
        if (this.selectedSymbol === symbol) {
            this.updateChart();
        }
        
        // Add to activity log
        this.addActivityItem('price-update', `${symbol}: $${price.toFixed(2)}`, timestamp);
        
        // Update symbol selector if new symbol
        this.updateSymbolSelector();
    }

    handleTradingSignal(data) {
        const { symbol, signal_type, price, timestamp, strategy, confidence } = data;
        
        // Add to signals panel
        this.addTradingSignal(data);
        
        // Add to activity log
        this.addActivityItem('signal', 
            `${signal_type.toUpperCase()} signal for ${symbol} at $${price.toFixed(2)}`, 
            timestamp
        );
    }

    setupChart() {
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        this.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Price: $${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    updateChart() {
        if (!this.priceChart || !this.selectedSymbol) return;
        
        const symbolData = this.priceData.get(this.selectedSymbol) || [];
        
        this.priceChart.data.labels = symbolData.map(d => d.time);
        this.priceChart.data.datasets[0].data = symbolData.map(d => d.price);
        this.priceChart.data.datasets[0].label = `${this.selectedSymbol} Price`;
        
        this.priceChart.update('none'); // No animation for real-time updates
    }

    updateSymbolSelector() {
        // With free-text input we don't need to maintain a list, but we'll still
        // offer auto-completion if a datalist is present.
        const input = document.getElementById('chart-symbol-input');
        if (!input) return;

        const datalistId = 'symbol-datalist';
        let datalist = document.getElementById(datalistId);
        if (!datalist) {
            datalist = document.createElement('datalist');
            datalist.id = datalistId;
            document.body.appendChild(datalist);
            input.setAttribute('list', datalistId);
        }

        const existing = new Set(Array.from(datalist.options).map(o => o.value));
        const availableSymbols = Array.from(this.priceData.keys()).sort();
        availableSymbols.forEach(symbol => {
            if (!existing.has(symbol)) {
                const option = document.createElement('option');
                option.value = symbol;
                datalist.appendChild(option);
            }
        });
    }

    setupEventListeners() {
        // Symbol text input – press Enter to load chart
        const input = document.getElementById('chart-symbol-input');
        if (input) {
            input.addEventListener('keyup', (e) => {
                if (e.key === 'Enter') {
                    this.selectedSymbol = e.target.value.trim().toUpperCase();
                    if (this.selectedSymbol) {
                        this.updateChart();
                    } else {
                        this.clearChart();
                    }
                }
            });
        }
    }

    clearChart() {
        if (this.priceChart) {
            this.priceChart.data.labels = [];
            this.priceChart.data.datasets[0].data = [];
            this.priceChart.update();
        }
    }

    updateConnectionStatus(connected) {
        const statusIcon = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (connected) {
            statusIcon.className = 'fas fa-circle text-success';
            statusText.textContent = 'Connected';
        } else {
            statusIcon.className = 'fas fa-circle text-danger';
            statusText.textContent = 'Disconnected';
        }
    }

    async loadInitialData() {
        try {
            // Kick off all loads in parallel to minimise total load time
            await Promise.all([
                this.loadSystemStats(),
                this.loadMarketSummary(),
                this.loadSystemHealth(),
                this.loadRecentSignals()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    async loadSystemStats() {
        try {
            console.log('Loading system stats...');
            const response = await fetch('/api/system-stats');
            console.log('Response status:', response.status);
            const stats = await response.json();
            console.log('System stats:', stats);
            
            document.getElementById('total-symbols').textContent = stats.total_symbols.toLocaleString();
            document.getElementById('active-symbols').textContent = stats.active_symbols.toLocaleString();
            document.getElementById('total-ticks').textContent = stats.total_ticks.toLocaleString();
            document.getElementById('total-signals').textContent = stats.total_signals.toLocaleString();
            
            console.log('Updated DOM elements with stats');
        } catch (error) {
            console.error('Error loading system stats:', error);
        }
    }

    async loadMarketSummary() {
        try {
            const response = await fetch('/api/market-summary?limit=20');
            const summary = await response.json();
            
            // ensure autocomplete datalist is populated
            const inputEl = document.getElementById('chart-symbol-input');
            if (inputEl) {
                const datalistId = 'symbol-datalist';
                let datalist = document.getElementById(datalistId);
                if (!datalist) {
                    datalist = document.createElement('datalist');
                    datalist.id = datalistId;
                    document.body.appendChild(datalist);
                    inputEl.setAttribute('list', datalistId);
                }
                const existing = new Set(Array.from(datalist.options).map(o=>o.value));
                summary.forEach(item => {
                    if (item.symbol && !existing.has(item.symbol)) {
                        const opt = document.createElement('option');
                        opt.value = item.symbol;
                        datalist.appendChild(opt);
                        existing.add(item.symbol);
                    }
                });
            }
            
            console.log('Market summary loaded with', summary.length, 'symbols');
            
            const tbody = document.getElementById('market-summary-body');
            
            if (summary.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No data available</td></tr>';
                return;
            }
            
            tbody.innerHTML = summary.map(item => {
                const changeClass = item.change > 0 ? 'price-change-positive' : 
                                  item.change < 0 ? 'price-change-negative' : 'price-change-neutral';
                const changeIcon = item.change > 0 ? '↗' : item.change < 0 ? '↘' : '→';
                
                return `
                    <tr>
                        <td><strong>${item.symbol}</strong></td>
                        <td>$${item.price.toFixed(2)}</td>
                        <td class="${changeClass}">
                            ${changeIcon} ${item.change.toFixed(2)} (${item.change_percent.toFixed(2)}%)
                        </td>
                        <td>${item.volume.toLocaleString()}</td>
                    </tr>
                `;
            }).join('');
            
            document.getElementById('market-last-update').textContent = 
                `Last updated: ${new Date().toLocaleTimeString()}`;
                
        } catch (error) {
            console.error('Error loading market summary:', error);
        }
    }

    async loadSystemHealth() {
        try {
            const response = await fetch('/api/system-health');
            const health = await response.json();
            
            const container = document.getElementById('system-health');
            
            if (health.length === 0) {
                container.innerHTML = '<div class="text-center text-muted">No health data available</div>';
                return;
            }
            
            container.innerHTML = health.map(service => {
                const statusColors = {
                    healthy: 'text-success',
                    degraded: 'text-warning',
                    unhealthy: 'text-danger',
                    dead: 'text-secondary',
                    unknown: 'text-muted',
                    error: 'text-orange'
                };
                const colorClass = statusColors[service.status] || 'text-muted';

                const lastHeartbeat = new Date(service.last_heartbeat);
                const timeDiff = Math.floor((new Date() - lastHeartbeat) / 1000);
                const timeAgo = timeDiff < 60 ? `${timeDiff}s ago` :
                                timeDiff < 3600 ? `${Math.floor(timeDiff/60)}m ago` :
                                `${Math.floor(timeDiff/3600)}h ago`;

                return `
                    <div class="health-service">
                        <div>
                            <div class="service-name">${service.service.replace('_', ' ')}</div>
                            <div class="service-uptime">Last seen: ${timeAgo}</div>
                        </div>
                        <div class="service-status">
                            <i class="fas fa-circle ${colorClass} me-2"></i>
                            <span>${service.status}</span>
                            ${service.error_count > 0 ? `<span class="badge bg-warning ms-2">${service.error_count}</span>` : ''}
                        </div>
                    </div>
                `;
            }).join('');
            
        } catch (error) {
            console.error('Error loading system health:', error);
        }
    }

    async loadRecentSignals() {
        try {
            const response = await fetch('/api/recent-signals?limit=20');
            const signals = await response.json();
            
            const container = document.getElementById('trading-signals');
            
            if (signals.length === 0) {
                container.innerHTML = '<div class="text-center text-muted">No signals yet</div>';
                return;
            }
            
            container.innerHTML = signals.map(signal => {
                // Also mirror into activity feed so user sees historical activity
                this.addActivityItem('signal',
                    `${signal.signal_type.toUpperCase()} signal for ${signal.symbol} at $${signal.price.toFixed(2)}`,
                    signal.timestamp);
                return this.createSignalElement(signal);
            }).join('');
            
        } catch (error) {
            console.error('Error loading recent signals:', error);
        }
    }

    addTradingSignal(signal) {
        const container = document.getElementById('trading-signals');
        
        // Clear "no signals" message if present
        if (container.innerHTML.includes('No signals yet')) {
            container.innerHTML = '';
        }
        
        const signalElement = this.createSignalElement(signal);
        container.insertAdjacentHTML('afterbegin', signalElement);
        
        // Keep only recent signals (limit to 20)
        const signals = container.children;
        while (signals.length > 20) {
            container.removeChild(signals[signals.length - 1]);
        }
    }

    createSignalElement(signal) {
        const signalClass = `signal-${signal.signal_type.toLowerCase()}`;
        const timestamp = new Date(signal.timestamp).toLocaleTimeString();
        const confidence = signal.confidence ? `(${(signal.confidence * 100).toFixed(0)}%)` : '';
        
        return `
            <div class="signal-item ${signalClass}">
                <div class="d-flex justify-content-between">
                    <strong>${signal.symbol}</strong>
                    <small>${timestamp}</small>
                </div>
                <div>${signal.signal_type.toUpperCase()} at $${signal.price.toFixed(2)} ${confidence}</div>
                <small class="text-muted">${signal.strategy || 'Unknown strategy'}</small>
            </div>
        `;
    }

    addActivityItem(type, message, timestamp) {
        const container = document.getElementById('activity-log');
        
        // Clear "waiting" message if present
        if (container.innerHTML.includes('Waiting for activity')) {
            container.innerHTML = '';
        }
        
        const time = new Date(timestamp).toLocaleTimeString();
        const typeClass = `activity-${type.replace('_', '-')}`;
        
        const activityElement = `
            <div class="activity-item">
                <span class="activity-type ${typeClass}">[${type.toUpperCase()}]</span>
                <span>${message}</span>
                <div class="activity-timestamp">${time}</div>
            </div>
        `;
        
        container.insertAdjacentHTML('afterbegin', activityElement);
        
        // Ensure the scroll container is scrolled to the top so the newest item is visible immediately.
        if (container.parentElement) {
            container.parentElement.scrollTop = 0;
        }
        
        // Keep only recent items
        this.activityLogItems.unshift({type, message, timestamp});
        if (this.activityLogItems.length > this.maxActivityItems) {
            this.activityLogItems.pop();
            const items = container.children;
            if (items.length > this.maxActivityItems) {
                container.removeChild(items[items.length - 1]);
            }
        }
    }

    startDataRefresh() {
        // Refresh data every 30 seconds
        setInterval(() => {
            this.loadSystemStats();
            this.loadMarketSummary();
            this.loadSystemHealth();
        }, 30000);
    }

    clearActivityLog() {
        document.getElementById('activity-log').innerHTML = 
            '<div class="text-center text-muted">Waiting for activity...</div>';
        this.activityLogItems = [];
    }
}

// Global functions
function clearActivityLog() {
    if (window.dashboard) {
        window.dashboard.clearActivityLog();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, creating dashboard...');
    window.dashboard = new TradingDashboard();
    console.log('Dashboard created successfully!');
});

// Handle page visibility changes to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Dashboard hidden, reducing update frequency');
    } else {
        console.log('Dashboard visible, resuming normal updates');
        if (window.dashboard) {
            window.dashboard.loadInitialData();
        }
    }
});

// ---------------- Back-test UI ----------------

document.addEventListener('DOMContentLoaded', () => {
  console.log('Setting up backtest button...');
  const btn = document.getElementById('bt-run');
  console.log('Backtest button found:', btn);
  if (!btn) {
    console.error('Backtest button not found!');
    return;
  }
  btn.addEventListener('click', async () => {
      console.log('Backtest button clicked!');
      const sym = document.getElementById('bt-symbol').value.trim().toUpperCase();
      const strat = document.getElementById('bt-strategy').value;
      const startRaw = document.getElementById('bt-start').value;
      const endRaw = document.getElementById('bt-end').value;
      console.log('Backtest inputs:', { sym, strat, startRaw, endRaw });
      
      // convert to ISO yyyy-mm-dd regardless of browser locale
      const start = new Date(startRaw).toISOString().slice(0,10);
      const end   = new Date(endRaw).toISOString().slice(0,10);
      const out = document.getElementById('bt-result');
      if (!sym || !start || !end) {
          out.textContent = 'Please enter symbol, start and end dates.';
          return;
      }
      out.textContent = 'Running back-test…';
      try {
          console.log('Making backtest API call...');
          const resp = await fetch(`/api/backtest?symbol=${sym}&strategy=${strat}&start=${start}&end=${end}`);
          console.log('Backtest response:', resp);
          if (!resp.ok) {
              const txt = await resp.text();
              throw new Error(txt);
          }
          const data = await resp.json();
          console.log('Backtest data:', data);
          out.textContent = `${sym} ${strat}\nReturn: ${data.total_return_pct.toFixed(2)}%\nTrades: ${data.total_trades}\nWin-rate: ${data.win_rate.toFixed(1)}%\nSharpe: ${data.sharpe_ratio.toFixed(2)}`;
      } catch (e) {
          console.error('Backtest error:', e);
          out.textContent = 'Error: ' + (e.message || e);
      }
  });
  console.log('Backtest button event listener added');
}); 