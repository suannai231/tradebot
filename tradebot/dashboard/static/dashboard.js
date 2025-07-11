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
        console.log('📨 WebSocket message received:', message);
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
        console.log('🔄 Frontend received price update:', data);
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
            console.log(`📊 Updating chart for selected symbol: ${symbol}`);
            this.updateChart();
        } else {
            console.log(`📊 Symbol ${symbol} not selected, current: ${this.selectedSymbol}`);
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
            const response = await fetch('/api/system-stats', {
                timeout: 5000,  // 5 second timeout
                signal: AbortSignal.timeout(5000)
            });
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const stats = await response.json();
            console.log('System stats:', stats);
            
            document.getElementById('total-symbols').textContent = stats.total_symbols.toLocaleString();
            document.getElementById('active-symbols').textContent = stats.active_symbols.toLocaleString();
            document.getElementById('total-ticks').textContent = stats.total_ticks.toLocaleString();
            document.getElementById('total-signals').textContent = stats.total_signals.toLocaleString();
            
            // Clear any previous API error status and restore connection
            this.clearApiError();
            this.updateConnectionStatus(true);
            console.log('Updated DOM elements with stats');
        } catch (error) {
            console.error('Error loading system stats:', error);
            this.handleApiError('Failed to load system statistics');
        }
    }

    async loadMarketSummary() {
        try {
            const response = await fetch('/api/market-summary?limit=20', {
                timeout: 5000,
                signal: AbortSignal.timeout(5000)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
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
                
            // Clear any previous API error status and restore connection
            this.clearApiError();
            this.updateConnectionStatus(true);
                
        } catch (error) {
            console.error('Error loading market summary:', error);
            this.handleApiError('Failed to load market data');
            
            // Show error state in market summary
            const tbody = document.getElementById('market-summary-body');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger"><i class="fas fa-exclamation-triangle"></i> Unable to load market data</td></tr>';
            }
            
            document.getElementById('market-last-update').textContent = 
                `Connection lost at ${new Date().toLocaleTimeString()}`;
        }
    }

    async loadSystemHealth() {
        try {
            const response = await fetch('/api/system-health', {
                timeout: 5000,
                signal: AbortSignal.timeout(5000)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
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
            
            // Clear any previous API error status and restore connection
            this.clearApiError();
            this.updateConnectionStatus(true);
            
        } catch (error) {
            console.error('Error loading system health:', error);
            this.handleApiError('Failed to load system health');
            
            // Show error state in system health
            const container = document.getElementById('system-health');
            if (container) {
                container.innerHTML = `
                    <div class="health-service">
                        <div>
                            <div class="service-name text-danger">Connection Failed</div>
                            <div class="service-uptime">Unable to reach backend services</div>
                        </div>
                        <div class="service-status">
                            <i class="fas fa-circle text-danger me-2"></i>
                            <span>disconnected</span>
                        </div>
                    </div>
                `;
            }
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

    handleApiError(message) {
        console.warn('API Error:', message);
        
        // Update connection status to show disconnection
        this.updateConnectionStatus(false);
        
        // Add error to activity log
        this.addActivityItem('error', message, new Date().toISOString());
        
        // Show error notification (if element exists)
        const errorContainer = document.getElementById('api-error-container');
        if (errorContainer) {
            errorContainer.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <i class="fas fa-exclamation-triangle"></i> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }
    }

    clearApiError() {
        // Clear any API error notifications
        const errorContainer = document.getElementById('api-error-container');
        if (errorContainer) {
            errorContainer.innerHTML = '';
        }
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
      const stratEl = document.getElementById('bt-strategy');
      // Fallback to 'mean_reversion' if single-strategy dropdown is not present
      const strat = stratEl ? stratEl.value : 'mean_reversion';
      const startRaw = document.getElementById('bt-start').value;
      const endRaw = document.getElementById('bt-end').value;
      console.log('Backtest inputs:', { sym, strat, startRaw, endRaw });
      
      // convert to ISO yyyy-mm-dd regardless of browser locale
      const start = new Date(startRaw).toISOString().slice(0,10);
      const end   = new Date(endRaw).toISOString().slice(0,10);
      const out = document.getElementById('bt-result');
      if (!sym || !start || !end) {
          if (out) out.textContent = 'Please enter symbol, start and end dates.';
          return;
      }
      if (out) out.textContent = 'Running back-test…';
      try {
          console.log('Making backtest API call...');
          const resp = await fetch(`/api/backtest?symbol=${sym}&strategy=${strat}&start=${start}&end=${end}&adjust_method=none`);
          console.log('Backtest response:', resp);
          if (!resp.ok) {
              const txt = await resp.text();
              throw new Error(txt);
          }
          const data = await resp.json();
          console.log('Backtest data:', data);
          if (out) out.textContent = `${sym} ${strat}\nReturn: ${data.total_return_pct.toFixed(2)}%\nTrades: ${data.total_trades}\nWin-rate: ${data.win_rate.toFixed(1)}%\nSharpe: ${data.sharpe_ratio.toFixed(2)}`;
      } catch (e) {
          console.error('Backtest error:', e);
          if (out) out.textContent = 'Error: ' + (e.message || e);
      }
  });
  console.log('Backtest button event listener added');
});

// ---------------- Advanced Backtest & ML Strategy Logic ----------------

// All available strategies
const ALL_STRATEGIES = [
    'mean_reversion', 'simple_ma', 'advanced', 'low_volume',
    'momentum_breakout', 'volatility_mean_reversion', 'gap_trading',
    'multi_timeframe', 'risk_managed', 'aggressive_mean_reversion',
    'enhanced_momentum', 'multi_timeframe_momentum',
    'ensemble_ml', 'lstm_ml', 'sentiment_ml', 'rl_ml'
];

// Strategy display names
const STRATEGY_NAMES = {
    'mean_reversion': 'Mean Reversion',
    'simple_ma': 'Simple MA',
    'advanced': 'Advanced',
    'low_volume': 'Low Volume',
    'momentum_breakout': 'Momentum Breakout',
    'volatility_mean_reversion': 'Volatility Mean Reversion',
    'gap_trading': 'Gap Trading',
    'multi_timeframe': 'Multi Timeframe',
    'risk_managed': 'Risk Managed',
    'aggressive_mean_reversion': 'Aggressive Mean Reversion',
    'enhanced_momentum': 'Enhanced Momentum',
    'multi_timeframe_momentum': 'Multi-Timeframe Momentum',
    'ensemble_ml': '🤖 Ensemble ML',
    'lstm_ml': '🧠 LSTM Deep Learning',
    'sentiment_ml': '📰 Sentiment Analysis',
    'rl_ml': '🎯 Reinforcement Learning'
};

// Strategy categories
const STRATEGY_CATEGORIES = {
    'Traditional': [
        'mean_reversion', 'simple_ma', 'advanced', 'low_volume',
        'momentum_breakout', 'volatility_mean_reversion', 'gap_trading',
        'multi_timeframe', 'risk_managed', 'aggressive_mean_reversion',
        'enhanced_momentum', 'multi_timeframe_momentum'
    ],
    'Machine Learning': [
        'ensemble_ml', 'lstm_ml', 'sentiment_ml', 'rl_ml'
    ]
};

// Initialize strategy selection UI
function initializeStrategySelection() {
    // Populate traditional strategies
    const traditionalList = document.getElementById('traditional-strategies-list');
    if (traditionalList) {
        STRATEGY_CATEGORIES['Traditional'].forEach(strategy => {
            const div = document.createElement('div');
            div.className = 'form-check';
            div.innerHTML = `
                <input class="form-check-input traditional-strategy-checkbox" type="checkbox" id="check-${strategy}" value="${strategy}" checked>
                <label class="form-check-label" for="check-${strategy}">
                    ${STRATEGY_NAMES[strategy]}
                </label>
            `;
            traditionalList.appendChild(div);
        });
    }

    // Populate ML strategies
    const mlList = document.getElementById('ml-strategies-list');
    if (mlList) {
        STRATEGY_CATEGORIES['Machine Learning'].forEach(strategy => {
            const div = document.createElement('div');
            div.className = 'form-check';
            div.innerHTML = `
                <input class="form-check-input ml-strategy-checkbox" type="checkbox" id="check-${strategy}" value="${strategy}" checked>
                <label class="form-check-label" for="check-${strategy}">
                    ${STRATEGY_NAMES[strategy]}
                </label>
            `;
            mlList.appendChild(div);
        });
    }

    // Setup event listeners
    const strategySelector = document.getElementById('bt-strategy-selector');
    if (strategySelector) {
        strategySelector.addEventListener('change', function() {
            const selection = this.value;
            const customSelection = document.getElementById('custom-strategy-selection');
            
            if (customSelection && customSelection.style) {
                if (selection === 'custom') {
                    customSelection.style.display = 'block';
                } else {
                    customSelection.style.display = 'none';
                }
            }
        });
    }

    // Select all traditional strategies
    const selectAllTraditional = document.getElementById('select-all-traditional');
    if (selectAllTraditional) {
        selectAllTraditional.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.traditional-strategy-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
    }

    // Select all ML strategies
    const selectAllML = document.getElementById('select-all-ml');
    if (selectAllML) {
        selectAllML.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.ml-strategy-checkbox');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
    }
}

// Get selected strategies based on current selection
function getSelectedStrategies() {
    const selector = document.getElementById('bt-strategy-selector');
    if (!selector) return ALL_STRATEGIES;
    
    const selection = selector.value;
    
    switch (selection) {
        case 'all':
            return ALL_STRATEGIES;
        case 'traditional':
            return STRATEGY_CATEGORIES['Traditional'];
        case 'ml':
            return STRATEGY_CATEGORIES['Machine Learning'];
        case 'custom':
            const selectedStrategies = [];
            document.querySelectorAll('.traditional-strategy-checkbox:checked, .ml-strategy-checkbox:checked').forEach(checkbox => {
                selectedStrategies.push(checkbox.value);
            });
            return selectedStrategies;
        default:
            return ALL_STRATEGIES;
    }
}

// Run all selected strategies for backtest
async function runAllStrategiesBacktest() {
    console.log('runAllStrategiesBacktest() called');
    
    // Check if required DOM elements exist
    const symbolInput = document.getElementById('bt-symbol');
    const startInput = document.getElementById('bt-start');
    const endInput = document.getElementById('bt-end');
    
    if (!symbolInput || !startInput || !endInput) {
        console.warn('Backtest UI elements not found, skipping backtest');
        return;
    }
    
    const sym = symbolInput.value.trim().toUpperCase();
    const startRaw = startInput.value;
    const endRaw = endInput.value;
    
    // Validate inputs
    if (!sym || !startRaw || !endRaw) {
        const statusDiv = document.getElementById('bt-status');
        if (statusDiv) {
            statusDiv.innerHTML = '<div class="alert alert-warning">Please enter symbol, start and end dates.</div>';
        }
        return;
    }
    
    // Get selected strategies
    const selectedStrategies = getSelectedStrategies();
    
    if (selectedStrategies.length === 0) {
        const statusDiv = document.getElementById('bt-status');
        if (statusDiv) {
            statusDiv.innerHTML = '<div class="alert alert-warning">Please select at least one strategy.</div>';
        }
        return;
    }
    
    // Convert dates to ISO format
    const start = new Date(startRaw).toISOString().slice(0,10);
    const end = new Date(endRaw).toISOString().slice(0,10);
    
    // Show loading status
    const statusDiv = document.getElementById('bt-status');
    const resultsContainer = document.getElementById('bt-results-container');
    const resultsTable = document.getElementById('bt-results-table');
    
    if (statusDiv) {
        statusDiv.innerHTML = `<div class="alert alert-info">
            <i class="fas fa-spinner fa-spin"></i> Running backtest for ${sym} across ${selectedStrategies.length} strategies...
            <div class="progress mt-2">
                <div id="bt-progress" class="progress-bar" role="progressbar" style="width: 0%"></div>
            </div>
        </div>`;
    }
    
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (resultsTable) resultsTable.innerHTML = '';
    
    const results = [];
    let completedStrategies = 0;
    
    // Test each selected strategy
    for (const strategy of selectedStrategies) {
        try {
            console.log(`Testing strategy: ${strategy}`);
            const response = await fetch(`/api/backtest?symbol=${sym}&strategy=${strategy}&start=${start}&end=${end}&adjust_method=backward`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
            
            const data = await response.json();
            results.push({
                strategy: strategy,
                name: STRATEGY_NAMES[strategy],
                ...data
            });
            
        } catch (error) {
            console.error(`Error testing ${strategy}:`, error);
            results.push({
                strategy: strategy,
                name: STRATEGY_NAMES[strategy],
                error: error.message,
                total_return_pct: 0,
                total_trades: 0,
                win_rate: 0,
                sharpe_ratio: 0,
                max_drawdown: 0,
                profit_factor: 0
            });
        }
        
        // Update progress
        completedStrategies++;
        const progress = (completedStrategies / selectedStrategies.length) * 100;
        const progressBar = document.getElementById('bt-progress');
        if (progressBar && progressBar.style) {
            progressBar.style.width = `${progress}%`;
        }
    }
    
    // Sort results by return percentage (best first)
    results.sort((a, b) => (b.total_return_pct || 0) - (a.total_return_pct || 0));
    
    // Display results in table
    displayBacktestResults(sym, results);
    
    // Hide loading status
    if (statusDiv) {
        statusDiv.innerHTML = `<div class="alert alert-success">
            <i class="fas fa-check-circle"></i> Backtest completed for ${sym}! 
            Tested ${selectedStrategies.length} strategies from ${start} to ${end}.
        </div>`;
    }
}

function displayBacktestResults(symbol, results) {
    const resultsTable = document.getElementById('bt-results-table');
    const resultsContainer = document.getElementById('bt-results-container');
    
    if (!resultsTable || !resultsContainer) return;
    
    resultsTable.innerHTML = '';
    
    results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        // Add ranking class for top performers
        if (index === 0 && result.total_return_pct > 0) {
            row.classList.add('table-success');
        } else if (index < 3 && result.total_return_pct > 0) {
            row.classList.add('table-light');
        } else if (result.total_return_pct < -10) {
            row.classList.add('table-danger');
        }
        
        // Add ML strategy indicator
        const isMLStrategy = STRATEGY_CATEGORIES['Machine Learning'].includes(result.strategy);
        if (isMLStrategy) {
            row.classList.add('ml-strategy-row');
        }
        
        const returnPct = result.total_return_pct || 0;
        const trades = result.total_trades || 0;
        const winRate = result.win_rate || 0;
        const sharpe = result.sharpe_ratio || 0;
        const maxDrawdown = result.max_drawdown || 0;
        const profitFactor = result.profit_factor || 0;
        
        // Format cells with appropriate styling
        const returnClass = returnPct > 0 ? 'text-success' : returnPct < 0 ? 'text-danger' : 'text-muted';
        const sharpeClass = sharpe > 0.5 ? 'text-success' : sharpe < 0 ? 'text-danger' : 'text-muted';
        
        // Add ML indicator to strategy name
        const strategyName = result.name;
        const mlIndicator = isMLStrategy ? ' <span class="ml-strategy-badge">ML</span>' : '';
        
        row.innerHTML = `
            <td><strong>${strategyName}</strong>${mlIndicator}${result.error ? ' <i class="fas fa-exclamation-triangle text-warning" title="' + result.error + '"></i>' : ''}</td>
            <td class="${returnClass}"><strong>${returnPct.toFixed(2)}%</strong></td>
            <td>${trades}</td>
            <td>${winRate.toFixed(1)}%</td>
            <td class="${sharpeClass}">${sharpe.toFixed(2)}</td>
            <td class="text-warning">${maxDrawdown.toFixed(1)}%</td>
            <td>${profitFactor.toFixed(2)}</td>
        `;
        
        resultsTable.appendChild(row);
    });
    
    if (resultsContainer && resultsContainer.style) {
        resultsContainer.style.display = 'block';
    }
}

// Set default backtest dates
function setDefaultBacktestDates() {
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    // Format dates as YYYY-MM-DD
    const formatDate = (date) => {
        return date.toISOString().split('T')[0];
    };
    
    const startInput = document.getElementById('bt-start');
    const endInput = document.getElementById('bt-end');
    
    if (startInput) startInput.value = formatDate(oneYearAgo);
    if (endInput) endInput.value = formatDate(today);
}

// Ensure these are available globally
window.initializeStrategySelection = initializeStrategySelection;
window.runAllStrategiesBacktest = runAllStrategiesBacktest;
window.setDefaultBacktestDates = setDefaultBacktestDates;

// Export strategy arrays for K-line chart
window.ALL_STRATEGIES = ALL_STRATEGIES;
window.STRATEGY_NAMES = STRATEGY_NAMES;
window.STRATEGY_CATEGORIES = STRATEGY_CATEGORIES; 