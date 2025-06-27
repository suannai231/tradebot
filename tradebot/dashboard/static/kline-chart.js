/**
 * K-Line Chart Component
 * Provides candlestick chart functionality with multiple timeframes
 */

// At the top of the file, after Chart.js is loaded
console.log('ChartJS version:', Chart.version);
console.log('ChartJS plugins:', Chart.registry ? Chart.registry.plugins.items : Chart.plugins ? Chart.plugins : 'unknown');

// Register the zoom plugin explicitly if available
if (window.ChartZoom) {
    Chart.register(window.ChartZoom);
    console.log('Registered ChartZoom from window.ChartZoom');
} else if (window['chartjs-plugin-zoom']) {
    Chart.register(window['chartjs-plugin-zoom']);
    console.log('Registered ChartZoom from window["chartjs-plugin-zoom"]');
} else {
    console.warn('ChartZoom plugin not found on window');
}

class KLineChart {
    constructor() {
        this.chart = null;
        this.currentSymbol = null;
        this.currentTimeframe = '1D';
        this.availableSymbols = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableSymbols();
        this.setupChart();
        this.setupHorizontalWheelPan(); // Enable horizontal wheel panning
        this.setupResetZoomButton(); // Add reset zoom button
    }

    setupEventListeners() {
        // Symbol selection
        const symbolSelect = document.getElementById('kline-symbol-select');
        const symbolInput = document.getElementById('kline-symbol-input');
        const loadBtn = document.getElementById('kline-load-btn');

        // Timeframe selection
        const timeframeInputs = document.querySelectorAll('input[name="timeframe"]');
        timeframeInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.currentTimeframe = e.target.value;
                if (this.currentSymbol) {
                    this.loadChartData();
                }
            });
        });

        // Load button
        loadBtn.addEventListener('click', () => {
            const symbol = symbolInput.value.trim().toUpperCase() || symbolSelect.value;
            if (symbol) {
                this.currentSymbol = symbol;
                this.loadChartData();
            }
        });

        // Enter key in input
        symbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadBtn.click();
            }
        });

        // Symbol select change
        symbolSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.currentSymbol = e.target.value;
                this.loadChartData();
            }
        });
    }

    async loadAvailableSymbols() {
        try {
            const response = await fetch('/api/available-symbols');
            const symbols = await response.json();
            this.availableSymbols = symbols;
            
            // Populate select dropdown
            const symbolSelect = document.getElementById('kline-symbol-select');
            symbolSelect.innerHTML = '<option value="">Select Symbol</option>';
            
            symbols.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                symbolSelect.appendChild(option);
            });

            // Auto-select first symbol if available
            if (symbols.length > 0) {
                this.currentSymbol = symbols[0];
                symbolSelect.value = this.currentSymbol;
                this.loadChartData();
            }
        } catch (error) {
            console.error('Error loading available symbols:', error);
        }
    }

    setupChart() {
        const ctx = document.getElementById('kline-chart').getContext('2d');
        
        // Custom candlestick plugin
        const candlestickPlugin = {
            id: 'candlestick',
            afterDraw: (chart) => {
                const { ctx, data, scales } = chart;
                const dataset = data.datasets[0];
                
                if (!dataset.data || dataset.data.length === 0) return;
                
                ctx.save();
                // Do NOT clearRect here!
                
                dataset.data.forEach((point, index) => {
                    const x = scales.x.getPixelForValue(point.x);
                    const openY = scales.y.getPixelForValue(point.o);
                    const closeY = scales.y.getPixelForValue(point.c);
                    const highY = scales.y.getPixelForValue(point.h);
                    const lowY = scales.y.getPixelForValue(point.l);
                    
                    if (index < 5) {
                        console.log(`Candle ${index}: x=${x}, openY=${openY}, closeY=${closeY}, highY=${highY}, lowY=${lowY}, o=${point.o}, c=${point.c}, h=${point.h}, l=${point.l}`);
                    }
                    
                    const isUp = point.c >= point.o;
                    const color = isUp ? '#26a69a' : '#ef5350';
                    
                    // Draw wick (high-low line)
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x, highY);
                    ctx.lineTo(x, lowY);
                    ctx.stroke();
                    
                    // Draw body
                    const bodyWidth = Math.max(2, 8);
                    const bodyHeight = Math.abs(closeY - openY);
                    const bodyY = isUp ? closeY : openY;
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(x - bodyWidth/2, bodyY, bodyWidth, bodyHeight);
                    
                    // Draw border
                    ctx.strokeStyle = color;
                    ctx.strokeRect(x - bodyWidth/2, bodyY, bodyWidth, bodyHeight);
                });
                
                ctx.restore();
            }
        };
        
        this.chart = new Chart(ctx, {
            type: 'line', // Use line as base, we'll override drawing
            data: {
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false
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
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM dd',
                                week: 'MMM dd',
                                month: 'MMM yyyy'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price ($)'
                        },
                        position: 'right',
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const dataPoint = context[0].raw;
                                return new Date(dataPoint.x).toLocaleDateString();
                            },
                            label: function(context) {
                                const dataPoint = context.raw;
                                return [
                                    `Open: $${dataPoint.o.toFixed(2)}`,
                                    `High: $${dataPoint.h.toFixed(2)}`,
                                    `Low: $${dataPoint.l.toFixed(2)}`,
                                    `Close: $${dataPoint.c.toFixed(2)}`,
                                    `Volume: ${dataPoint.v.toLocaleString()}`
                                ];
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                            modifierKey: null,
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true,
                            },
                            mode: 'x',
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            },
            plugins: [candlestickPlugin]
        });
    }

    async loadChartData() {
        if (!this.currentSymbol) return;

        const loadingDiv = document.getElementById('kline-loading');
        loadingDiv.style.display = 'block';

        try {
            const response = await fetch(`/api/historical-data/${this.currentSymbol}?timeframe=${this.currentTimeframe}&limit=0`);
            const data = await response.json();
            console.log('K-line API data:', data); // DEBUG

            if (data.length === 0) {
                this.showNoDataMessage();
                return;
            }

            // Convert to candlestick format
            const chartData = data.map(candle => ({
                x: new Date(candle.timestamp), // Date object for time scale
                y: candle.close,               // dummy y so Chart.js uses the point
                o: candle.open,
                h: candle.high,
                l: candle.low,
                c: candle.close,
                v: candle.volume
            }));
            console.log('K-line chartData:', chartData); // DEBUG

            // Dynamically set y-axis min/max
            const allPrices = chartData.flatMap(d => [d.o, d.h, d.l, d.c]);
            const minPrice = Math.min(...allPrices);
            const maxPrice = Math.max(...allPrices);
            const padding = (maxPrice - minPrice) * 0.05 || 1;
            this.chart.options.scales.y.min = minPrice - padding;
            this.chart.options.scales.y.max = maxPrice + padding;

            // Update chart
            this.chart.data.datasets[0].data = chartData;
            this.chart.data.datasets[0].label = `${this.currentSymbol} (${this.currentTimeframe})`;
            // Force Chart.js to recalculate x scale
            this.chart.options.scales.x.min = undefined;
            this.chart.options.scales.x.max = undefined;
            this.chart.update();
            console.log('K-line chart updated:', this.chart.data.datasets[0].data); // DEBUG

            // Update chart title
            this.updateChartTitle();

        } catch (error) {
            console.error('Error loading chart data:', error);
            this.showErrorMessage('Failed to load chart data');
        } finally {
            loadingDiv.style.display = 'none';
        }
    }

    updateChartTitle() {
        const cardHeader = document.querySelector('.card-header h5');
        if (cardHeader) {
            cardHeader.innerHTML = `<i class="fas fa-chart-bar"></i> K-Line Chart - ${this.currentSymbol} (${this.currentTimeframe})`;
        }
    }

    showNoDataMessage() {
        if (this.chart) {
            this.chart.data.datasets[0].data = [];
            this.chart.update();
        }
        
        // Show message in chart area
        const container = document.getElementById('kline-chart-container');
        const message = document.createElement('div');
        message.className = 'position-absolute top-50 start-50 translate-middle text-muted';
        message.innerHTML = '<i class="fas fa-info-circle"></i> No data available for this symbol/timeframe';
        container.appendChild(message);
    }

    showErrorMessage(message) {
        if (this.chart) {
            this.chart.data.datasets[0].data = [];
            this.chart.update();
        }
        
        // Show error message in chart area
        const container = document.getElementById('kline-chart-container');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'position-absolute top-50 start-50 translate-middle text-danger';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        container.appendChild(errorDiv);
    }

    // Public method to load a specific symbol
    loadSymbol(symbol) {
        this.currentSymbol = symbol.toUpperCase();
        const symbolInput = document.getElementById('kline-symbol-input');
        const symbolSelect = document.getElementById('kline-symbol-select');
        
        symbolInput.value = this.currentSymbol;
        symbolSelect.value = this.currentSymbol;
        
        this.loadChartData();
    }

    // Public method to change timeframe
    changeTimeframe(timeframe) {
        this.currentTimeframe = timeframe;
        const radioBtn = document.getElementById(`timeframe-${timeframe.toLowerCase()}`);
        if (radioBtn) {
            radioBtn.checked = true;
        }
        
        if (this.currentSymbol) {
            this.loadChartData();
        }
    }

    setupResetZoomButton() {
        // Add a reset zoom button if not already present
        let btn = document.getElementById('kline-reset-zoom-btn');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'kline-reset-zoom-btn';
            btn.className = 'btn btn-outline-secondary btn-sm ms-2';
            btn.innerHTML = '<i class="fas fa-search-minus"></i> Reset Zoom';
            btn.style.display = 'inline-block';
            const header = document.querySelector('.card-header.d-flex');
            if (header) header.appendChild(btn);
        }
        btn.onclick = () => {
            if (this.chart) {
                this.chart.resetZoom && this.chart.resetZoom();
            }
        };
    }

    setupHorizontalWheelPan() {
        // Pan the chart horizontally when the horizontal wheel is used
        const container = document.getElementById('kline-chart-container');
        container.addEventListener('wheel', (event) => {
            if (event.deltaX !== 0 && this.chart) {
                // Pan by a factor proportional to deltaX
                const panAmount = event.deltaX * 5; // Adjust sensitivity as needed
                this.chart.pan({ x: panAmount }, undefined, 'default');
                event.preventDefault();
            }
        }, { passive: false });
    }
}

// Initialize K-line chart when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.klineChart = new KLineChart();
});

// Export for global access
window.KLineChart = KLineChart; 