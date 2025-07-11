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
} else if (typeof ChartZoom !== 'undefined') {
    Chart.register(ChartZoom);
    console.log('Registered ChartZoom from global ChartZoom');
} else {
    console.warn('ChartZoom plugin not found on window');
}

// ---- Global tooltip safeguard to avoid toFixed on undefined ----
if (Chart && Chart.defaults && Chart.defaults.plugins && Chart.defaults.plugins.tooltip) {
    const origLabel = Chart.defaults.plugins.tooltip.callbacks.label;
    Chart.defaults.plugins.tooltip.callbacks.label = function(ctx) {
        try {
            if (ctx.dataset && ctx.dataset.id === 'trades') {
                // use the custom marker label (origLabel is our scatter callback)
                return typeof origLabel === 'function' ? origLabel(ctx) : '';
            }
            const p = ctx.raw;
            if (!p || typeof p.y !== 'number' || !isFinite(p.y)) return '';
            return Number(p.y).toFixed(2);
        } catch(e) {
            return '';
        }
    };
}

// Fallback pan method for when zoom plugin is not available
Chart.prototype.fallbackPan = function(deltaX) {
    const xScale = this.scales.x;
    if (xScale && xScale.min !== undefined && xScale.max !== undefined) {
        const range = xScale.max - xScale.min;
        const panAmount = (deltaX / this.width) * range;
        xScale.min += panAmount;
        xScale.max += panAmount;
        this.update('none');
    }
};

class KLineChart {
    constructor() {
        this.chart = null;
        this.volumeChart = null;
        this.currentSymbol = null;
        this.currentTimeframe = '1D';
        this.availableSymbols = [];
        this.isDragging = false;
        this.lastX = 0;
        this.crosshair = { x: null, y: null, active: false };
        this.priceAdjustmentMethod = null;
        // Use the same strategies list defined for back-test (global ALL_STRATEGIES)
        this.strategyList = Array.isArray(window.ALL_STRATEGIES) && window.ALL_STRATEGIES.length > 0
            ? window.ALL_STRATEGIES.slice() // clone
            : [
                'mean_reversion',
                'simple_ma',
                'advanced',
                'low_volume',
                'momentum_breakout',
                'volatility_mean_reversion',
                'gap_trading',
                'multi_timeframe',
                'risk_managed',
                'aggressive_mean_reversion',
                'enhanced_momentum',
                'multi_timeframe_momentum'
            ];
        this.currentStrategy = localStorage.getItem('kline_last_strategy') || null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.injectStrategyDropdown();
        this.loadAvailableSymbols();
        this.setupChart();
        this.setupMouseDragPan(); // Enable mouse drag panning
    }

    setupEventListeners() {
        // Symbol input and load button
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
            const symbol = symbolInput.value.trim().toUpperCase();
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
    }

    injectStrategyDropdown() {
        const container = document.getElementById('kline-controls') || document.body;
        let select = document.createElement('select');
        select.id = 'kline-strategy-select';
        select.className = 'form-select form-select-sm';
        let defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.text = 'Show Trades (strategy…)';
        select.appendChild(defaultOpt);
        this.strategyList.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            const display = (window.STRATEGY_NAMES && window.STRATEGY_NAMES[s]) ? window.STRATEGY_NAMES[s] : s.replace(/_/g,' ');
            opt.text = display;
            select.appendChild(opt);
        });
        // Preselect if saved
        if (this.currentStrategy) {
            select.value = this.currentStrategy;
        }
        select.addEventListener('change', () => {
            const val = select.value;
            if (val) {
                this.currentStrategy = val;
                localStorage.setItem('kline_last_strategy', val);
                this.loadTradesForStrategy();
            } else {
                this.currentStrategy = null;
                localStorage.removeItem('kline_last_strategy');
                this.removeTradeMarkers();
            }
        });
        container.prepend(select);
    }

    removeTradeMarkers() {
        // Remove any dataset with id === 'trades' or 'tradeLines'
        if (!this.chart) return;
        this.chart.data.datasets = this.chart.data.datasets.filter(ds => ds.id !== 'trades' && ds.id !== 'tradeLines');
        this.chart.update();
    }

    async loadTradesForStrategy() {
        if (!this.currentSymbol || !this.currentStrategy || !this.chart) return;
        // Determine visible date range
        const xScale = this.chart.scales.x;
        const startDate = new Date(xScale.min).toISOString().slice(0,10);
        const endDate   = new Date(xScale.max).toISOString().slice(0,10);
        const adjustMethod = this.priceAdjustmentMethod || 'backward';
        const url = `/api/backtest?symbol=${this.currentSymbol}&strategy=${this.currentStrategy}&start=${startDate}&end=${endDate}&adjust_method=${adjustMethod}`;
        try {
            const resp = await fetch(url);
            const data = await resp.json();
            if (!data.trades || data.trades.length === 0) {
                this.removeTradeMarkers();
                return;
            }
            const points = [];
            // Map candles by timestamp (ms) for quick lookup
            const candleMap = {};
            if (this.chart && this.chart.data && this.chart.data.datasets.length) {
                this.chart.data.datasets[0].data.forEach(c => { candleMap[c.x] = c; });
            }

            const tsToMs = d => new Date(d).getTime();

            data.trades.forEach(tr => {
                const entryTs = tsToMs(tr.entry_time);
                const entryCandle = candleMap[entryTs];
                const entryY = entryCandle ? entryCandle.l : tr.entry_price;
                points.push({ x: entryTs, y: entryY, _type: 'entry', _trade: tr, _candle: entryCandle });

                if (tr.exit_time && tr.exit_price !== null) {
                    const exitTs = tsToMs(tr.exit_time);
                    const exitCandle = candleMap[exitTs];
                    const exitY = exitCandle ? exitCandle.h : tr.exit_price;
                    points.push({ x: exitTs, y: exitY, _type: 'exit', _trade: tr, _candle: exitCandle });
                }
            });
            // Configure styles per point
            const bgColors = points.map(p => p._type === 'entry' ? 'lime' : 'red');
            const shapes   = points.map(p => p._type === 'entry' ? 'triangle' : 'rectRot');
            // Remove old
            this.removeTradeMarkers();
            // Add scatter dataset
            this.chart.data.datasets.push({
                id: 'trades',
                type: 'scatter',
                label: `Trades ${this.currentStrategy}`,
                data: points,
                backgroundColor: bgColors,
                pointRadius: 6,
                pointStyle: shapes,
                parsing: false,
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const p = ctx.raw;
                            if (!p || typeof p.y !== 'number' || !isFinite(p.y)) return '';
                            const candle = p._candle;
                            const trade  = p._trade;
                            const dir = p._type === 'entry' ? 'Entry' : 'Exit';
                            const lines = [];
                            if (candle) {
                                const fmt=(v)=>v!==undefined?`$${v.toFixed(2)}`:'—';
                                lines.push(`Open:  ${fmt(candle.o)}`);
                                lines.push(`High:  ${fmt(candle.h)}`);
                                lines.push(`Low:   ${fmt(candle.l)}`);
                                lines.push(`Close: ${fmt(candle.c)}`);
                            }
                            let base = `${dir} @ $${p.y.toFixed(2)}`;
                            if (trade && typeof trade.return_pct === 'number' && isFinite(trade.return_pct)) {
                                base += `  P/L ${trade.return_pct.toFixed(1)}%`;
                            }
                            lines.push(base);
                            return lines;
                        }
                    }
                },
                yAxisID: 'y',
                order: 0
            });

            // create separate small line dataset per trade
            data.trades.forEach(tr=>{
                if (tr.exit_time && tr.exit_price !== null){
                    const entryTs = tsToMs(tr.entry_time);
                    const exitTs  = tsToMs(tr.exit_time);
                    const entryC = candleMap[entryTs];
                    const exitC  = candleMap[exitTs];
                    const entryY = entryC ? entryC.l : tr.entry_price;
                    const exitY  = exitC  ? exitC.h  : tr.exit_price;
                    this.chart.data.datasets.push({
                        id:'tradeLines',
                        type:'line',
                        data:[{x:entryTs,y:entryY},{x:exitTs,y:exitY}],
                        borderColor:'#888',
                        borderWidth:1,
                        pointRadius:0,
                        yAxisID:'y',
                        order:-1,
                        spanGaps:false
                    });
                }
            });

            this.chart.update();
        } catch(err) {
            console.error('Failed to load trades', err);
        }
    }

    async loadAvailableSymbols() {
        try {
            const response = await fetch('/api/available-symbols');
            const symbols = await response.json();
            this.availableSymbols = symbols;
            // No dropdown to update, so nothing else to do here
        } catch (error) {
            console.error('Error loading available symbols:', error);
        }
    }

    setupChart() {
        const ctx = document.getElementById('kline-chart').getContext('2d');
        
        // Check if zoom plugin is available
        console.log('Setting up charts...');
        console.log('ChartZoom available:', !!window.ChartZoom);
        console.log('Chart.registry.plugins:', Chart.registry ? Object.keys(Chart.registry.plugins.items) : 'No registry');
        
        // Custom candlestick plugin
        const candlestickPlugin = {
            id: 'candlestick',
            afterDraw: (chart) => {
                const { ctx, data, scales } = chart;
                const chartArea = chart.chartArea;
                const dataset = data.datasets[0];
                
                if (!dataset.data || dataset.data.length === 0) return;
                
                ctx.save();
                // Do NOT clearRect here!
                
                const xScale = chart.scales.x;
                // Determine body width from first two visible candles
                let baseWidth = 8;
                const vis = dataset.data.filter(p => p && p.x >= xScale.min && p.x <= xScale.max);
                if (vis.length > 1) {
                    const px0 = xScale.getPixelForValue(vis[0].x);
                    const px1 = xScale.getPixelForValue(vis[1].x);
                    baseWidth = Math.abs(px1 - px0) * 0.7; // 70% of gap
                }
                baseWidth = Math.max(1, Math.min(40, baseWidth));
                
                dataset.data.forEach((point, index) => {
                    if (!point || point.x === undefined) return; // safeguard
                    const x = xScale.getPixelForValue(point.x);
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
                    
                    const bodyWidth = baseWidth;
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

        // Auto-adjust y-axis and percent plugin
        const autoAdjustYAxisPlugin = {
            id: 'autoAdjustYAxis',
            afterUpdate: (chart) => {
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                const data = chart.data.datasets[0].data;
                if (!xScale || !yScale || !data || data.length === 0) return;
                const minX = xScale.min;
                const maxX = xScale.max;
                const visible = data.filter(d => d.x >= minX && d.x <= maxX);
                if (visible.length === 0) return;
                const allPrices = visible.flatMap(d => [d.o, d.h, d.l, d.c]);
                const minPrice = Math.min(...allPrices);
                const maxPrice = Math.max(...allPrices);
                const padding = (maxPrice - minPrice) * 0.05 || 1;
                chart.options.scales.y.min = minPrice - padding;
                chart.options.scales.y.max = maxPrice + padding;

                // Recalculate percent change for visible candles
                const base = visible[0].o;
                data.forEach(d => {
                    if (d.x >= minX && d.x <= maxX) {
                        d.percent = ((d.c - base) / base) * 100;
                    } else {
                        d.percent = undefined;
                    }
                });
                // Update the right y-axis (percent) scale
                const visiblePercents = visible.map(d => d.percent).filter(p => p !== undefined);
                if (visiblePercents.length > 0) {
                    const minPercent = Math.min(...visiblePercents);
                    const maxPercent = Math.max(...visiblePercents);
                    chart.options.scales.yPercent.min = minPercent - 5;
                    chart.options.scales.yPercent.max = maxPercent + 5;
                }
            }
        };

        // Custom crosshair plugin for main chart
        const crosshairPlugin = {
            id: 'crosshair',
            afterDraw: (chart) => {
                const ctx = chart.ctx;
                const chartArea = chart.chartArea;
                // Use the stored mouse position for the crosshair
                const crosshair = this.crosshair;
                if (!crosshair.active || crosshair.x === null || crosshair.y === null) return;
                const x = crosshair.x;
                const y = crosshair.y;
                // Find the nearest data point for the date label
                let d = {};
                let minDist = Infinity;
                let nearest = null;
                const data = chart.data.datasets[0].data;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                if (data && data.length > 0 && xScale) {
                    data.forEach(point => {
                        const px = xScale.getPixelForValue(point.x);
                        const dist = Math.abs(px - x);
                        if (dist < minDist) {
                            minDist = dist;
                            nearest = point;
                        }
                    });
                    if (nearest) d = nearest;
                }
                // Calculate price at cursor y
                let priceAtCursor = '-';
                let percentAtCursor = '-';
                if (yScale) {
                    const price = yScale.getValueForPixel(y);
                    if (typeof price === 'number' && isFinite(price)) {
                        // Dynamically adjust decimal precision so very small prices are not rounded to 0.00
                        const absPrice = Math.abs(price);
                        let decimals = 2;
                        if (absPrice < 1) {
                            decimals = absPrice >= 0.1 ? 3 : absPrice >= 0.01 ? 4 : 6;
                        }
                        priceAtCursor = price.toFixed(decimals);
                        // Find first visible candle for percent calculation
                        let base = null;
                        if (data && data.length > 0 && xScale) {
                            const minX = xScale.min;
                            const maxX = xScale.max;
                            const visible = data.filter(d => d.x >= minX && d.x <= maxX);
                            if (visible.length > 0) {
                                base = visible[0].o;
                            }
                        }
                        if (base) {
                            percentAtCursor = (((price - base) / base) * 100).toFixed(2) + '%';
                        }
                    }
                }
                // Draw vertical and horizontal lines
                ctx.save();
                ctx.strokeStyle = 'rgba(180,180,180,0.7)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x, chartArea.top);
                ctx.lineTo(x, chartArea.bottom);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y);
                ctx.lineTo(chartArea.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
                // Draw price label (left)
                ctx.font = 'bold 16px sans-serif';
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'left';
                ctx.fillStyle = 'red';
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 3;
                const priceLabel = priceAtCursor;
                const priceY = y;
                const priceX = chartArea.left;
                ctx.strokeText(priceLabel, priceX + 20, priceY);
                ctx.fillText(priceLabel, priceX + 20, priceY);
                // Draw percent label (right)
                ctx.textAlign = 'right';
                const percentLabel = percentAtCursor;
                const percentX = chartArea.right;
                ctx.strokeText(percentLabel, percentX - 20, priceY);
                ctx.fillText(percentLabel, percentX - 20, priceY);
            }
        };

        // Custom crosshair plugin for volume chart
        const volumeCrosshairPlugin = {
            id: 'volumeCrosshair',
            afterDraw: (chart) => {
                const ctx = chart.ctx;
                const chartArea = chart.chartArea;
                // Use the stored mouse position for the crosshair
                const crosshair = this.crosshair;
                if (!crosshair.active || crosshair.volumeX === null || crosshair.volumeY === null) return;
                const x = crosshair.volumeX;
                const y = crosshair.volumeY;
                
                // Find the nearest data point for volume value
                let volumeValue = '-';
                const data = chart.data.datasets[0].data;
                const xScale = chart.scales.x;
                const yScale = chart.scales.yVolumeLeft;
                
                if (data && data.length > 0 && xScale && yScale) {
                    let minDist = Infinity;
                    let nearest = null;
                    data.forEach(point => {
                        const px = xScale.getPixelForValue(point.x);
                        const dist = Math.abs(px - x);
                        if (dist < minDist) {
                            minDist = dist;
                            nearest = point;
                        }
                    });
                    if (nearest && nearest.v) {
                        volumeValue = nearest.v.toLocaleString();
                    }
                }
                
                // Draw vertical and horizontal lines
                ctx.save();
                ctx.strokeStyle = 'rgba(180,180,180,0.7)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x, chartArea.top);
                ctx.lineTo(x, chartArea.bottom);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y);
                ctx.lineTo(chartArea.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
                
                // Draw volume label (left)
                ctx.font = 'bold 16px sans-serif';
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'left';
                ctx.fillStyle = 'blue';
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 3;
                const volumeLabel = volumeValue;
                const volumeY = y;
                const volumeX = chartArea.left;
                ctx.strokeText(volumeLabel, volumeX + 20, volumeY);
                ctx.fillText(volumeLabel, volumeX + 20, volumeY);
                
                // Draw date label (bottom)
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillStyle = 'red';
                let dateLabel = '-';
                if (data && data.length > 0 && xScale) {
                    let minDist = Infinity;
                    let nearest = null;
                    data.forEach(point => {
                        const px = xScale.getPixelForValue(point.x);
                        const dist = Math.abs(px - x);
                        if (dist < minDist) {
                            minDist = dist;
                            nearest = point;
                        }
                    });
                    if (nearest) {
                        dateLabel = nearest.dateStr || KLineChart.formatDateUTC(new Date(nearest.x));
                    }
                }
                const dateY = chartArea.bottom;
                ctx.strokeText(dateLabel, x, dateY + 20);
                ctx.fillText(dateLabel, x, dateY + 20);
            }
        };

        // Keep right volume y-axis synced with the left one so both show identical scale
        const mirrorVolumeYScale = {
            id: 'mirrorVolumeY',
            afterUpdate(chart) {
                const left = chart.scales.yVolumeLeft;
                const right = chart.scales.yVolumeRight;
                if (left && right) {
                    right.options.min = left.min;
                    right.options.max = left.max;
                    // ensure no grid lines drawn on mirrored axis
                    right.options.grid = { drawOnChartArea: false };
                }
            }
        };

        // Dynamically set volume bar thickness to match candle width for perfect alignment
        const autoVolumeBarWidth = {
            id: 'autoVolBarWidth',
            afterUpdate(chart) {
                const ds = chart.data.datasets[0];
                if (!ds || !ds.data || ds.data.length < 2) return;
                const xScale = chart.scales.x;
                
                // Find first two visible data points to calculate spacing
                const minX = xScale.min;
                const maxX = xScale.max;
                const visible = ds.data.filter(d => d && d.x >= minX && d.x <= maxX);
                
                if (visible.length < 2) return;
                
                const first = visible[0].x;
                const second = visible[1].x;
                if (!first || !second) return;
                
                const pixelSpacing = Math.abs(xScale.getPixelForValue(second) - xScale.getPixelForValue(first));
                // Set bar thickness to 70% of spacing to match candlestick width
                const barWidth = Math.max(1, Math.floor(pixelSpacing * 0.7));
                
                // Apply the calculated width
                ds.barThickness = barWidth;
                console.log('Volume bar width set to:', barWidth, 'pixels (spacing:', pixelSpacing, ')');
            }
        };

        // Robust sync: volume chart always follows main chart (time scale)
        const syncVolumeChartX = (chart) => {
            console.log('syncVolumeChartX called');
            if (!this.volumeChart) {
                console.warn('Volume chart not available for sync');
                return;
            }
            const xScale = chart.scales.x;
            this.volumeChart.options.scales.x.min = xScale.min;
            this.volumeChart.options.scales.x.max = xScale.max;
            
            // Force the volume chart to recalculate bar widths after sync
            setTimeout(() => {
                this.volumeChart.update('none');
                console.log('Volume chart x-axis synced and updated');
            }, 0);
        };
        
        console.log('Setting up main chart with zoom handlers');
        // Main chart
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Price',
                        data: [],
                        borderColor: 'transparent',
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        fill: false,
                        yAxisID: 'y',
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: 0
                },
                interaction: {
                    intersect: false,
                    mode: 'nearest'
                },
                scales: {
                    x: {
                        type: 'time',
                        bounds: 'data',
                        offset: false,
                        grid: {
                            drawOnChartArea: false,
                            drawTicks: false
                        },
                        time: {
                            unit: 'day',
                            minUnit: 'day',
                            displayFormats: {
                                day: 'MMM dd',
                                week: 'MMM dd',
                                month: 'MMM yyyy'
                            }
                        },
                        display: false,
                        title: {
                            display: false,
                            text: 'Date'
                        },
                        ticks: {
                            source: 'data',
                            maxTicksLimit: 10,
                            callback: function(value) {
                                const d = new Date(value);
                                return isFinite(d) ? d.toLocaleDateString('en-US', { month: 'short' }) : '';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price ($)'
                        },
                        position: 'left',
                        min: 0,
                        max: 1,
                        weight: 2,
                        area: 'main',
                        grid: {
                            drawOnChartArea: true
                        },
                        ticks: {
                            maxTicksLimit: 8,
                            padding: 8,
                            callback: function(value) {
                                const absVal = Math.abs(value);
                                if (absVal >= 1) return value.toFixed(2);
                                if (absVal >= 0.1) return value.toFixed(3);
                                if (absVal >= 0.01) return value.toFixed(4);
                                return value.toFixed(6);
                            }
                        }
                    },
                    yPercent: {
                        position: 'right',
                        title: {
                            display: true,
                            text: '% Change'
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(2) + '%';
                            }
                        },
                        area: 'main'
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
                                return dataPoint.dateStr || KLineChart.formatDateUTC(new Date(dataPoint.x));
                            },
                            label: function(context) {
                                const dataPoint = context.raw;
                                if (dataPoint.o === undefined) return ''; // skip non-candle items
                                const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : '—';
                                return [
                                    `Open:  $${fmt(dataPoint.o)}`,
                                    `High:  $${fmt(dataPoint.h)}`,
                                    `Low:   $${fmt(dataPoint.l)}`,
                                    `Close: $${fmt(dataPoint.c)}`,
                                    `Change: ${typeof dataPoint.percent === 'number' && isFinite(dataPoint.percent) ? dataPoint.percent.toFixed(2) + '%' : ''}`,
                                    `Volume: ${dataPoint.v ? dataPoint.v.toLocaleString() : ''}`
                                ];
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                            modifierKey: null,
                            threshold: 10,
                            animation: {
                                duration: 300,
                                easing: 'easeOutCubic'
                            }
                        },
                        zoom: {
                            wheel: {
                                enabled: false
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                            drag: {
                                enabled: false
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            },
            plugins: [candlestickPlugin, autoAdjustYAxisPlugin, crosshairPlugin]
        });
        
        console.log('Main chart created, zoom plugin enabled:', !!this.chart.options.plugins.zoom);

        // Initialize the volume chart below
        const vctx = document.getElementById('volume-chart').getContext('2d');
        this.volumeChart = new Chart(vctx, {
            type: 'bar',
            data: {
                datasets: [
                    {
                        label: 'Volume',
                        data: [],
                        backgroundColor: 'rgba(60, 120, 216, 0.3)',
                        borderColor: 'rgba(60, 120, 216, 0.7)',
                        yAxisID: 'yVolumeLeft',
                        order: 1,
                        barPercentage: 0.8,
                        categoryPercentage: 0.9,
                        barThickness: 'flex',
                        borderWidth: 0,
                        maxBarThickness: 40
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: 0
                },
                scales: {
                    x: {
                        type: 'time',
                        bounds: 'data',
                        offset: false,
                        grid: {
                            drawOnChartArea: false,
                            drawTicks: false
                        },
                        time: {
                            unit: 'day',
                            minUnit: 'day',
                            displayFormats: {
                                day: 'MMM dd',
                                week: 'MMM dd',
                                month: 'MMM yyyy'
                            }
                        },
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: {
                            source: 'data',
                            maxTicksLimit: 10,
                            callback: function(value, index) {
                                const d = new Date(value);
                                if (!isFinite(d)) return '';
                                const m = d.toLocaleDateString('en-US', { month: 'short' });
                                const y = d.getFullYear();
                                return index === 0 || d.getMonth() === 0 ? `${m} ${y}` : m;
                            }
                        }
                    },
                    yVolumeLeft: {
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Volume'
                        },
                        beginAtZero: true,
                        display: true,
                        min: 0,
                        grid: {
                            drawOnChartArea: true
                        },
                        ticks: {
                            maxTicksLimit: 8,
                            padding: 8,
                            callback: function(value, index, ticks) {
                                // Determine the unit based on the maximum value
                                const maxValue = Math.max(...ticks.map(tick => tick.value));
                                let unit = '';
                                let divisor = 1;
                                
                                if (maxValue >= 1000000000) {
                                    unit = 'B';
                                    divisor = 1000000000;
                                } else if (maxValue >= 1000000) {
                                    unit = 'M';
                                    divisor = 1000000;
                                } else if (maxValue >= 1000) {
                                    unit = 'K';
                                    divisor = 1000;
                                }
                                
                                // Show unit only on the first (bottom) tick
                                if (index === 0 && unit) {
                                    return unit;
                                }
                                
                                // Show formatted number for other ticks
                                const formattedValue = (value / divisor).toFixed(2);
                                return formattedValue;
                            }
                        }
                    },
                    yVolumeRight: {
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Volume'
                        },
                        beginAtZero: true,
                        display: true,
                        min: 0,
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            maxTicksLimit: 8,
                            padding: 8,
                            callback: function(value, index, ticks) {
                                // Determine the unit based on the maximum value
                                const maxValue = Math.max(...ticks.map(tick => tick.value));
                                let unit = '';
                                let divisor = 1;
                                
                                if (maxValue >= 1000000000) {
                                    unit = 'B';
                                    divisor = 1000000000;
                                } else if (maxValue >= 1000000) {
                                    unit = 'M';
                                    divisor = 1000000;
                                } else if (maxValue >= 1000) {
                                    unit = 'K';
                                    divisor = 1000;
                                }
                                
                                // Show unit only on the first (bottom) tick
                                if (index === 0 && unit) {
                                    return unit;
                                }
                                
                                // Show formatted number for other ticks
                                const formattedValue = (value / divisor).toFixed(2);
                                return formattedValue;
                            }
                        }
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
                                return dataPoint.dateStr || KLineChart.formatDateUTC(new Date(dataPoint.x));
                            },
                            label: function(context) {
                                const dataPoint = context.raw;
                                return `Volume: ${dataPoint.v ? dataPoint.v.toLocaleString() : context.parsed.y.toLocaleString()}`;
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: false
                        },
                        zoom: {
                            wheel: {
                                enabled: false
                            },
                            pinch: {
                                enabled: false
                            },
                            mode: 'x',
                            drag: {
                                enabled: false
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            },
            plugins: [volumeCrosshairPlugin, mirrorVolumeYScale, autoVolumeBarWidth]
        });

        // Ensure both canvases have the same width on resize and load
        const syncChartWidths = () => {
            const mainCanvas = document.getElementById('kline-chart');
            const volumeCanvas = document.getElementById('volume-chart');
            const mainContainer = document.getElementById('kline-chart-container');
            const volumeContainer = document.getElementById('volume-chart-container');
            
            if (mainCanvas && volumeCanvas && mainContainer && volumeContainer) {
                // Force identical bitmap width for canvases to keep scales in perfect sync
                const pixelWidth = mainCanvas.width; // actual render width in pixels
                mainCanvas.style.width  = '100%';
                volumeCanvas.style.width = '100%';
                if (volumeCanvas.width !== pixelWidth) {
                    volumeCanvas.width = pixelWidth;
                }
                
                // Resize charts
                if (this.chart) this.chart.resize();
                if (this.volumeChart) this.volumeChart.resize();
            }
        };
        
        window.addEventListener('resize', syncChartWidths);
        
        // Initial sync
        setTimeout(syncChartWidths, 100);

        // If no data, set x-axis to show the most recent year ending today
        if (this.chart.data.datasets[0].data.length === 0) {
            const today = new Date();
            const minDate = new Date(today);
            minDate.setFullYear(today.getFullYear() - 1);
            this.chart.options.scales.x.min = minDate;
            this.chart.options.scales.x.max = today;
            this.chart.update();
        }

        // Add custom wheel handler with manual sync
        const canvas = ctx.canvas;
        canvas.addEventListener('wheel', (event) => {
            if (!this.chart) return;
            if (event.deltaX !== 0) {
                // Side wheel: pan
                const panAmount = event.deltaX * 5;
                this.chart.pan({ x: panAmount }, undefined, 'default');
                console.log('Manual pan triggered, syncing volume chart');
                syncVolumeChartX(this.chart);
                event.preventDefault();
            } else if (event.deltaY !== 0) {
                // Main wheel: zoom
                if (this.chart.zoom) {
                    this.chart.zoom({ x: event.deltaY < 0 ? 1.1 : 0.9 });
                    console.log('Manual zoom triggered, syncing volume chart');
                    syncVolumeChartX(this.chart);
                    event.preventDefault();
                }
            }
        }, { passive: false });
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate corresponding x position on volume chart
            const volumeCanvas = document.getElementById('volume-chart');
            let volumeX = x;
            let volumeY = null;
            
            if (volumeCanvas && this.volumeChart) {
                const volumeRect = volumeCanvas.getBoundingClientRect();
                const volumeChartArea = this.volumeChart.chartArea;
                if (volumeChartArea) {
                    volumeY = volumeChartArea.top + (volumeChartArea.bottom - volumeChartArea.top) / 2; // Middle of volume chart
                }
            }
            
            this.crosshair = { x, y, volumeX, volumeY, active: true };
            if (this.chart) this.chart.draw();
            if (this.volumeChart) this.volumeChart.draw();
        });
        canvas.addEventListener('mouseleave', () => {
            this.crosshair = { x: null, y: null, volumeX: null, volumeY: null, active: false };
            if (this.chart) this.chart.draw();
            if (this.volumeChart) this.volumeChart.draw();
        });

        // Add mouse events to volume chart
        const volumeCanvas = document.getElementById('volume-chart');
        volumeCanvas.addEventListener('mousemove', (e) => {
            const rect = volumeCanvas.getBoundingClientRect();
            const volumeX = e.clientX - rect.left;
            const volumeY = e.clientY - rect.top;
            
            // Calculate corresponding x position on main chart
            const mainCanvas = document.getElementById('kline-chart');
            let x = volumeX;
            let y = null;
            
            if (mainCanvas && this.chart) {
                const mainRect = mainCanvas.getBoundingClientRect();
                const mainChartArea = this.chart.chartArea;
                if (mainChartArea) {
                    y = mainChartArea.top + (mainChartArea.bottom - mainChartArea.top) / 2; // Middle of main chart
                }
            }
            
            this.crosshair = { x, y, volumeX, volumeY, active: true };
            if (this.chart) this.chart.draw();
            if (this.volumeChart) this.volumeChart.draw();
        });
        volumeCanvas.addEventListener('mouseleave', () => {
            this.crosshair = { x: null, y: null, volumeX: null, volumeY: null, active: false };
            if (this.chart) this.chart.draw();
            if (this.volumeChart) this.volumeChart.draw();
        });

        // Add context menu for price adjustment
        this.createPriceAdjustmentMenu(canvas);

        // After global tooltip override, add filter
        Chart.defaults.plugins.tooltip.filter = function(ctx) {
            // Exclude items whose parsed.y is not a finite number
            if (typeof ctx.parsed.y !== 'number' || !isFinite(ctx.parsed.y)) return false;
            // When hovering markers ensure only one item shown: allow trades dataset; if dataset id isn't 'trades' ensure parsed object has o property (candlestick)
            return true;
        };
    }

    createPriceAdjustmentMenu(canvas) {
        // Create main menu element
        let menu = document.createElement('div');
        menu.id = 'price-adjustment-menu';
        menu.style.position = 'absolute';
        menu.style.display = 'none';
        menu.style.zIndex = 1000;
        menu.style.background = '#23272e';
        menu.style.border = '1px solid #444';
        menu.style.borderRadius = '6px';
        menu.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
        menu.style.padding = '4px 0';
        menu.style.minWidth = '180px';
        menu.style.fontFamily = 'inherit';
        menu.style.color = '#fff';
        menu.innerHTML = `
            <div class="menu-item has-submenu" style="padding: 8px 16px; cursor: pointer; position: relative;">
                Price Adjustment <span style="float:right;">&#9654;</span>
                <div class="submenu" style="display:none; position:absolute; left:100%; top:0; background:#23272e; border:1px solid #444; border-radius:6px; min-width:180px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                    <div class="menu-item" data-method="backward" style="padding: 8px 16px; cursor: pointer;">Backward Adjusted</div>
                    <div class="menu-item" data-method="none" style="padding: 8px 16px; cursor: pointer;">Unadjusted</div>
                    <div class="menu-item" data-method="forward" style="padding: 8px 16px; cursor: pointer;">Forward Adjusted</div>
                </div>
            </div>
        `;
        document.body.appendChild(menu);

        // Hide menu on click elsewhere
        document.addEventListener('click', () => { menu.style.display = 'none'; });
        // Prevent default context menu and show custom menu
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            menu.style.display = 'block';
            menu.style.left = e.pageX + 'px';
            menu.style.top = e.pageY + 'px';
        });
        // Submenu show/hide logic
        const parentItem = menu.querySelector('.has-submenu');
        const submenu = parentItem.querySelector('.submenu');
        parentItem.addEventListener('mouseenter', () => {
            submenu.style.display = 'block';
        });
        parentItem.addEventListener('mouseleave', () => {
            submenu.style.display = 'none';
        });
        submenu.addEventListener('mouseenter', () => {
            submenu.style.display = 'block';
        });
        submenu.addEventListener('mouseleave', () => {
            submenu.style.display = 'none';
        });
        // Handle submenu item click
        submenu.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const method = item.getAttribute('data-method');
                menu.style.display = 'none';
                this.setPriceAdjustmentMethod(method);
            });
        });
    }

    setPriceAdjustmentMethod(method) {
        // Store the selected method and reload chart data
        this.priceAdjustmentMethod = method;
        this.loadChartData();
    }

    // Filter out non-trading dates (weekends and holidays)
    filterTradingDays(data) {
        return data.filter(item => {
            const date = new Date(item.x);
            const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
            
            // Remove weekends (Saturday = 6, Sunday = 0)
            if (dayOfWeek === 0 || dayOfWeek === 6) {
                return false;
            }
            
            // Optional: Remove common US market holidays
            const month = date.getMonth() + 1; // 1-based month
            const day = date.getDate();
            
            // New Year's Day (January 1)
            if (month === 1 && day === 1) return false;
            
            // Independence Day (July 4)
            if (month === 7 && day === 4) return false;
            
            // Christmas Day (December 25)
            if (month === 12 && day === 25) return false;
            
            // Add more holidays as needed:
            // Martin Luther King Jr. Day (3rd Monday in January)
            // Presidents Day (3rd Monday in February)
            // Good Friday (varies)
            // Memorial Day (last Monday in May)
            // Labor Day (1st Monday in September)
            // Thanksgiving (4th Thursday in November)
            
            return true; // Keep trading days
        });
    }

    async loadChartData() {
        if (!this.currentSymbol) return;
        const loadingDiv = document.getElementById('kline-loading');
        loadingDiv.style.display = 'block';

        // Add timeout protection to ensure loading spinner is always hidden
        const loadingTimeout = setTimeout(() => {
            console.warn('K-line chart loading timed out, hiding spinner');
            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
        }, 15000); // 15 second timeout

        // Helper: clear any previous overlay messages
        const clearOverlays = () => {
            const container = document.getElementById('kline-chart-container');
            if (!container) return;
            [...container.querySelectorAll('.kline-overlay-msg')].forEach(el => el.remove());
        };
        clearOverlays();
        try {
            // Fetch split-adjusted data for smooth chart display with aggressive cache-busting
            const cacheBuster = Date.now() + Math.random();
            const adjustMethod = this.priceAdjustmentMethod || 'backward';
            const response = await fetch(`/api/historical-data/${this.currentSymbol}?timeframe=${this.currentTimeframe}&limit=0&adjust_for_splits=true&adjust_method=${adjustMethod}&_t=${cacheBuster}`, {
                cache: 'no-cache',
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            });
            const data = await response.json();
            console.log('K-line API data:', data); // DEBUG
            
            // Check if response is an error
            if (!response.ok || data.detail || !Array.isArray(data)) {
                console.error('API error:', data);
                this.showErrorMessage(data.detail || 'Failed to load chart data');
                return;
            }
            
            // Debug: Check a few sample data points for split adjustment
            const feb2025Data = data.filter(d => d.timestamp && d.timestamp.startsWith('2025-02-0')).slice(0, 5);
            console.log('February 2025 sample data (should be split-adjusted):', feb2025Data);
            if (data.length === 0) {
                this.showNoDataMessage();
                return;
            }
            
            // Also fetch split events for display annotations (optional enhancement)
            const splitsResponse = await fetch(`/api/splits/${this.currentSymbol}`);
            const splits = await splitsResponse.json();
            console.log('Split events:', splits);
            
            // Sort chartData by date using real trading dates (no compression)
            const chartData = data.map(candle => {
                const dateObj = new Date(candle.timestamp); // parse ISO / epoch equally
                const epoch = dateObj.getTime();            // numeric ms since 1970-01-01
                const dateStr = dateObj.toISOString().slice(0,10); // YYYY-MM-DD
                return {
                    x: epoch,         // Chart.js time scale accepts epoch ms
                    dateStr,          // cached human-readable date
                    y: candle.close,
                    o: candle.open,
                    h: candle.high,
                    l: candle.low,
                    c: candle.close,
                    v: candle.volume
                };
            }).sort((a, b) => a.x - b.x);
            
            // Filter out non-trading dates
            const filteredChartData = this.filterTradingDays(chartData);
            console.log('Filtered out', chartData.length - filteredChartData.length, 'non-trading days');

            // Use real trading dates without compression
            const displayData = filteredChartData;

            // Calculate percent change from first visible candle
            let base = displayData[0].o;
            displayData.forEach(d => {
                d.percent = ((d.c - base) / base) * 100;
            });
            // Volume data for bar chart
            const volumeData = displayData.map(d => ({
                x: d.x,
                y: d.v,
                v: d.v,
                up: d.c >= d.o,
                dateStr: d.dateStr
            }));

            // Build colour arrays for volume bars
            const volBg = volumeData.map(d => d.up ? 'rgba(0, 200, 90, 0.4)' : 'rgba(220, 53, 69, 0.4)');
            const volBorder = volumeData.map(d => d.up ? 'rgba(0, 200, 90, 0.9)' : 'rgba(220, 53, 69, 0.9)');

            console.log('Volume data for chart:', volumeData); // DEBUG
            // Set chart data
            this.chart.data.datasets[0].data = displayData;
            this.chart.data.datasets[0].label = `${this.currentSymbol} (${this.currentTimeframe})`;
            // Adjust visible window to last year of data
            if (displayData.length > 0) {
                const maxDate = displayData[displayData.length - 1].x;
                const minDate = new Date(maxDate);
                minDate.setFullYear(minDate.getFullYear() - 1);
                this.chart.options.scales.x.min = minDate;
                this.chart.options.scales.x.max = maxDate;
                this.volumeChart.options.scales.x.min = minDate;
                this.volumeChart.options.scales.x.max = maxDate;
            }
            this.chart.update();
            // Set volume chart data
            this.volumeChart.data.datasets[0].data = volumeData;
            this.volumeChart.data.datasets[0].backgroundColor = volBg;
            this.volumeChart.data.datasets[0].borderColor = volBorder;
            this.volumeChart.update();

            // ----- Sync Strategy Comparison Back-test (non-blocking) -----
            try {
                const btSymbolInput = document.getElementById('bt-symbol');
                if (btSymbolInput) {
                    btSymbolInput.value = this.currentSymbol;
                }
                if (typeof window.runAllStrategiesBacktest === 'function') {
                    // call asynchronously so UI stays responsive and doesn't block chart loading
                    setTimeout(() => {
                        try {
                            window.runAllStrategiesBacktest();
                        } catch (btErr) {
                            console.warn('Backtest sync failed:', btErr);
                        }
                    }, 100);
                }
            } catch (syncErr) {
                console.warn('Failed to trigger back-test sync:', syncErr);
            }

            // If a strategy is already selected (cached), plot its trades automatically
            if (this.currentStrategy) {
                this.loadTradesForStrategy();
            }

            // Remove any leftover overlay now that we have data
            clearOverlays();
            setTimeout(() => {
                if (this.chart && this.chart.update) this.chart.update();
                if (this.volumeChart && this.volumeChart.update) this.volumeChart.update();
                
                // Ensure alignment after both charts are updated
                setTimeout(() => {
                    if (this.chart && this.volumeChart) {
                        const xScale = this.chart.scales.x;
                        this.volumeChart.options.scales.x.min = xScale.min;
                        this.volumeChart.options.scales.x.max = xScale.max;
                        this.volumeChart.update('none');
                        console.log('Final chart alignment sync completed');
                    }
                }, 100);
            }, 0);
            setTimeout(() => {
                const chart = this.chart;
                if (!chart) return;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                const data = chart.data.datasets[0].data;
                if (!xScale || !yScale || !data || data.length === 0) return;
                const minX = xScale.min;
                const maxX = xScale.max;
                const visible = data.filter(d => d.x >= minX && d.x <= maxX);
                if (visible.length === 0) return;
                const allPrices = visible.flatMap(d => [d.o, d.h, d.l, d.c]);
                const minPrice = Math.min(...allPrices);
                const maxPrice = Math.max(...allPrices);
                const padding = (maxPrice - minPrice) * 0.05 || 1;
                chart.options.scales.y.min = minPrice - padding;
                chart.options.scales.y.max = maxPrice + padding;
                chart.update();
                // Sync volume chart x-axis
                if (this.volumeChart) {
                    this.volumeChart.options.scales.x.min = xScale.min;
                    this.volumeChart.options.scales.x.max = xScale.max;
                    this.volumeChart.update('none');
                }
            }, 50);
            console.log('K-line chart updated with split-adjusted data for smooth display');
        } catch (error) {
            console.error('Error loading chart data:', error);
            this.showErrorMessage('Failed to load chart data');
        } finally {
            // Clear the timeout and hide loading spinner
            clearTimeout(loadingTimeout);
            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
        }
    }

    updateChartTitle() {
        const cardHeader = document.querySelector('#kline-card-header h5');
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
        message.className = 'position-absolute top-50 start-50 translate-middle text-muted kline-overlay-msg';
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
        errorDiv.className = 'position-absolute top-50 start-50 translate-middle text-danger kline-overlay-msg';
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

    setupMouseDragPan() {
        const canvas = document.getElementById('kline-chart');
        const container = document.getElementById('kline-chart-container');
        // Mouse down event
        canvas.addEventListener('mousedown', (e) => {
            // Only start dragging on left mouse button
            if (e.button === 0) {
                this.isDragging = true;
                this.lastX = e.clientX;
                canvas.style.cursor = 'grabbing'; // hand when dragging
                e.preventDefault();
            }
        });
        // Mouse move event
        canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging && this.chart) {
                const deltaX = e.clientX - this.lastX;
                const panAmount = deltaX * 1.5; // Adjust sensitivity
                // Use Chart.js zoom plugin pan method if available
                if (this.chart.pan) {
                    this.chart.pan({ x: panAmount }, undefined, 'default');
                    console.log('Manual drag pan triggered, syncing volume chart');
                    // Sync volume chart after panning
                    if (this.volumeChart) {
                        const xScale = this.chart.scales.x;
                        this.volumeChart.options.scales.x.min = xScale.min;
                        this.volumeChart.options.scales.x.max = xScale.max;
                        this.volumeChart.update('none');
                        console.log('Volume chart synced after drag pan');
                    }
                } else if (this.chart.fallbackPan) {
                    // Use fallback pan method
                    this.chart.fallbackPan(panAmount);
                    console.log('Manual fallback pan triggered, syncing volume chart');
                    // Sync volume chart after fallback panning
                    if (this.volumeChart) {
                        const xScale = this.chart.scales.x;
                        this.volumeChart.options.scales.x.min = xScale.min;
                        this.volumeChart.options.scales.x.max = xScale.max;
                        this.volumeChart.update('none');
                        console.log('Volume chart synced after fallback pan');
                    }
                }
                this.lastX = e.clientX;
                e.preventDefault();
            }
        });
        // Mouse up event
        canvas.addEventListener('mouseup', (e) => {
            if (this.isDragging) {
                this.isDragging = false;
                canvas.style.cursor = 'default'; // arrow when not dragging
                e.preventDefault();
            }
        });
        // Mouse leave event
        canvas.addEventListener('mouseleave', (e) => {
            if (this.isDragging) {
                this.isDragging = false;
                canvas.style.cursor = 'default'; // arrow when not dragging
            }
        });
        // Prevent context menu on right click
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
        // Set initial cursor style
        canvas.style.cursor = 'default';
        // Add touch support for mobile devices
        let touchStartX = 0;
        let touchStartY = 0;
        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                touchStartX = touch.clientX;
                touchStartY = touch.clientY;
                this.isDragging = true;
                e.preventDefault();
            }
        });
        canvas.addEventListener('touchmove', (e) => {
            if (this.isDragging && e.touches.length === 1 && this.chart) {
                const touch = e.touches[0];
                const deltaX = touch.clientX - touchStartX;
                const deltaY = touch.clientY - touchStartY;
                // Only pan if horizontal movement is greater than vertical
                if (Math.abs(deltaX) > Math.abs(deltaY)) {
                    const panAmount = deltaX * 2;
                    if (this.chart.pan) {
                        this.chart.pan({ x: panAmount }, undefined, 'default');
                        console.log('Manual touch pan triggered, syncing volume chart');
                        // Sync volume chart after touch panning
                        if (this.volumeChart) {
                            const xScale = this.chart.scales.x;
                            this.volumeChart.options.scales.x.min = xScale.min;
                            this.volumeChart.options.scales.x.max = xScale.max;
                            this.volumeChart.update('none');
                            console.log('Volume chart synced after touch pan');
                        }
                    } else if (this.chart.fallbackPan) {
                        // Use fallback pan method
                        this.chart.fallbackPan(panAmount);
                        console.log('Manual touch fallback pan triggered, syncing volume chart');
                        // Sync volume chart after touch fallback panning
                        if (this.volumeChart) {
                            const xScale = this.chart.scales.x;
                            this.volumeChart.options.scales.x.min = xScale.min;
                            this.volumeChart.options.scales.x.max = xScale.max;
                            this.volumeChart.update('none');
                            console.log('Volume chart synced after touch fallback pan');
                        }
                    }
                    touchStartX = touch.clientX;
                    e.preventDefault();
                }
            }
        });
        canvas.addEventListener('touchend', (e) => {
            if (this.isDragging) {
                this.isDragging = false;
            }
        });
    }

    // Helper to format a Date as YYYY-MM-DD using its UTC components to avoid timezone shifts
    static formatDateUTC(date) {
        const y = date.getUTCFullYear();
        const m = String(date.getUTCMonth() + 1).padStart(2, '0');
        const d = String(date.getUTCDate()).padStart(2, '0');
        return `${m}/${d}/${y}`; // keeps same style as default locale but without TZ influence
    }
}

// Initialize K-line chart when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.klineChart = new KLineChart();
});

// Export for global access
window.KLineChart = KLineChart; 