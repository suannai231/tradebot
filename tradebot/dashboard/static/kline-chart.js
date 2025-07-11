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
        this.priceAdjustmentMethod = 'backward';
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
        this.loadAvailableSymbols();
        this.setupChart();
        this.setupEventListeners();
        this.injectStrategyDropdown();
        
        // Periodic cleanup of stuck error messages
        setInterval(() => {
            const container = document.getElementById('kline-chart-container');
            if (container && this.chart && this.chart.data.datasets[0].data.length > 0) {
                // If we have data but still see error messages, remove them
                container.querySelectorAll('.position-absolute').forEach(el => {
                    if (el.textContent && (el.textContent.includes('Failed to load') || el.textContent.includes('Error'))) {
                        el.remove();
                    }
                });
            }
        }, 2000); // Check every 2 seconds
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
        // Use raw prices to match the chart display (no split adjustments)
        const url = `/api/backtest?symbol=${this.currentSymbol}&strategy=${this.currentStrategy}&start=${startDate}&end=${endDate}&adjust_method=none`;
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
        
        // Restore CLEAN candlestick plugin (no debug rectangle or spam)
        const candlestickPlugin = {
            id: 'candlestick',
            afterDraw: (chart) => {
                const { ctx, data } = chart;
                const chartArea = chart.chartArea;
                const dataset = data.datasets[0];
                if (!dataset || !dataset.data || dataset.data.length === 0) return;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                if (!xScale || !yScale) return;
                ctx.save();
                // Determine candle body width
                let bodyWidth = 8;
                const visible = dataset.data.filter(p => p && p.x >= xScale.min && p.x <= xScale.max);
                if (visible.length > 1) {
                    const p0 = xScale.getPixelForValue(visible[0].x);
                    const p1 = xScale.getPixelForValue(visible[1].x);
                    bodyWidth = Math.max(1, Math.min(40, Math.abs(p1 - p0) * 0.7));
                }
                // Draw each candle
                dataset.data.forEach(pt => {
                    if (!pt || typeof pt.o !== 'number') return;
                    const x = xScale.getPixelForValue(pt.x);
                    if (x < chartArea.left || x > chartArea.right) return;
                    const openY  = yScale.getPixelForValue(pt.o);
                    const closeY = yScale.getPixelForValue(pt.c);
                    const highY  = yScale.getPixelForValue(pt.h);
                    const lowY   = yScale.getPixelForValue(pt.l);
                    const isUp = pt.c >= pt.o;
                    const color = isUp ? '#26a69a' : '#ef5350';
                    // wick
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x, highY);
                    ctx.lineTo(x, lowY);
                    ctx.stroke();
                    // body
                    const bodyH = Math.max(1, Math.abs(closeY - openY));
                    const bodyY = isUp ? closeY : openY;
                    ctx.fillStyle = color;
                    ctx.fillRect(x - bodyWidth/2, bodyY, bodyWidth, bodyH);
                    ctx.strokeStyle = color;
                    ctx.strokeRect(x - bodyWidth/2, bodyY, bodyWidth, bodyH);
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


        
        console.log('Setting up combined chart with candlesticks and volume');
        // Combined chart with candlesticks and volume bars
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
                    },
                    {
                        label: 'Volume',
                        data: [],
                        type: 'bar',
                        backgroundColor: [],
                        borderColor: [],
                        borderWidth: 1,
                        yAxisID: 'yVolume',
                        order: 2,
                        barThickness: 8
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
                    },
                    yVolume: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Volume'
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            callback: function(value) {
                                if (value >= 1e9) {
                                    return (value / 1e9).toFixed(1) + 'B';
                                } else if (value >= 1e6) {
                                    return (value / 1e6).toFixed(1) + 'M';
                                } else if (value >= 1e3) {
                                    return (value / 1e3).toFixed(1) + 'K';
                                }
                                return value.toString();
                            }
                        },
                        // Position volume scale at bottom 20% of chart
                        max: function(context) {
                            const chart = context.chart;
                            const datasets = chart.data.datasets;
                            const volumeDataset = datasets.find(d => d.yAxisID === 'yVolume');
                            if (volumeDataset && volumeDataset.data.length > 0) {
                                const volumes = volumeDataset.data.map(d => d.y || 0).filter(v => v > 0);
                                if (volumes.length > 0) {
                                    const maxVolume = Math.max(...volumes);
                                    const result = maxVolume * 5; // Scale to use bottom 20% of chart
                                    console.log(`Volume scale callback: max=${result.toLocaleString()}, volumes: [${Math.min(...volumes)}, ${maxVolume}]`);
                                    return result;
                                }
                            }
                            console.log('Volume scale callback: using default 1000000');
                            return 1000000;
                        },
                        min: 0
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
        
        console.log('Combined chart created, zoom plugin enabled:', !!this.chart.options.plugins.zoom);

        // Handle chart resize
        window.addEventListener('resize', () => {
            if (this.chart) this.chart.resize();
        });

        // If no data, set x-axis to show the most recent year ending today
        if (this.chart.data.datasets[0].data.length === 0) {
            const today = new Date();
            const minDate = new Date(today);
            minDate.setFullYear(today.getFullYear() - 1);
            this.chart.options.scales.x.min = minDate;
            this.chart.options.scales.x.max = today;
            this.chart.update();
        }

        // Add custom wheel handler
        const canvas = ctx.canvas;
        canvas.addEventListener('wheel', (event) => {
            if (!this.chart) return;
            if (event.deltaX !== 0) {
                // Side wheel: pan
                const panAmount = event.deltaX * 5;
                this.chart.pan({ x: panAmount }, undefined, 'default');
                console.log('Manual pan triggered');
                event.preventDefault();
            } else if (event.deltaY !== 0) {
                // Main wheel: zoom
                if (this.chart.zoom) {
                    this.chart.zoom({ x: event.deltaY < 0 ? 1.1 : 0.9 });
                    console.log('Manual zoom triggered');
                    event.preventDefault();
                }
            }
        }, { passive: false });
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.crosshair = { x, y, active: true };
            if (this.chart) this.chart.draw();
        });
        canvas.addEventListener('mouseleave', () => {
            this.crosshair = { x: null, y: null, active: false };
            if (this.chart) this.chart.draw();
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
        
        // Setup mouse drag panning functionality
        this.setupMouseDragPan();
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
        console.log(`Price adjustment method changed to: ${method}`);
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
            
            // Also clear any error messages that might be stuck
            const errorElements = container.querySelectorAll('.position-absolute');
            errorElements.forEach(el => {
                if (el.textContent && (el.textContent.includes('Failed to load') || el.textContent.includes('No data'))) {
                    el.remove();
                }
            });
        };
        clearOverlays();
        try {
            // Determine adjustment method - use selected method or default to 'none' (unadjusted)
            const adjustmentMethod = this.priceAdjustmentMethod || 'backward';
            
            // Map adjustment methods to API parameters
            const adjustmentParams = {
                'none': 'adjust_for_splits=false',
                'backward': 'adjust_for_splits=true&adjust_method=backward',
                'forward': 'adjust_for_splits=true&adjust_method=forward'
            };
            
            const adjustmentParam = adjustmentParams[adjustmentMethod] || 'adjust_for_splits=false';
            
            const cacheBuster = Date.now() + Math.random();
            // Limit data points to prevent browser overload: 1000 for daily, 5000 for weekly/monthly
            const limit = this.currentTimeframe === '1D' ? 1000 : 5000;
            const response = await fetch(`/api/historical-data/${this.currentSymbol}?timeframe=${this.currentTimeframe}&limit=${limit}&${adjustmentParam}&_t=${cacheBuster}`, {
                cache: 'no-cache',
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            });
            const responseText = await response.text();
            console.log('Raw API response text:', responseText);
            let data;
            try {
                data = JSON.parse(responseText);
            } catch (parseError) {
                console.error('JSON parse error:', parseError);
                this.showErrorMessage('Invalid API response format');
                return;
            }
            console.log(`K-line API data (${adjustmentMethod} adjustment):`, data); // DEBUG
            console.log('Response status:', response.status, 'OK:', response.ok);
            console.log('Data type:', typeof data, 'Is array:', Array.isArray(data));
            console.log('Data has detail:', !!data.detail);
            
            // Check if response is an error
            if (!response.ok || data.detail || !Array.isArray(data)) {
                console.error('API error condition triggered:', {ok: response.ok, detail: data.detail, isArray: Array.isArray(data)});
                console.error('Full data:', data);
                this.showErrorMessage(data.detail || 'Failed to load chart data');
                return;
            }
            
            // Debug: Check a few sample data points for split adjustment
            const feb2025Data = data.filter(d => d.timestamp && d.timestamp.startsWith('2025-02-0')).slice(0, 5);
            console.log(`February 2025 sample data (${adjustmentMethod} adjustment):`, feb2025Data);
            if (data.length === 0) {
                this.showNoDataMessage();
                return;
            }
            
            let chartData;
            try {
                // Sort chartData by date using real trading dates (no compression)
                chartData = data.map(candle => {
                    // Add safety checks for candle properties
                    if (!candle || typeof candle !== 'object') {
                        console.warn('Invalid candle data:', candle);
                        return null;
                    }
                    
                    const dateObj = new Date(candle.timestamp); // parse ISO / epoch equally
                    if (isNaN(dateObj.getTime())) {
                        console.warn('Invalid timestamp:', candle.timestamp);
                        return null;
                    }
                    const epoch = dateObj.getTime();            // numeric ms since 1970-01-01
                    const dateStr = dateObj.toISOString().slice(0,10); // YYYY-MM-DD
                    return {
                        x: epoch,         // Chart.js time scale accepts epoch ms
                        dateStr,          // cached human-readable date
                        y: candle.close || 0,
                        o: candle.open || 0,
                        h: candle.high || 0,
                        l: candle.low || 0,
                        c: candle.close || 0,
                        v: candle.volume || 0
                    };
                }).filter(item => item !== null).sort((a, b) => a.x - b.x);
            } catch (mapError) {
                console.error('Error mapping chart data:', mapError);
                this.showErrorMessage('Error processing chart data');
                return;
            }
            
            // Filter out non-trading dates only when using daily data
            let filteredChartData;
            try {
                filteredChartData = this.currentTimeframe === '1D'
                    ? this.filterTradingDays(chartData)
                    : chartData;
                console.log('Filtered out', chartData.length - filteredChartData.length, 'non-trading days (timeframe:', this.currentTimeframe, ')');
            } catch (filterError) {
                console.error('Error filtering trading days:', filterError);
                this.showErrorMessage('Error filtering chart data');
                return;
            }

            // Use real trading dates without compression
            const displayData = filteredChartData;

            // Calculate percent change from first visible candle with safety checks
            if (displayData.length > 0 && displayData[0] && typeof displayData[0].o === 'number') {
                let base = displayData[0].o;
                if (base > 0) {  // Avoid division by zero
                    displayData.forEach(d => {
                        if (d && typeof d.c === 'number') {
                            d.percent = ((d.c - base) / base) * 100;
                        } else {
                            d.percent = 0;
                        }
                    });
                } else {
                    // If base is 0 or invalid, set all percents to 0
                    displayData.forEach(d => {
                        if (d) d.percent = 0;
                    });
                }
            } else {
                console.warn('No valid data for percent calculation');
            }
            
            // Volume data for bar chart with safety checks
            const volumeData = displayData.map(d => ({
                x: d.x,
                y: d.v || 0,
                v: d.v || 0,
                up: (d.c || 0) >= (d.o || 0),
                dateStr: d.dateStr
            }));

            // Build colour arrays for volume bars
            const volBg = volumeData.map(d => d.up ? 'rgba(0, 200, 90, 0.4)' : 'rgba(220, 53, 69, 0.4)');
            const volBorder = volumeData.map(d => d.up ? 'rgba(0, 200, 90, 0.9)' : 'rgba(220, 53, 69, 0.9)');

            console.log('Volume data for chart:', volumeData); // DEBUG
            // Set chart data
            this.chart.data.datasets[0].data = displayData;
            const adjustmentLabel = adjustmentMethod === 'none' ? 'Unadjusted' : 
                                  adjustmentMethod === 'backward' ? 'Backward Adj.' : 
                                  adjustmentMethod === 'forward' ? 'Forward Adj.' : 'Unadjusted';
            this.chart.data.datasets[0].label = `${this.currentSymbol} (${this.currentTimeframe}) - ${adjustmentLabel}`;
            
            // Clear any error overlays since we have successful data
            clearOverlays();
            this.forceRemoveErrorMessages();
            
            // Set volume data in combined chart
            this.chart.data.datasets[1].data = volumeData;
            this.chart.data.datasets[1].backgroundColor = volBg;
            this.chart.data.datasets[1].borderColor = volBorder;
            
            // Auto-scale volume Y-axis based on data
            if (volumeData.length > 0) {
                const volumes = volumeData.map(d => d.v || 0).filter(v => v > 0);
                if (volumes.length > 0) {
                    const maxVolume = Math.max(...volumes);
                    const newMax = maxVolume * 5; // Scale to use bottom 20% of chart
                    
                    // Update the volume Y-axis scale
                    this.chart.options.scales.yVolume.max = newMax;
                    this.chart.options.scales.yVolume.min = 0;
                    
                    console.log(`Volume Y-axis scaled to max: ${newMax.toLocaleString()}, volumes range: [${Math.min(...volumes)}, ${maxVolume}]`);
                } else {
                    // No volume data, set a default small scale to ensure bars are visible
                    this.chart.options.scales.yVolume.max = 1000;
                    this.chart.options.scales.yVolume.min = 0;
                    console.log('No volume data found, using default volume scale');
                }
            } else {
                // No volume data at all, set minimal scale
                this.chart.options.scales.yVolume.max = 1000;
                this.chart.options.scales.yVolume.min = 0;
                console.log('No volume data, setting minimal volume scale');
            }
            
            // Adjust visible window to last year of data
            if (displayData.length > 0) {
                const maxDate = displayData[displayData.length - 1].x;
                const minDate = new Date(maxDate);
                minDate.setFullYear(minDate.getFullYear() - 1);
                this.chart.options.scales.x.min = minDate;
                this.chart.options.scales.x.max = maxDate;
            }
            
            // Update chart with resize mode to ensure volume scale is properly recalculated
            this.chart.update('resize');
            
            // Force volume scale recalculation after chart is fully rendered
            setTimeout(() => {
                if (this.chart && volumeData.length > 0) {
                    this.updateVolumeScale();
                    this.chart.update('none'); // Silent update
                    console.log('Volume scale force-updated after render');
                }
            }, 100);

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

            // Final cleanup - remove any leftover overlay messages after successful load
            clearOverlays();
            this.forceRemoveErrorMessages();
            
            setTimeout(() => {
                if (this.chart && this.chart.update) this.chart.update();
                
                // Final aggressive cleanup after all updates
                this.forceRemoveErrorMessages();
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
                const visible = data.filter(d => d && d.x >= minX && d.x <= maxX);
                if (visible.length === 0) return;
                
                // Safely extract all prices with null checks
                const allPrices = visible.flatMap(d => {
                    if (!d) return [];
                    const prices = [];
                    if (typeof d.o === 'number') prices.push(d.o);
                    if (typeof d.h === 'number') prices.push(d.h);
                    if (typeof d.l === 'number') prices.push(d.l);
                    if (typeof d.c === 'number') prices.push(d.c);
                    return prices;
                });
                
                if (allPrices.length === 0) return; // No valid prices found
                
                const minPrice = Math.min(...allPrices);
                const maxPrice = Math.max(...allPrices);
                const padding = (maxPrice - minPrice) * 0.05 || 1;
                chart.options.scales.y.min = minPrice - padding;
                chart.options.scales.y.max = maxPrice + padding;
                chart.update();
            }, 50);
            console.log('K-line chart updated with raw historical data for accurate visual display');
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

    // Aggressive error message cleanup
    forceRemoveErrorMessages() {
        const container = document.getElementById('kline-chart-container');
        if (!container) return;
        
        // Remove all error-related elements
        const selectors = [
            '.kline-overlay-msg',
            '.position-absolute',
            '[class*="overlay"]',
            '[class*="error"]'
        ];
        
        selectors.forEach(selector => {
            container.querySelectorAll(selector).forEach(el => {
                if (el.textContent && (
                    el.textContent.includes('Failed to load') || 
                    el.textContent.includes('No data') ||
                    el.textContent.includes('Error') ||
                    el.innerHTML.includes('exclamation-triangle')
                )) {
                    el.remove();
                }
            });
        });
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

    // Helper function to update volume Y-axis scale based on visible data
    updateVolumeScale() {
        if (!this.chart) return;
        
        const xScale = this.chart.scales.x;
        const volumeData = this.chart.data.datasets[1].data; // Volume dataset
        
        if (!volumeData || volumeData.length === 0) return;
        
        let volumesToUse = [];
        
        // If we have a valid x-scale with min/max, use visible data
        if (xScale && xScale.min !== undefined && xScale.max !== undefined) {
            const minX = xScale.min;
            const maxX = xScale.max;
            
            // Filter volume data to only visible range
            const visibleVolumeData = volumeData.filter(d => d && d.x >= minX && d.x <= maxX);
            
            if (visibleVolumeData.length > 0) {
                volumesToUse = visibleVolumeData.map(d => d.v || 0).filter(v => v > 0);
            }
        }
        
        // If no visible data or no x-scale, use all data (initial load case)
        if (volumesToUse.length === 0) {
            volumesToUse = volumeData.map(d => d.v || 0).filter(v => v > 0);
        }
        
        if (volumesToUse.length > 0) {
            const maxVolume = Math.max(...volumesToUse);
            const newMax = maxVolume * 5; // Scale to use bottom 20% of chart
            
            // Update volume Y-axis scale
            this.chart.options.scales.yVolume.max = newMax;
            this.chart.options.scales.yVolume.min = 0;
            
            console.log(`Volume scale updated: max = ${newMax.toLocaleString()}, volumes: [${Math.min(...volumesToUse)}, ${maxVolume}]`);
        } else {
            // No volume data, set default scale
            this.chart.options.scales.yVolume.max = 1000;
            this.chart.options.scales.yVolume.min = 0;
            console.log('No volume data found, using default volume scale');
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
                    console.log('Manual drag pan triggered');
                    this.updateVolumeScale(); // Auto-adjust volume scale for visible range
                    this.chart.update('none'); // Silent update to apply volume scale
                } else if (this.chart.fallbackPan) {
                    // Use fallback pan method
                    this.chart.fallbackPan(panAmount);
                    console.log('Manual fallback pan triggered');
                    this.updateVolumeScale(); // Auto-adjust volume scale for visible range
                    this.chart.update('none'); // Silent update to apply volume scale
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
                        console.log('Manual touch pan triggered');
                        this.updateVolumeScale(); // Auto-adjust volume scale for visible range
                        this.chart.update('none'); // Silent update to apply volume scale
                    } else if (this.chart.fallbackPan) {
                        // Use fallback pan method
                        this.chart.fallbackPan(panAmount);
                        console.log('Manual touch fallback pan triggered');
                        this.updateVolumeScale(); // Auto-adjust volume scale for visible range
                        this.chart.update('none'); // Silent update to apply volume scale
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