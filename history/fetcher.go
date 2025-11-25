package history

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/chromedp/chromedp"
)

// PriceData represents a single day's OHLC data
type PriceData struct {
	Date   time.Time
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume int64
}

// FetchHistoricalData fetches historical price data from Yahoo Finance using chromedp
func FetchHistoricalData(ctx context.Context, ticker string, days int) ([]PriceData, error) {
	// Calculate date range
	endDate := time.Now()
	startDate := endDate.AddDate(0, 0, -days)
	
	// Yahoo Finance historical data URL
	url := fmt.Sprintf("https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d",
		ticker, startDate.Unix(), endDate.Unix())

	var tableData string
	
	// Navigate and extract table data
	err := chromedp.Run(ctx,
		chromedp.Navigate(url),
		chromedp.Sleep(1*time.Second), // Wait for page load
		chromedp.Evaluate(`
			(() => {
				const data = [];
				const rows = document.querySelectorAll('table tbody tr');
				
				rows.forEach(row => {
					const cells = row.querySelectorAll('td');
					if (cells.length >= 6) {
						// Skip dividend/split rows
						if (cells[1].innerText.includes('Dividend') || cells[1].innerText.includes('Split')) {
							return;
						}
						
						data.push({
							date: cells[0].innerText.trim(),
							open: cells[1].innerText.trim(),
							high: cells[2].innerText.trim(),
							low: cells[3].innerText.trim(),
							close: cells[4].innerText.trim(),
							volume: cells[6].innerText.trim()
						});
					}
				});
				
				return JSON.stringify(data);
			})()
		`, &tableData),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to fetch data for %s: %w", ticker, err)
	}

	// Parse JSON
	var rawData []struct {
		Date   string `json:"date"`
		Open   string `json:"open"`
		High   string `json:"high"`
		Low    string `json:"low"`
		Close  string `json:"close"`
		Volume string `json:"volume"`
	}

	if err := json.Unmarshal([]byte(tableData), &rawData); err != nil {
		return nil, fmt.Errorf("failed to parse data for %s: %w", ticker, err)
	}

	// Convert to PriceData
	var data []PriceData
	for _, row := range rawData {
		// Parse date (format: "Nov 22, 2024")
		date, err := time.Parse("Jan 02, 2006", row.Date)
		if err != nil {
			continue
		}

		// Parse prices (remove commas)
		open, _ := strconv.ParseFloat(row.Open, 64)
		high, _ := strconv.ParseFloat(row.High, 64)
		low, _ := strconv.ParseFloat(row.Low, 64)
		close, _ := strconv.ParseFloat(row.Close, 64)
		
		// Parse volume (remove commas)
		volumeStr := row.Volume
		for i := 0; i < len(volumeStr); i++ {
			if volumeStr[i] == ',' {
				volumeStr = volumeStr[:i] + volumeStr[i+1:]
			}
		}
		volume, _ := strconv.ParseInt(volumeStr, 10, 64)

		data = append(data, PriceData{
			Date:   date,
			Open:   open,
			High:   high,
			Low:    low,
			Close:  close,
			Volume: volume,
		})
	}

	return data, nil
}
