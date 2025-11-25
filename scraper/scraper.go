package scraper

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/chromedp/chromedp"
	"github.com/suannai231/tradebot/analysis"
	"github.com/suannai231/tradebot/history"
	"github.com/suannai231/tradebot/models"
)

// FetchStocks scrapes the Finviz page for stocks.
func FetchStocks() ([]models.Stock, error) {
	// Create a context with a timeout
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	defer cancel()

	// Create a new allocator context for the browser
	// We use DefaultExecAllocatorOptions but could add more options here if needed (e.g. user agent)
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
		chromedp.Flag("headless", false), // Run in headful mode to see if it helps (sometimes helps with bot detection)
		chromedp.Flag("disable-gpu", false),
		chromedp.Flag("enable-automation", false),
		chromedp.Flag("disable-extensions", false),
	)
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx, opts...)
	defer cancelAlloc()

	// Create a new browser context with a custom logger to suppress verbose internal errors
	// These errors are often harmless "could not unmarshal event" warnings from the browser driver
	ctx, cancelCtx := chromedp.NewContext(allocCtx,
		chromedp.WithLogf(func(string, ...interface{}) {}),
		chromedp.WithErrorf(func(string, ...interface{}) {}),
	)
	defer cancelCtx()

	// Define all pattern URLs to scrape
	// Using s= parameter for signal-based filtering (works better than f=)
	patternURLs := []string{
		"https://finviz.com/screener.ashx?v=111&s=ta_p_doublebottom&f=sh_avgvol_o1000,sh_price_o1&ft=1",
		"https://finviz.com/screener.ashx?v=111&s=ta_p_headandshouldersinv&f=sh_avgvol_o1000,sh_price_o1&ft=1",
		"https://finviz.com/screener.ashx?v=111&s=ta_p_channelup&f=sh_avgvol_o1000,sh_price_o1&ft=1",
		"https://finviz.com/screener.ashx?v=111&s=ta_p_wedgeup&f=sh_avgvol_o1000,sh_price_o1&ft=1",
	}

	var allStocks []models.Stock
	stockMap := make(map[string]models.Stock) // Deduplicate stocks

	// Scrape each pattern URL
	for patternIdx, baseURL := range patternURLs {
		log.Printf("Scraping pattern %d/%d...", patternIdx+1, len(patternURLs))
		pageNum := 1

		for {
			// Construct URL with pagination
			url := fmt.Sprintf("%s&r=%d", baseURL, (pageNum-1)*20+1)

			var stocksJSON string
			var buf []byte

			// Navigate and extract data
			err := chromedp.Run(ctx,
				chromedp.Navigate(url),
				chromedp.Sleep(3*time.Second),
				chromedp.CaptureScreenshot(&buf),
				chromedp.Evaluate(`
					(() => {
						const stocks = [];
						const rows = document.querySelectorAll('table.screener_table tr');
						
						if (rows.length === 0) {
							return JSON.stringify([]);
						}
						
						rows.forEach(row => {
							const cols = row.querySelectorAll('td');
							if (cols.length < 10) return;
							if (!cols[1].querySelector('a')) return;

							stocks.push({
								ticker: cols[1].innerText.trim(),
								company: cols[2].innerText.trim(),
								sector: cols[3].innerText.trim(),
								industry: cols[4].innerText.trim(),
								country: cols[5].innerText.trim(),
								market_cap: cols[6].innerText.trim(),
								price: cols[8].innerText.trim(),
								change: cols[9].innerText.trim(),
								volume: cols[10].innerText.trim()
							});
						});
						
						return JSON.stringify(stocks);
					})()
				`, &stocksJSON),
			)

			if err != nil {
				return nil, fmt.Errorf("chromedp run failed on pattern %d, page %d: %w", patternIdx+1, pageNum, err)
			}

			var stocks []models.Stock
			if err := json.Unmarshal([]byte(stocksJSON), &stocks); err != nil {
				return nil, fmt.Errorf("failed to unmarshal stocks on pattern %d, page %d: %w", patternIdx+1, pageNum, err)
			}

			// If no stocks found, we've reached the end for this pattern
			if len(stocks) == 0 {
				break
			}

			// Add to map (deduplicates)
			for _, stock := range stocks {
				stockMap[stock.Ticker] = stock
			}

			// If we got fewer than 20 stocks, this is the last page
			if len(stocks) < 20 {
				break
			}

			// Limit to 10 pages per pattern
			if pageNum >= 10 {
				log.Printf("Reached page limit for pattern %d", patternIdx+1)
				break
			}

			pageNum++
		}
	}

	// Convert map to slice
	for _, stock := range stockMap {
		allStocks = append(allStocks, stock)
	}

	log.Printf("Collected %d unique stocks across all patterns", len(allStocks))

	// Filter out ETFs - Finviz URL parameters don't reliably exclude them
	var filteredStocks []models.Stock
	for _, stock := range allStocks {
		if stock.Industry == "Exchange Traded Fund" {
			continue
		}
		filteredStocks = append(filteredStocks, stock)
	}

	log.Printf("Collected %d stocks after ETF filtering. Now analyzing price history...", len(filteredStocks))

	// Filter for stocks in downtrend using parallel processing
	type result struct {
		stock        models.Stock
		valid        bool
		failureReason string
	}

	// Create channels for work distribution
	jobs := make(chan models.Stock, len(filteredStocks))
	results := make(chan result, len(filteredStocks))

	// Number of parallel workers (browser instances)
	numWorkers := 5

	// Start workers
	for w := 1; w <= numWorkers; w++ {
		go func(workerID int) {
			// Create a separate browser context for this worker
			allocCtx, allocCancel := chromedp.NewExecAllocator(context.Background(),
				append(chromedp.DefaultExecAllocatorOptions[:],
					chromedp.Flag("headless", false),
					chromedp.Flag("disable-gpu", false),
					chromedp.Flag("enable-automation", false),
					chromedp.Flag("disable-extensions", false),
				)...,
			)
			defer allocCancel()

			workerCtx, workerCancel := chromedp.NewContext(allocCtx,
				chromedp.WithLogf(func(string, ...interface{}) {}),
				chromedp.WithErrorf(func(string, ...interface{}) {}),
			)
			defer workerCancel()

			for stock := range jobs {
				// Fetch historical data
				data, err := history.FetchHistoricalData(workerCtx, stock.Ticker, 60)
				if err != nil {
					log.Printf("Worker %d: Skipping %s: %v", workerID, stock.Ticker, err)
					results <- result{stock: stock, valid: false, failureReason: "fetch_error"}
					continue
				}

				// Check if it was in a downtrend
				if !analysis.IsInDowntrend(data) {
					results <- result{stock: stock, valid: false, failureReason: "not_downtrend"}
					continue
				}

				// Parse current price
				var currentPrice float64
				fmt.Sscanf(stock.Price, "%f", &currentPrice)

				// Check if current price is still within cost range (near reversal point)
				if !analysis.IsWithinCostRange(data, currentPrice) {
					results <- result{stock: stock, valid: false, failureReason: "cost_range"}
					continue
				}

				results <- result{stock: stock, valid: true}
			}
		}(w)
	}

	// Send jobs
	for _, stock := range filteredStocks {
		jobs <- stock
	}
	close(jobs)

	// Collect results
	var downtrendStocks []models.Stock
	stats := map[string]int{
		"fetch_error":   0,
		"not_downtrend": 0,
		"cost_range":    0,
		"passed":        0,
	}

	processed := 0
	for i := 0; i < len(filteredStocks); i++ {
		res := <-results
		processed++
		
		if processed % 20 == 0 {
			log.Printf("Analyzed %d/%d stocks...", processed, len(filteredStocks))
		}

		if res.valid {
			downtrendStocks = append(downtrendStocks, res.stock)
			stats["passed"]++
		} else {
			stats[res.failureReason]++
		}
	}

	log.Printf("Analysis Complete:")
	log.Printf("- Passed: %d", stats["passed"])
	log.Printf("- Failed (Not Downtrend): %d", stats["not_downtrend"])
	log.Printf("- Failed (Cost Range >20%%): %d", stats["cost_range"])
	log.Printf("- Failed (Fetch Error): %d", stats["fetch_error"])

	log.Printf("Found %d stocks with confirmed downtrend before reversal", len(downtrendStocks))

	return downtrendStocks, nil
}
