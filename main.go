package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/suannai231/tradebot/scraper"
)

func main() {
	fmt.Println("Starting Finviz Scraper...")
	stocks, err := scraper.FetchStocks()
	if err != nil {
		log.Fatalf("Error fetching stocks: %v", err)
	}

	fmt.Println("Found stocks:")
	for _, s := range stocks {
		fmt.Printf("Ticker: %s | Sector: %s | Industry: %s | Price: %s | Change: %s\n",
			s.Ticker, s.Sector, s.Industry, s.Price, s.Change)
	}

	// Export to CSV
	if len(stocks) > 0 {
		filename := fmt.Sprintf("stocks_%s.csv", time.Now().Format("2006-01-02_15-04"))
		file, err := os.Create(filename)
		if err != nil {
			log.Printf("Failed to create CSV file: %v", err)
			return
		}
		defer file.Close()

		writer := csv.NewWriter(file)
		defer writer.Flush()

		// Write header
		header := []string{"Ticker", "Company", "Sector", "Industry", "Country", "Market Cap", "Price", "Change", "Volume"}
		if err := writer.Write(header); err != nil {
			log.Printf("Failed to write header: %v", err)
			return
		}

		// Write data
		for _, s := range stocks {
			record := []string{
				s.Ticker,
				s.Company,
				s.Sector,
				s.Industry,
				s.Country,
				s.MarketCap,
				s.Price,
				s.Change,
				s.Volume,
			}
			if err := writer.Write(record); err != nil {
				log.Printf("Failed to write record for %s: %v", s.Ticker, err)
			}
		}

		fmt.Printf("\nResults saved to %s\n", filename)
	}
}
