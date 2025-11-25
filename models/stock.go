package models

// Stock represents the data scraped for a single stock.
type Stock struct {
	Ticker    string `json:"ticker"`
	Company   string `json:"company"`
	Sector    string `json:"sector"`
	Industry  string `json:"industry"`
	Country   string `json:"country"`
	MarketCap string `json:"market_cap"`
	Price     string `json:"price"`
	Change    string `json:"change"`
	Volume    string `json:"volume"`
}
