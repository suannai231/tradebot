package analysis

import (
	"github.com/suannai231/tradebot/history"
)

// SwingPoint represents a local high or low
type SwingPoint struct {
	Index int
	Price float64
	IsHigh bool
}

// FindSwingPoints identifies local highs and lows in price data
func FindSwingPoints(data []history.PriceData, window int) []SwingPoint {
	var swings []SwingPoint
	
	for i := window; i < len(data)-window; i++ {
		// Check if it's a local high
		isHigh := true
		for j := i - window; j <= i + window; j++ {
			if j != i && data[j].High >= data[i].High {
				isHigh = false
				break
			}
		}
		if isHigh {
			swings = append(swings, SwingPoint{
				Index: i,
				Price: data[i].High,
				IsHigh: true,
			})
		}

		// Check if it's a local low
		isLow := true
		for j := i - window; j <= i + window; j++ {
			if j != i && data[j].Low <= data[i].Low {
				isLow = false
				break
			}
		}
		if isLow {
			swings = append(swings, SwingPoint{
				Index: i,
				Price: data[i].Low,
				IsHigh: false,
			})
		}
	}

	return swings
}

// IsInDowntrend checks if the stock was in a continuous downtrend
// Returns true if there are lower highs and lower lows
func IsInDowntrend(data []history.PriceData) bool {
	if len(data) < 20 {
		return false // Not enough data
	}

	// Find swing points (using 3-day window)
	swings := FindSwingPoints(data, 3)
	
	if len(swings) < 4 {
		return false // Need at least 2 highs and 2 lows
	}

	// Separate highs and lows
	var highs []SwingPoint
	var lows []SwingPoint
	for _, swing := range swings {
		if swing.IsHigh {
			highs = append(highs, swing)
		} else {
			lows = append(lows, swing)
		}
	}

	if len(highs) < 2 || len(lows) < 2 {
		return false
	}

	// Check for lower highs (each high is lower than the previous)
	lowerHighsCount := 0
	for i := 1; i < len(highs); i++ {
		if highs[i].Price < highs[i-1].Price {
			lowerHighsCount++
		}
	}

	// Check for lower lows (each low is lower than the previous)
	lowerLowsCount := 0
	for i := 1; i < len(lows); i++ {
		if lows[i].Price < lows[i-1].Price {
			lowerLowsCount++
		}
	}

	// Require at least 2 lower highs and 2 lower lows for a "continuous" downtrend
	return lowerHighsCount >= 2 && lowerLowsCount >= 2
}

// IsWithinCostRange checks if current price is still near the reversal point
// Returns true if current price is within 20% of the lowest low in the downtrend
func IsWithinCostRange(data []history.PriceData, currentPrice float64) bool {
	if len(data) < 10 {
		return false
	}

	// Find the lowest low in the data (the bottom of the downtrend)
	lowestLow := data[0].Low
	for _, d := range data {
		if d.Low < lowestLow {
			lowestLow = d.Low
		}
	}

	// Check if current price is within 20% above the lowest low
	// This ensures we're catching stocks near the reversal, not ones that already ran up
	maxAllowedPrice := lowestLow * 1.20
	return currentPrice <= maxAllowedPrice
}
