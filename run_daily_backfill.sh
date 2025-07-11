#!/bin/bash

# Daily Backfill Script
# This script runs the comprehensive backfill process and logs the results

# Set the project directory
PROJECT_DIR="/Users/yue.yin/Library/CloudStorage/GoogleDrive-suannai231@gmail.com/My Drive/Projects/Trade"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/backfill_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment (if you have one)
# source venv/bin/activate

# Set environment variables if needed
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Log start time
echo "========================================" >> "$LOG_FILE"
echo "Daily Backfill Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the backfill with output logging
python -m tradebot.backfill.alpaca_historical >> "$LOG_FILE" 2>&1

# Check exit code
EXIT_CODE=$?

# Log completion
echo "========================================" >> "$LOG_FILE"
echo "Daily Backfill Completed: $(date)" >> "$LOG_FILE"
echo "Exit Code: $EXIT_CODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "backfill_*.log" -type f -mtime +30 -delete

# Exit with same code as the Python script
exit $EXIT_CODE 