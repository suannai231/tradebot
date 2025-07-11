#!/bin/bash

# Backfill Schedule Management Script
# Use this script to manage the daily backfill schedule

PLIST_FILE="$HOME/Library/LaunchAgents/com.tradebot.backfill.plist"
SERVICE_NAME="com.tradebot.backfill"

case "$1" in
    start)
        echo "Starting backfill schedule..."
        launchctl load "$PLIST_FILE"
        echo "✅ Backfill schedule started (runs daily at 9:00 AM)"
        ;;
    stop)
        echo "Stopping backfill schedule..."
        launchctl unload "$PLIST_FILE"
        echo "✅ Backfill schedule stopped"
        ;;
    restart)
        echo "Restarting backfill schedule..."
        launchctl unload "$PLIST_FILE" 2>/dev/null
        launchctl load "$PLIST_FILE"
        echo "✅ Backfill schedule restarted"
        ;;
    status)
        echo "Checking backfill schedule status..."
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo "✅ Backfill schedule is ACTIVE"
            echo "📅 Scheduled to run daily at 9:00 AM"
            
            # Show last run info if logs exist
            LOG_DIR="/Users/yue.yin/Library/CloudStorage/GoogleDrive-suannai231@gmail.com/My Drive/Projects/Trade/logs"
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR"/backfill_*.log 2>/dev/null | head -1)
                if [ -n "$LATEST_LOG" ]; then
                    echo "📝 Last run log: $LATEST_LOG"
                    echo "🕐 Last run time: $(grep "Daily Backfill Started" "$LATEST_LOG" | tail -1 | cut -d: -f2-)"
                fi
            fi
        else
            echo "❌ Backfill schedule is NOT ACTIVE"
        fi
        ;;
    run-now)
        echo "Running backfill immediately..."
        "./run_daily_backfill.sh"
        ;;
    logs)
        LOG_DIR="/Users/yue.yin/Library/CloudStorage/GoogleDrive-suannai231@gmail.com/My Drive/Projects/Trade/logs"
        if [ -d "$LOG_DIR" ]; then
            echo "📝 Recent backfill logs:"
            ls -lt "$LOG_DIR"/backfill_*.log | head -5
            echo ""
            echo "To view the latest log:"
            echo "cat \"$(ls -t "$LOG_DIR"/backfill_*.log | head -1)\""
        else
            echo "❌ No log directory found"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|run-now|logs}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the daily backfill schedule"
        echo "  stop      - Stop the daily backfill schedule"
        echo "  restart   - Restart the daily backfill schedule"
        echo "  status    - Check if the schedule is running"
        echo "  run-now   - Run backfill immediately (for testing)"
        echo "  logs      - Show recent backfill logs"
        exit 1
        ;;
esac 