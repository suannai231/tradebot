# Backfill Scheduling Documentation

## Overview
Your daily backfill is now scheduled using macOS **launchd** (instead of cron) to run at **9:00 AM daily**.

## 🔋 Sleep/Wake Behavior

### Will it run when Mac is sleeping?
**Short answer**: Not reliably by default, but we've configured it to be more resilient.

### What we've implemented:
1. **WakeInterval**: Set to 86400 seconds (24 hours) to attempt periodic wake
2. **launchd**: Uses macOS native scheduler (better than cron for sleep scenarios)
3. **Comprehensive logging**: Track when runs succeed/fail

### Scenarios:
- ✅ **Mac awake**: Runs normally at 9:00 AM
- ⚠️ **Mac sleeping**: May run when system wakes up naturally
- ❌ **Mac powered off**: Won't run (obviously)
- ⚠️ **User logged out**: May or may not run (depends on macOS version)

## 🎯 Best Practices for Reliable Execution

### Option 1: Keep Mac awake during backfill
```bash
# Set Mac to never sleep (when plugged in)
sudo pmset -c sleep 0
```

### Option 2: Schedule for when you're likely awake
```bash
# Change to run at 9:00 AM instead of 6:00 AM
# Edit: ~/Library/LaunchAgents/com.tradebot.backfill.plist
# Change <integer>6</integer> to <integer>9</integer>
# Then restart: ./manage_backfill_schedule.sh restart
```

### Option 3: Use Power Nap (if supported)
```bash
# Enable Power Nap (allows some tasks during sleep)
sudo pmset -a powernap 1
```

## 📱 Management Commands

```bash
# Check status
./manage_backfill_schedule.sh status

# Start/stop/restart
./manage_backfill_schedule.sh start
./manage_backfill_schedule.sh stop
./manage_backfill_schedule.sh restart

# Run immediately (for testing)
./manage_backfill_schedule.sh run-now

# View logs
./manage_backfill_schedule.sh logs
```

## 📊 Monitoring

### Check if it's running:
```bash
launchctl list | grep tradebot
```

### View recent logs:
```bash
ls -la logs/backfill_*.log
```

### View live log:
```bash
tail -f logs/backfill_$(date +%Y%m%d)_*.log
```

## 🔧 Configuration Files

- **Schedule**: `~/Library/LaunchAgents/com.tradebot.backfill.plist`
- **Script**: `./run_daily_backfill.sh`
- **Management**: `./manage_backfill_schedule.sh`
- **Logs**: `./logs/backfill_*.log`

## 🚨 Troubleshooting

### If backfill doesn't run:
1. Check if Mac was sleeping: `pmset -g log | grep -i sleep`
2. Check system logs: `log show --predicate 'subsystem == "com.apple.launchd"'`
3. Run manually: `./manage_backfill_schedule.sh run-now`

### If you want guaranteed execution:
- **Option A**: Run during active hours (9 AM - 6 PM)
- **Option B**: Set up remote trigger (cloud function, etc.)
- **Option C**: Use external server/VPS for scheduling

## 📈 Alternative Scheduling Options

### Change Time:
Edit the plist file and change the hour:
```xml
<key>Hour</key>
<integer>9</integer>  <!-- Changed from 6 to 9 AM -->
```

### Change Frequency:
For multiple times per day, use multiple StartCalendarInterval entries:
```xml
<key>StartCalendarInterval</key>
<array>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <dict>
        <key>Hour</key>
        <integer>18</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</array>
```

## 🎯 Recommendation

For most reliable execution, I recommend:
1. **Schedule for 9:00 AM** (when you're likely to be using the Mac)
2. **Check logs weekly** to ensure it's running
3. **Set up manual fallback** for missed runs

Would you like me to change the schedule time or implement any of these alternatives? 