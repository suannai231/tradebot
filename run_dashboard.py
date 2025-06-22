#!/usr/bin/env python3
"""
Trading Bot Dashboard Runner

Quick start script to run the dashboard service standalone.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the trading bot dashboard"""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Set environment variables if not already set
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/tradebot')
    env.setdefault('REDIS_URL', 'redis://localhost:6379')
    
    print("🚀 Starting Trading Bot Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8001")
    print("📡 Make sure Redis and TimescaleDB are running!")
    print("=" * 60)
    
    try:
        # Run the dashboard
        subprocess.run([
            sys.executable, '-m', 'tradebot.dashboard.main'
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 