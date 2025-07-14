#!/usr/bin/env python3
"""
Simple ML Training Trigger (Without Unicode)

This script manually triggers ML training by adding jobs to the Redis queue.
"""

import asyncio
import json
import time
from datetime import datetime
import redis
import os
import sys

def main():
    """Manually trigger ML training for all strategies and symbols"""
    
    # Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Available strategies
    strategies = ['ensemble', 'lstm', 'sentiment', 'rl']
    
    # Get symbols from environment or use default
    symbols = os.getenv('SYMBOLS', 'TNXP').split(',')
    
    try:
        # Connect to Redis
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()
        print(f"Connected to Redis at {REDIS_URL}")
        
        # Check current queue size
        queue_size = redis_client.llen('ml_training_queue')
        print(f"Current queue size: {queue_size} jobs")
        
        # Add training jobs to queue
        jobs_added = 0
        for symbol in symbols:
            for strategy_type in strategies:
                job_id = f"manual_{strategy_type}_{symbol}_{int(time.time())}"
                
                # Create job data
                job_data = {
                    'job_id': job_id,
                    'strategy_type': strategy_type,
                    'symbol': symbol,
                    'created_at': datetime.now().isoformat(),
                    'priority': 1
                }
                
                # Push to Redis queue
                redis_client.lpush('ml_training_queue', json.dumps(job_data))
                jobs_added += 1
                
                print(f"Queued: {job_id}")
        
        # Get updated queue size
        new_queue_size = redis_client.llen('ml_training_queue')
        
        print(f"\nSuccessfully queued {jobs_added} training jobs!")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"New queue size: {new_queue_size} jobs")
        
        print(f"\nTraining jobs will be processed by the ML training service")
        print(f"Check logs with: docker logs tradebot-ml_training-1 -f")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 