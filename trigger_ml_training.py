#!/usr/bin/env python3
"""
Manual ML Training Trigger

This script manually triggers ML training by adding jobs to the Redis queue.
Instead of waiting for the scheduled cron jobs, you can run this to start training immediately.
"""

import asyncio
import json
import time
from datetime import datetime
import redis
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print(f"✅ Connected to Redis at {REDIS_URL}")
        
        # Check current queue size
        queue_size = redis_client.llen('ml_training_queue')
        print(f"📋 Current queue size: {queue_size} jobs")
        
        # Add training jobs to queue
        jobs_added = 0
        for symbol in symbols:
            for strategy_type in strategies:
                job_id = f"manual_{strategy_type}_{symbol.strip()}_{int(time.time())}"
                
                job_data = {
                    'job_id': job_id,
                    'strategy_type': strategy_type,
                    'symbol': symbol.strip(),
                    'priority': 1,
                    'created_at': datetime.now().isoformat(),
                    'metadata': {
                        'trigger': 'manual',
                        'user': 'admin'
                    }
                }
                
                # Add to Redis queue
                redis_client.lpush('ml_training_queue', json.dumps(job_data))
                jobs_added += 1
                
                print(f"🚀 Queued: {job_id}")
        
        print(f"\n✅ Successfully queued {jobs_added} training jobs!")
        print(f"📊 Strategies: {', '.join(strategies)}")
        print(f"📈 Symbols: {', '.join(symbols)}")
        
        # Check new queue size
        new_queue_size = redis_client.llen('ml_training_queue')
        print(f"📋 New queue size: {new_queue_size} jobs")
        
        print(f"\n🔄 Training jobs will be processed by the ML training service")
        print(f"📝 Check logs with: docker compose logs ml_training -f")
        
    except redis.RedisError as e:
        print(f"❌ Redis connection error: {e}")
        print(f"💡 Make sure Redis is running and accessible at {REDIS_URL}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 