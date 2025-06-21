#!/usr/bin/env python3
"""
Backfill Performance Benchmark

Compare different backfill configurations to show performance improvements.
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

async def run_backfill_test(name, batch_size, concurrent_requests, rate_delay, max_symbols=20):
    """Run a backfill test with specific parameters."""
    print(f"\n🧪 Testing {name}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Concurrent: {concurrent_requests}")
    print(f"   Rate Delay: {rate_delay}s")
    print(f"   Max Symbols: {max_symbols}")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'BACKFILL_BATCH_SIZE': str(batch_size),
        'BACKFILL_CONCURRENT_REQUESTS': str(concurrent_requests),
        'RATE_LIMIT_DELAY': str(rate_delay),
        'MAX_SYMBOLS': str(max_symbols),
        'LOG_LEVEL': 'WARNING'
    })
    
    start_time = time.time()
    
    # Run the fast backfill script
    process = await asyncio.create_subprocess_exec(
        'python', 'fast_backfill.py',
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Parse results from output
    output = stdout.decode()
    if "Total symbols processed:" in output:
        lines = output.split('\n')
        for line in lines:
            if "Total symbols processed:" in line:
                processed = int(line.split(': ')[1])
                break
        else:
            processed = 0
    else:
        processed = 0
    
    rate = processed / duration if duration > 0 else 0
    
    print(f"   ✅ Processed: {processed} symbols")
    print(f"   ⏱️  Duration: {duration:.1f} seconds")
    print(f"   🚀 Rate: {rate:.1f} symbols/second")
    
    return {
        'name': name,
        'processed': processed,
        'duration': duration,
        'rate': rate,
        'batch_size': batch_size,
        'concurrent': concurrent_requests,
        'delay': rate_delay
    }

async def main():
    """Run benchmark tests."""
    print("🏁 BACKFILL PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Test configurations
    tests = [
        ("Conservative (Current)", 10, 3, 0.5, 20),
        ("Balanced", 15, 6, 0.3, 20),
        ("Aggressive", 20, 8, 0.2, 20),
        ("High Performance", 25, 10, 0.15, 20),
    ]
    
    results = []
    
    for name, batch_size, concurrent, delay, max_symbols in tests:
        try:
            result = await run_backfill_test(name, batch_size, concurrent, delay, max_symbols)
            results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"{'Configuration':<20} {'Rate (sym/s)':<12} {'Duration':<10} {'Speedup':<8}")
    print("-" * 50)
    
    baseline_rate = results[0]['rate'] if results else 1
    
    for result in results:
        speedup = result['rate'] / baseline_rate if baseline_rate > 0 else 1
        print(f"{result['name']:<20} {result['rate']:<12.1f} {result['duration']:<10.1f} {speedup:<8.1f}x")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    if results:
        best = max(results, key=lambda x: x['rate'])
        print(f"🏆 Best Performance: {best['name']}")
        print(f"   Settings: Batch={best['batch_size']}, Concurrent={best['concurrent']}, Delay={best['delay']}s")
        print(f"   Rate: {best['rate']:.1f} symbols/second")
        
        if best['rate'] > baseline_rate:
            improvement = (best['rate'] / baseline_rate - 1) * 100
            print(f"   🚀 {improvement:.0f}% faster than conservative settings")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}") 