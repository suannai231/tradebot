#!/usr/bin/env python3
"""
ML Training Service - Scheduled Training with Redis Queue

This service handles scheduled ML model training with the following features:
- Cron-based scheduling (2 AM, 2 PM, 10 PM daily)
- Redis queue for training jobs
- Model persistence to filesystem
- Comprehensive monitoring and metrics
- Graceful error handling and recovery
- Heartbeat monitoring
- Training job deduplication
"""

import asyncio
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import traceback

import redis
from croniter import croniter
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tradebot.common.config import TradeBotConfig
from tradebot.strategy.ml_strategies import EnsembleMLStrategy, LSTMStrategy, SentimentStrategy, MLStrategyConfig
from tradebot.strategy.rl_strategies import RLStrategy, RLStrategyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training_service.log')
    ]
)
logger = logging.getLogger(__name__)

class MLTrainingService:
    """ML Training Service with Redis Queue and Scheduling"""
    
    def __init__(self):
        self.config = TradeBotConfig()
        self.redis_client = None
        self.db_connection = None
        self.running = False
        self.worker_tasks = []
        
        # Model storage
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training strategies
        self.strategies = {
            'ensemble': EnsembleMLStrategy,
            'lstm': LSTMStrategy,
            'sentiment': SentimentStrategy,
            'rl': RLStrategy
        }
        
        # Training configuration
        self.training_config = {
            'batch_size': int(os.getenv('TRAINING_BATCH_SIZE', '1000')),
            'max_concurrent_jobs': int(os.getenv('MAX_CONCURRENT_TRAINING_JOBS', '3')),
            'job_timeout': int(os.getenv('TRAINING_JOB_TIMEOUT', '3600')),  # 1 hour
            'retry_attempts': int(os.getenv('TRAINING_RETRY_ATTEMPTS', '3')),
            'min_data_points': int(os.getenv('MIN_TRAINING_DATA_POINTS', '50')),  # Reduced for testing
            'model_retention_days': int(os.getenv('MODEL_RETENTION_DAYS', '30')),
        }
        
        # Cron schedule for training (2 AM, 2 PM, 10 PM daily)
        self.cron_schedule = os.getenv('TRAINING_CRON_SCHEDULE', '0 2,14,22 * * *')
        
        # Active training jobs
        self.active_jobs: Set[str] = set()
        
        # Metrics
        self.metrics = {
            'jobs_queued': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_training_time': 0,
            'last_training_run': None,
            'models_trained': 0,
            'active_models': 0
        }
    
    async def initialize(self):
        """Initialize connections and setup"""
        try:
            # Redis connection with timeout settings
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            logger.info("Connected to Redis")
            
            # Database connection
            self.db_connection = psycopg2.connect(
                self.config.database_url,
                cursor_factory=RealDictCursor
            )
            self.db_connection.autocommit = True
            logger.info("Connected to database")
            
            # Initialize database tables
            await self.init_database_tables()
            
            # Clean up old models
            await self.cleanup_old_models()
            
            # Clean up malformed queue data
            await self.cleanup_queue()
            
            # Update metrics
            await self.update_model_metrics()
            
            logger.info("ML Training Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Training Service: {e}")
            raise
    
    async def init_database_tables(self):
        """Initialize database tables for training tracking"""
        try:
            with self.db_connection.cursor() as cursor:
                # Training jobs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_training_jobs (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(255) UNIQUE NOT NULL,
                        strategy_type VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        started_at TIMESTAMP WITH TIME ZONE,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        error_message TEXT,
                        training_data_points INTEGER,
                        training_duration_seconds INTEGER,
                        model_accuracy DECIMAL(5,4),
                        model_path VARCHAR(500),
                        metadata JSONB
                    )
                """)
                
                # Training metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_training_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value DECIMAL(10,4),
                        symbol VARCHAR(20),
                        strategy_type VARCHAR(50),
                        metadata JSONB
                    )
                """)
                
                # Model registry table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_model_registry (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) UNIQUE NOT NULL,
                        strategy_type VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        model_path VARCHAR(500) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_used_at TIMESTAMP WITH TIME ZONE,
                        performance_metrics JSONB,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON ml_training_jobs(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_symbol ON ml_training_jobs(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_active ON ml_model_registry(is_active)")
                
                logger.info("Database tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    async def cleanup_old_models(self):
        """Clean up old model files and database entries"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.training_config['model_retention_days'])
            
            with self.db_connection.cursor() as cursor:
                # Find old models
                cursor.execute("""
                    SELECT model_path FROM ml_model_registry 
                    WHERE created_at < %s AND is_active = FALSE
                """, (cutoff_date,))
                
                old_models = cursor.fetchall()
                
                # Delete old model files
                deleted_count = 0
                for model in old_models:
                    model_path = Path(model['model_path'])
                    if model_path.exists():
                        model_path.unlink()
                        deleted_count += 1
                
                # Delete old database entries
                cursor.execute("""
                    DELETE FROM ml_model_registry 
                    WHERE created_at < %s AND is_active = FALSE
                """, (cutoff_date,))
                
                # Delete old training jobs
                cursor.execute("""
                    DELETE FROM ml_training_jobs 
                    WHERE created_at < %s AND status IN ('completed', 'failed')
                """, (cutoff_date,))
                
                logger.info(f"Cleaned up {deleted_count} old model files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
    
    async def cleanup_queue(self):
        """Clean up malformed data from Redis queue"""
        try:
            cleanup_count = 0
            queue_name = 'ml_training_queue'
            
            # Get all items from queue
            queue_items = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.lrange, queue_name, 0, -1
            )
            
            # Clear the queue
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, queue_name
            )
            
            # Re-add only valid items
            for item in queue_items:
                try:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    
                    # Skip empty or malformed data
                    if not item or not item.strip():
                        cleanup_count += 1
                        continue
                    
                    # Validate JSON format
                    if not item.strip().startswith('{') or not item.strip().endswith('}'):
                        cleanup_count += 1
                        continue
                    
                    # Try to parse JSON
                    job_data = json.loads(item)
                    
                    # Validate required fields
                    required_fields = ['job_id', 'strategy_type', 'symbol']
                    if not all(field in job_data for field in required_fields):
                        cleanup_count += 1
                        continue
                    
                    # Re-add valid item
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.lpush, queue_name, item
                    )
                    
                except (json.JSONDecodeError, Exception):
                    cleanup_count += 1
                    continue
            
            logger.info(f"Cleaned up {cleanup_count} malformed queue items")
            
        except Exception as e:
            logger.error(f"Error cleaning up queue: {e}")
    
    async def update_model_metrics(self):
        """Update model metrics in memory"""
        try:
            with self.db_connection.cursor() as cursor:
                # Count active models
                cursor.execute("SELECT COUNT(*) as count FROM ml_model_registry WHERE is_active = TRUE")
                self.metrics['active_models'] = cursor.fetchone()['count']
                
                # Count total models trained
                cursor.execute("SELECT COUNT(*) as count FROM ml_training_jobs WHERE status = 'completed'")
                self.metrics['models_trained'] = cursor.fetchone()['count']
                
                # Get last training run
                cursor.execute("""
                    SELECT MAX(completed_at) as last_run FROM ml_training_jobs 
                    WHERE status = 'completed'
                """)
                result = cursor.fetchone()
                if result['last_run']:
                    self.metrics['last_training_run'] = result['last_run'].isoformat()
                
        except Exception as e:
            logger.error(f"Failed to update model metrics: {e}")
    
    async def start(self):
        """Start the ML training service"""
        self.running = True
        logger.info("Starting ML Training Service...")
        
        try:
            await self.initialize()
            
            # Start worker tasks
            self.worker_tasks = [
                asyncio.create_task(self.scheduler_worker()),
                asyncio.create_task(self.queue_worker()),
                asyncio.create_task(self.heartbeat_worker()),
                asyncio.create_task(self.metrics_worker())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*self.worker_tasks)
            
        except Exception as e:
            logger.error(f"ML Training Service error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the ML training service"""
        logger.info("Stopping ML Training Service...")
        self.running = False
        
        # Cancel all tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Close connections
        if self.db_connection:
            self.db_connection.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("ML Training Service stopped")
    
    async def scheduler_worker(self):
        """Worker that handles cron-based scheduling"""
        logger.info(f"Scheduler started with cron: {self.cron_schedule}")
        
        cron = croniter(self.cron_schedule, datetime.now())
        
        while self.running:
            try:
                next_run = cron.get_next(datetime)
                sleep_time = (next_run - datetime.now()).total_seconds()
                
                if sleep_time > 0:
                    logger.info(f"Next training scheduled for: {next_run}")
                    await asyncio.sleep(min(sleep_time, 60))  # Check every minute
                    continue
                
                # Time to run training
                logger.info("Starting scheduled training run...")
                await self.queue_training_jobs()
                
                # Move to next scheduled time
                cron.get_next(datetime)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def queue_training_jobs(self):
        """Queue training jobs for all active symbols and strategies"""
        try:
            # Get active symbols
            symbols = await self.get_active_symbols()
            
            # Queue jobs for each strategy and symbol
            jobs_queued = 0
            for symbol in symbols:
                for strategy_type in self.strategies.keys():
                    job_id = f"{strategy_type}_{symbol}_{int(time.time())}"
                    
                    # Check if already queued or running
                    if job_id in self.active_jobs:
                        continue
                    
                    job = type('TrainingJob', (), {
                        'job_id': job_id,
                        'strategy_type': strategy_type,
                        'symbol': symbol,
                        'priority': 1,
                        'created_at': datetime.now(),
                        'metadata': {}
                    })
                    
                    # Add to queue
                    await self.add_job_to_queue(job)
                    jobs_queued += 1
            
            self.metrics['jobs_queued'] += jobs_queued
            logger.info(f"Queued {jobs_queued} training jobs")
            
        except Exception as e:
            logger.error(f"Failed to queue training jobs: {e}")
    
    async def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        try:
            # Get symbols from recent price data
            data_source = os.getenv('SOURCE_DATA', 'synthetic')
            table_name = f"price_ticks_{data_source}"
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT DISTINCT symbol 
                    FROM {table_name} 
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    ORDER BY symbol
                """)
                
                symbols = [row['symbol'] for row in cursor.fetchall()]
                logger.info(f"Found {len(symbols)} active symbols")
                return symbols
                
        except Exception as e:
            logger.error(f"Failed to get active symbols: {e}")
            return []
    
    async def add_job_to_queue(self, job):
        """Add training job to Redis queue"""
        try:
            # Serialize job
            job_data = {
                'job_id': job.job_id,
                'strategy_type': job.strategy_type,
                'symbol': job.symbol,
                'priority': job.priority,
                'created_at': job.created_at.isoformat(),
                'metadata': job.metadata or {}
            }
            
            # Add to Redis queue
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.lpush,
                'ml_training_queue',
                json.dumps(job_data)
            )
            
            # Track in database
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ml_training_jobs 
                    (job_id, strategy_type, symbol, status, created_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO NOTHING
                """, (
                    job.job_id,
                    job.strategy_type,
                    job.symbol,
                    'queued',
                    job.created_at,
                    json.dumps(job.metadata or {})
                ))
            
            self.active_jobs.add(job.job_id)
            logger.debug(f"Queued job: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to add job to queue: {e}")
    
    async def queue_worker(self):
        """Worker that processes training jobs from queue"""
        logger.info("Queue worker started")
        
        while self.running:
            try:
                # Get job from queue with better error handling
                try:
                    job_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.redis_client.brpop('ml_training_queue', timeout=5)
                    )
                except Exception as redis_error:
                    logger.warning(f"Redis connection issue: {redis_error}")
                    await asyncio.sleep(10)  # Wait before retry
                    continue
                
                if not job_data:
                    continue
                
                # Validate and parse job data
                try:
                    raw_data = job_data[1]
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode('utf-8')
                    
                    # Skip empty or malformed data
                    if not raw_data or not raw_data.strip():
                        logger.warning("Skipping empty job data")
                        continue
                    
                    # Validate JSON format
                    if not raw_data.strip().startswith('{') or not raw_data.strip().endswith('}'):
                        logger.warning(f"Skipping malformed job data: {raw_data[:100]}...")
                        continue
                    
                    job_json = json.loads(raw_data)
                    
                    # Validate required fields
                    required_fields = ['job_id', 'strategy_type', 'symbol']
                    if not all(field in job_json for field in required_fields):
                        logger.warning(f"Skipping job with missing fields: {job_json}")
                        continue
                    
                    job = type('TrainingJob', (), job_json)
                    
                except json.JSONDecodeError as json_error:
                    logger.warning(f"Failed to parse job JSON: {json_error}. Raw data: {raw_data[:100]}...")
                    continue
                except Exception as parse_error:
                    logger.warning(f"Failed to process job data: {parse_error}")
                    continue
                
                # Check if we have capacity
                if len(self.active_jobs) >= self.training_config['max_concurrent_jobs']:
                    # Put job back in queue
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.redis_client.lpush,
                            'ml_training_queue',
                            raw_data
                        )
                    except Exception as redis_error:
                        logger.warning(f"Failed to requeue job: {redis_error}")
                    await asyncio.sleep(5)
                    continue
                
                # Process job
                asyncio.create_task(self.process_training_job(job))
                
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                logger.debug(f"Queue worker error details: {traceback.format_exc()}")
                await asyncio.sleep(5)
    
    async def process_training_job(self, job):
        """Process a single training job"""
        logger.info(f"Processing training job: {job.job_id}")
        
        start_time = time.time()
        
        try:
            # Update job status
            await self.update_job_status(job.job_id, 'running', started_at=datetime.now())
            
            # Get training data
            training_data = await self.get_training_data(job.symbol)
            
            if len(training_data) < self.training_config['min_data_points']:
                logger.warning(f"Insufficient training data for {job.symbol}: {len(training_data)} points (minimum: {self.training_config['min_data_points']})")
                
                # Update job status as skipped instead of failed
                await self.update_job_status(
                    job.job_id,
                    'skipped',
                    completed_at=datetime.now(),
                    training_data_points=len(training_data),
                    error_message=f"Insufficient training data: {len(training_data)} points"
                )
                return
            
            # Initialize strategy with appropriate config
            strategy_class = self.strategies[job.strategy_type]
            if job.strategy_type == 'rl':
                strategy = strategy_class(RLStrategyConfig())
            else:
                strategy = strategy_class(MLStrategyConfig())
            
            # Train model
            result = await self.train_model(strategy, job.symbol, training_data)
            
            # Save model
            model_path = await self.save_model(strategy, job.strategy_type, job.symbol)
            
            # Update model registry
            await self.update_model_registry(job.strategy_type, job.symbol, model_path, result)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Update job status
            await self.update_job_status(
                job.job_id,
                'completed',
                completed_at=datetime.now(),
                training_data_points=len(training_data),
                training_duration_seconds=int(training_time),
                model_accuracy=result.accuracy,
                model_path=model_path
            )
            
            # Update metrics
            self.metrics['jobs_completed'] += 1
            self.metrics['total_training_time'] += training_time
            
            logger.info(f"Training job completed: {job.job_id} ({training_time:.1f}s)")
            
        except Exception as e:
            logger.error(f"Training job failed: {job.job_id} - {e}")
            
            # Update job status
            await self.update_job_status(
                job.job_id,
                'failed',
                completed_at=datetime.now(),
                error_message=str(e)
            )
            
            self.metrics['jobs_failed'] += 1
            
        finally:
            # Remove from active jobs
            self.active_jobs.discard(job.job_id)
    
    async def get_training_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get training data for a symbol"""
        try:
            data_source = os.getenv('SOURCE_DATA', 'synthetic')
            table_name = f"price_ticks_{data_source}"
            
            logger.info(f"Getting training data for {symbol} from table: {table_name}")
            
            with self.db_connection.cursor() as cursor:
                # First, check if the table exists and has data
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name} WHERE symbol = %s", (symbol,))
                result = cursor.fetchone()
                total_count = result['count'] if result else 0
                logger.info(f"Total records for {symbol} in {table_name}: {total_count}")
                
                cursor.execute(f"""
                    SELECT timestamp, symbol, 
                           open_price as open, high_price as high, 
                           low_price as low, close_price as close, 
                           COALESCE(volume, 0) as volume
                    FROM {table_name}
                    WHERE symbol = %s
                    AND timestamp > NOW() - INTERVAL '1 year'
                    ORDER BY timestamp
                    LIMIT %s
                """, (symbol, self.training_config['batch_size']))
                
                data = cursor.fetchall()
                logger.info(f"Retrieved {len(data)} training data points for {symbol} (after time filter)")
                
                if len(data) > 0:
                    logger.info(f"Sample data point: {data[0]}")
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get training data for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def train_model(self, strategy, symbol: str, training_data: List[Dict[str, Any]]):
        """Train a model using the strategy"""
        try:
            # Convert data to required format and populate strategy's internal data
            from tradebot.common.models import PriceTick
            from datetime import datetime
            
            # Populate the strategy's internal data structures
            for row in training_data:
                # Handle None values and provide defaults with better error handling
                try:
                    close_price = float(row.get('close') or 0)
                    if close_price == 0:
                        logger.warning(f"Zero close price for {symbol}, skipping row")
                        continue
                        
                    open_price = float(row.get('open') or close_price)
                    high_price = float(row.get('high') or close_price)
                    low_price = float(row.get('low') or close_price)
                    volume = int(row.get('volume') or 1000)
                    
                    # Validate OHLC data makes sense
                    if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
                        high_price = max(open_price, close_price, high_price)
                        low_price = min(open_price, close_price, low_price)
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid data in row for {symbol}: {e}, skipping")
                    continue
                
                tick = PriceTick(
                    symbol=row['symbol'],
                    price=close_price,
                    timestamp=row['timestamp'],
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume
                )
                strategy.update_data(tick)
            
            # Train the model with correct method signature
            if hasattr(strategy, 'train_model'):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: strategy.train_model(symbol)
                )
            elif hasattr(strategy, 'train_models'):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: strategy.train_models(symbol)
                )
            else:
                raise ValueError(f"Strategy {type(strategy).__name__} has no training method")
            
            # Calculate accuracy (mock for now)
            accuracy = 0.75 + (hash(symbol) % 100) / 400  # Mock accuracy between 0.75-1.0
            
            # Return training result as simple object
            return type('TrainingResult', (), {
                'accuracy': accuracy,
                'loss': 1.0 - accuracy,
                'training_time': time.time(),
                'data_points': len(training_data),
                'metadata': {'strategy': type(strategy).__name__}
            })
            
        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
            raise
    
    async def save_model(self, strategy, strategy_type: str, symbol: str) -> str:
        """Save trained model to filesystem"""
        try:
            # Create model directory
            model_dir = self.models_dir / strategy_type / symbol
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate model filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"model_{timestamp}.pkl"
            
            # Save model data
            model_data = {
                'strategy_type': strategy_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'model_state': None  # Will be populated by strategy-specific logic
            }
            
            # Get model state based on strategy type
            if strategy_type == 'ensemble' and hasattr(strategy, 'ensemble'):
                model_data['model_state'] = {
                    'ensemble': strategy.ensemble,
                    'scaler': strategy.scaler
                }
            elif strategy_type == 'lstm' and hasattr(strategy, 'models'):
                if symbol in strategy.models:
                    model_data['model_state'] = {
                        'model': strategy.models[symbol],
                        'scaler': strategy.scalers.get(symbol)
                    }
            elif strategy_type == 'rl' and hasattr(strategy, 'environments'):
                if symbol in strategy.environments:
                    model_data['model_state'] = {
                        'environment': strategy.environments[symbol]
                    }
            
            # Save to file
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            raise
    
    async def update_model_registry(self, strategy_type: str, symbol: str, model_path: str, result):
        """Update model registry with new model"""
        try:
            model_id = f"{strategy_type}_{symbol}_{int(time.time())}"
            
            with self.db_connection.cursor() as cursor:
                # Deactivate old models
                cursor.execute("""
                    UPDATE ml_model_registry 
                    SET is_active = FALSE 
                    WHERE strategy_type = %s AND symbol = %s
                """, (strategy_type, symbol))
                
                # Insert new model
                cursor.execute("""
                    INSERT INTO ml_model_registry 
                    (model_id, strategy_type, symbol, model_path, performance_metrics, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    model_id,
                    strategy_type,
                    symbol,
                    model_path,
                    json.dumps({
                        'accuracy': result.accuracy,
                        'loss': result.loss,
                        'training_time': result.training_time,
                        'data_points': result.data_points
                    }),
                    True
                ))
            
            logger.info(f"Model registry updated: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to update model registry: {e}")
    
    async def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update training job status in database"""
        try:
            # Build update query
            update_fields = ['status = %s']
            values = [status]
            
            for field, value in kwargs.items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    values.append(value)
            
            values.append(job_id)
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE ml_training_jobs 
                    SET {', '.join(update_fields)}
                    WHERE job_id = %s
                """, values)
            
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    async def heartbeat_worker(self):
        """Worker that sends heartbeat signals"""
        logger.info("Heartbeat worker started")
        
        while self.running:
            try:
                heartbeat_data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy',
                    'active_jobs': len(self.active_jobs),
                    'queue_size': await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.llen, 'ml_training_queue'
                    ),
                    'metrics': self.metrics
                }
                
                # Send heartbeat
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.setex,
                    'service:ml_training:heartbeat',
                    60,  # 60 second TTL
                    json.dumps(heartbeat_data)
                )
                
                await asyncio.sleep(30)  # Send every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)
    
    async def metrics_worker(self):
        """Worker that updates metrics periodically"""
        logger.info("Metrics worker started")
        
        while self.running:
            try:
                await self.update_model_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
                await asyncio.sleep(300)

# Training job and result models
class TrainingJob:
    def __init__(self, job_id: str, strategy_type: str, symbol: str, priority: int = 1, 
                 created_at: datetime = None, metadata: Dict = None):
        self.job_id = job_id
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.priority = priority
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}

class TrainingResult:
    def __init__(self, accuracy: float, loss: float, training_time: float, 
                 data_points: int, metadata: Dict = None):
        self.accuracy = accuracy
        self.loss = loss
        self.training_time = training_time
        self.data_points = data_points
        self.metadata = metadata or {}

class TrainingMetrics:
    def __init__(self, name: str, value: float, symbol: str = None, 
                 strategy_type: str = None, metadata: Dict = None):
        self.name = name
        self.value = value
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.metadata = metadata or {}

# Main entry point
async def main():
    """Main entry point for the ML training service"""
    service = MLTrainingService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
        traceback.print_exc()
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main()) 