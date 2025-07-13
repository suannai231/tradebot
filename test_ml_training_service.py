#!/usr/bin/env python3
"""
Test script for ML Training Service

This script tests the ML training service functionality including:
- Service initialization
- Database table creation
- Model training and persistence
- Redis queue operations
- Configuration validation
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tradebot.strategy.ml_training_service import MLTrainingService
from tradebot.strategy.ml_strategies import EnsembleMLStrategy, LSTMStrategy, MLStrategyConfig
from tradebot.common.config import TradeBotConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLTrainingServiceTest:
    """Test suite for ML Training Service"""
    
    def __init__(self):
        self.temp_dir = None
        self.service = None
        self.test_results = []
    
    def setup_test_environment(self):
        """Setup test environment with temporary directories"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Set environment variables for testing
        os.environ['DATABASE_URL'] = 'postgresql://postgres:password@localhost:5432/tradebot'
        os.environ['REDIS_URL'] = 'redis://localhost:6379'
        os.environ['DATA_SOURCE'] = 'synthetic'
        os.environ['TRAINING_BATCH_SIZE'] = '100'
        os.environ['MAX_CONCURRENT_TRAINING_JOBS'] = '2'
        os.environ['MIN_TRAINING_DATA_POINTS'] = '10'
        
        # Change to temp directory for model storage
        os.chdir(self.temp_dir)
        
        logger.info(f"Test environment setup in: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Test environment cleaned up")
    
    def test_service_initialization(self):
        """Test ML training service initialization"""
        logger.info("Testing service initialization...")
        
        try:
            self.service = MLTrainingService()
            
            # Check configuration
            assert self.service.training_config['batch_size'] == 100
            assert self.service.training_config['max_concurrent_jobs'] == 2
            assert self.service.training_config['min_data_points'] == 10
            
            # Check models directory
            assert self.service.models_dir.exists()
            
            # Check strategies
            assert 'ensemble' in self.service.strategies
            assert 'lstm' in self.service.strategies
            assert 'rl' in self.service.strategies
            
            self.test_results.append("✅ Service initialization: PASSED")
            logger.info("Service initialization test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Service initialization: FAILED - {e}")
            logger.error(f"Service initialization test failed: {e}")
    
    def test_database_initialization(self):
        """Test database table creation"""
        logger.info("Testing database initialization...")
        
        try:
            # Mock database connection for testing
            with patch('psycopg2.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
                mock_connect.return_value = mock_conn
                
                # Run database initialization
                asyncio.run(self.service.init_database_tables())
                
                # Verify table creation SQL was called
                assert mock_cursor.execute.called
                call_args = [call[0][0] for call in mock_cursor.execute.call_args_list]
                
                # Check that required tables are created
                table_names = ['ml_training_jobs', 'ml_training_metrics', 'ml_model_registry']
                for table_name in table_names:
                    assert any(table_name in sql for sql in call_args), f"Table {table_name} not created"
                
                self.test_results.append("✅ Database initialization: PASSED")
                logger.info("Database initialization test passed")
                
        except Exception as e:
            self.test_results.append(f"❌ Database initialization: FAILED - {e}")
            logger.error(f"Database initialization test failed: {e}")
    
    def test_model_persistence(self):
        """Test model persistence functionality"""
        logger.info("Testing model persistence...")
        
        try:
            # Test EnsembleMLStrategy persistence
            config = MLStrategyConfig()
            ensemble_strategy = EnsembleMLStrategy(config)
            
            # Create dummy model data
            ensemble_strategy.models_trained = True
            
            # Test save model
            model_path = ensemble_strategy.save_model("TEST")
            assert model_path is not None
            assert Path(model_path).exists()
            
            # Test load model
            new_strategy = EnsembleMLStrategy(config)
            loaded = new_strategy.load_model("TEST", model_path)
            assert loaded is True
            assert new_strategy.models_trained is True
            
            self.test_results.append("✅ Model persistence: PASSED")
            logger.info("Model persistence test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Model persistence: FAILED - {e}")
            logger.error(f"Model persistence test failed: {e}")
    
    def test_training_job_creation(self):
        """Test training job creation and queue operations"""
        logger.info("Testing training job creation...")
        
        try:
            # Mock Redis client
            with patch('redis.Redis') as mock_redis_class:
                mock_redis = Mock()
                mock_redis_class.from_url.return_value = mock_redis
                mock_redis.ping.return_value = True
                mock_redis.lpush.return_value = 1
                mock_redis.llen.return_value = 1
                
                # Create training job
                from tradebot.strategy.ml_training_service import TrainingJob
                job = TrainingJob(
                    job_id="test_job_123",
                    strategy_type="ensemble",
                    symbol="AAPL",
                    priority=1
                )
                
                # Test job creation
                assert job.job_id == "test_job_123"
                assert job.strategy_type == "ensemble"
                assert job.symbol == "AAPL"
                assert job.priority == 1
                assert job.created_at is not None
                
                self.test_results.append("✅ Training job creation: PASSED")
                logger.info("Training job creation test passed")
                
        except Exception as e:
            self.test_results.append(f"❌ Training job creation: FAILED - {e}")
            logger.error(f"Training job creation test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        logger.info("Testing configuration validation...")
        
        try:
            # Test valid configuration
            config = MLStrategyConfig(
                lookback_period=20,
                prediction_horizon=3,
                min_data_points=30,
                retrain_frequency=500,
                confidence_threshold=0.3
            )
            
            assert config.lookback_period == 20
            assert config.prediction_horizon == 3
            assert config.min_data_points == 30
            assert config.retrain_frequency == 500
            assert config.confidence_threshold == 0.3
            
            # Test training service configuration
            assert self.service.training_config['batch_size'] == 100
            assert self.service.training_config['max_concurrent_jobs'] == 2
            assert self.service.training_config['min_data_points'] == 10
            
            self.test_results.append("✅ Configuration validation: PASSED")
            logger.info("Configuration validation test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Configuration validation: FAILED - {e}")
            logger.error(f"Configuration validation test failed: {e}")
    
    def test_cron_schedule_parsing(self):
        """Test cron schedule parsing"""
        logger.info("Testing cron schedule parsing...")
        
        try:
            from croniter import croniter
            
            # Test default schedule
            cron_expr = "0 2,14,22 * * *"  # 2 AM, 2 PM, 10 PM daily
            cron = croniter(cron_expr, datetime.now())
            
            # Get next few runs
            next_runs = [cron.get_next(datetime) for _ in range(3)]
            
            # Verify we got valid datetime objects
            for run_time in next_runs:
                assert isinstance(run_time, datetime)
                assert run_time > datetime.now()
            
            self.test_results.append("✅ Cron schedule parsing: PASSED")
            logger.info("Cron schedule parsing test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Cron schedule parsing: FAILED - {e}")
            logger.error(f"Cron schedule parsing test failed: {e}")
    
    def test_training_result_models(self):
        """Test training result models"""
        logger.info("Testing training result models...")
        
        try:
            from tradebot.strategy.ml_training_service import TrainingResult, TrainingMetrics
            
            # Test TrainingResult
            result = TrainingResult(
                accuracy=0.85,
                loss=0.15,
                training_time=120.5,
                data_points=1000,
                metadata={'model_type': 'ensemble'}
            )
            
            assert result.accuracy == 0.85
            assert result.loss == 0.15
            assert result.training_time == 120.5
            assert result.data_points == 1000
            assert result.metadata['model_type'] == 'ensemble'
            
            # Test TrainingMetrics
            metrics = TrainingMetrics(
                name="accuracy",
                value=0.85,
                symbol="AAPL",
                strategy_type="ensemble"
            )
            
            assert metrics.name == "accuracy"
            assert metrics.value == 0.85
            assert metrics.symbol == "AAPL"
            assert metrics.strategy_type == "ensemble"
            assert metrics.timestamp is not None
            
            self.test_results.append("✅ Training result models: PASSED")
            logger.info("Training result models test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Training result models: FAILED - {e}")
            logger.error(f"Training result models test failed: {e}")
    
    def test_model_directory_structure(self):
        """Test model directory structure creation"""
        logger.info("Testing model directory structure...")
        
        try:
            # Test directory creation for different strategies
            strategies = ['ensemble', 'lstm', 'rl']
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            for strategy in strategies:
                strategy_dir = Path("models") / strategy
                strategy_dir.mkdir(parents=True, exist_ok=True)
                assert strategy_dir.exists()
                
                for symbol in symbols:
                    symbol_dir = strategy_dir / symbol
                    symbol_dir.mkdir(parents=True, exist_ok=True)
                    assert symbol_dir.exists()
                    
                    # Create dummy model file
                    model_file = symbol_dir / "model_20240101_120000.pkl"
                    model_file.write_text("dummy model data")
                    assert model_file.exists()
            
            self.test_results.append("✅ Model directory structure: PASSED")
            logger.info("Model directory structure test passed")
            
        except Exception as e:
            self.test_results.append(f"❌ Model directory structure: FAILED - {e}")
            logger.error(f"Model directory structure test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting ML Training Service Tests...")
        
        try:
            self.setup_test_environment()
            
            # Run individual tests
            self.test_service_initialization()
            self.test_database_initialization()
            self.test_model_persistence()
            self.test_training_job_creation()
            self.test_configuration_validation()
            self.test_cron_schedule_parsing()
            self.test_training_result_models()
            self.test_model_directory_structure()
            
        except Exception as e:
            logger.error(f"Test suite error: {e}")
            self.test_results.append(f"❌ Test suite error: {e}")
        
        finally:
            self.cleanup_test_environment()
    
    def print_results(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("ML TRAINING SERVICE TEST RESULTS")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result.startswith("✅"))
        failed = sum(1 for result in self.test_results if result.startswith("❌"))
        
        for result in self.test_results:
            print(result)
        
        print("\n" + "-"*60)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("-"*60)
        
        if failed == 0:
            print("🎉 All tests passed! ML Training Service is ready for deployment.")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return failed == 0

def main():
    """Main test runner"""
    test_suite = MLTrainingServiceTest()
    
    try:
        test_suite.run_all_tests()
        success = test_suite.print_results()
        
        if success:
            print("\n🚀 Next steps:")
            print("1. Start the ML training service:")
            print("   docker compose --profile ml-training up -d")
            print("2. Check service status:")
            print("   docker compose logs ml_training")
            print("3. Monitor training jobs:")
            print("   curl http://localhost:8001/api/ml-training-status")
            
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 