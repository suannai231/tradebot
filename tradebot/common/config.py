"""
Configuration Management Service

Centralized configuration for the trading bot system.
Handles symbol management, limits, and environment settings.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class TradeBotConfig:
    """Centralized configuration for the trading bot."""
    
    def __init__(self):
        # Symbol Configuration
        self.symbol_mode = os.getenv("SYMBOL_MODE", "custom")
        self.custom_symbols = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")
        self.max_symbols = int(os.getenv("MAX_SYMBOLS", "500"))
        self.max_websocket_symbols = int(os.getenv("MAX_WEBSOCKET_SYMBOLS", "30"))
        
        # Database Configuration
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # API Credentials
        self.alpaca_key = os.getenv("ALPACA_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET")
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        
        # Rate Limiting
        self.backfill_batch_size = int(os.getenv("BACKFILL_BATCH_SIZE", "10"))
        self.backfill_concurrent_requests = int(os.getenv("BACKFILL_CONCURRENT_REQUESTS", "3"))
        self.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", "0.5"))
        
        # Market Data Configuration
        self.tick_interval = float(os.getenv("TICK_INTERVAL", "1.0"))
        self.enable_ohlcv = os.getenv("ENABLE_OHLCV", "true").lower() == "true"
        
        # Strategy Configuration
        self.strategy_enabled = os.getenv("STRATEGY_ENABLED", "true").lower() == "true"
        self.sma_short_window = int(os.getenv("SMA_SHORT_WINDOW", "5"))
        self.sma_long_window = int(os.getenv("SMA_LONG_WINDOW", "20"))
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    def get_symbol_limits(self) -> Dict[str, int]:
        """Get symbol limits for different services."""
        return {
            "market_data": self.max_symbols,
            "websocket": self.max_websocket_symbols,
            "backfill": self.max_symbols,
            "strategy": self.max_symbols
        }
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "batch_size": self.backfill_batch_size,
            "concurrent_requests": self.backfill_concurrent_requests,
            "delay_between_requests": self.rate_limit_delay,
            "delay_between_batches": 5.0
        }
    
    def has_alpaca_credentials(self) -> bool:
        """Check if Alpaca credentials are available."""
        return bool(self.alpaca_key and self.alpaca_secret)
    
    def has_polygon_credentials(self) -> bool:
        """Check if Polygon credentials are available."""
        return bool(self.polygon_api_key)
    
    def get_symbol_mode_description(self) -> str:
        """Get human-readable description of symbol mode."""
        descriptions = {
            "all": f"All tradeable US stocks (limited to {self.max_symbols})",
            "popular": "Popular large-cap stocks (~100 symbols)",
            "large": "Large-cap stocks (~50 symbols)",
            "mid": "Mid-cap stocks (~50 symbols)",
            "small": "Small-cap stocks (~100 symbols)",
            "custom": f"Custom symbols: {', '.join(self.custom_symbols)}"
        }
        return descriptions.get(self.symbol_mode, f"Unknown mode: {self.symbol_mode}")
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check symbol mode
        valid_modes = ["all", "popular", "large", "mid", "small", "custom"]
        if self.symbol_mode not in valid_modes:
            issues.append(f"Invalid SYMBOL_MODE: {self.symbol_mode}. Valid options: {valid_modes}")
        
        # Check limits
        if self.max_symbols <= 0:
            issues.append("MAX_SYMBOLS must be positive")
        
        if self.max_websocket_symbols <= 0:
            issues.append("MAX_WEBSOCKET_SYMBOLS must be positive")
        
        if self.max_websocket_symbols > 100:
            issues.append("MAX_WEBSOCKET_SYMBOLS should not exceed 100 (API limitations)")
        
        # Check custom symbols
        if self.symbol_mode == "custom" and not self.custom_symbols:
            issues.append("SYMBOLS must be provided when SYMBOL_MODE=custom")
        
        # Check database URL
        if not self.database_url:
            issues.append("DATABASE_URL is required")
        
        # Check Redis URL
        if not self.redis_url:
            issues.append("REDIS_URL is required")
        
        return issues
    
    def print_configuration(self):
        """Print current configuration summary."""
        print("=== TradingBot Configuration ===")
        print(f"Symbol Mode: {self.symbol_mode}")
        print(f"Description: {self.get_symbol_mode_description()}")
        print(f"Max Symbols: {self.max_symbols}")
        print(f"Max WebSocket Symbols: {self.max_websocket_symbols}")
        print(f"Alpaca Credentials: {'✓' if self.has_alpaca_credentials() else '✗'}")
        print(f"Polygon Credentials: {'✓' if self.has_polygon_credentials() else '✗'}")
        print(f"Database URL: {self.database_url}")
        print(f"Redis URL: {self.redis_url}")
        print(f"Log Level: {self.log_level}")
        
        # Validate and show issues
        issues = self.validate_configuration()
        if issues:
            print("\n⚠️  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ Configuration is valid")
        print("=" * 35)


# Global configuration instance
config = TradeBotConfig()


def get_config() -> TradeBotConfig:
    """Get the global configuration instance."""
    return config 