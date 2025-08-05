"""
Real Data Provider with Multiple Sources
Ensures continuous real market data without rate limiting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json
import random
import os

logger = logging.getLogger(__name__)

class RealMarketDataProvider:
    """Enhanced real data provider with multiple sources and smart caching"""
    
    def __init__(self):
        self.cache = {}
        self.last_request_time = {}
        self.request_count = 0
        self.max_requests_per_minute = 10
        
    def _rate_limit_check(self, symbol: str):
        """Aggressive rate limiting to avoid blocks"""
        current_time = time.time()
        
        # Global rate limiting - max 3 requests per minute
        if self.request_count >= 3:
            if current_time - getattr(self, 'last_reset_time', 0) < 60:
                sleep_time = 60 - (current_time - getattr(self, 'last_reset_time', 0))
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_reset_time = current_time
        
        # Per-symbol rate limiting - minimum 20 seconds
        if symbol in self.last_request_time:
            time_since_last = current_time - self.last_request_time[symbol]
            if time_since_last < 20:
                time.sleep(20 - time_since_last)
        
        self.last_request_time[symbol] = time.time()
        self.request_count += 1
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Alias for get_live_price for compatibility"""
        return self.get_live_price(symbol)
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with aggressive rate limiting"""
        cache_key = f"{symbol}_live_price"
        
        # Check cache (10 minute cache to reduce API calls)
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 600:
                return price
        
        # Aggressive rate limiting - 10 second delay
        time.sleep(10)
        
        try:
            # Create new session for each request
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            ticker = yf.Ticker(f"{symbol}.NS", session=session)
            
            # Try recent history with minimal data
            data = ticker.history(period="1d", interval="5m")
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                self.cache[cache_key] = (price, datetime.now())
                logger.info(f"[LIVE] {symbol}: Rs.{price:.2f}")
                return price
                
        except Exception as e:
            logger.warning(f"Live price failed for {symbol}: {e}")
            
            # Use cached price if available
            if cache_key in self.cache:
                price, timestamp = self.cache[cache_key]
                logger.info(f"[CACHED] {symbol}: Rs.{price:.2f}")
                return price
        
        return None
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Get intraday data with smart caching"""
        cache_key = f"{symbol}_intraday_{interval}"
        
        # Check cache (10 minute cache for intraday data)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 600:
                return data
        
        self._rate_limit_check(symbol)
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get today's intraday data
            data = ticker.history(period="1d", interval=interval)
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                
                # Filter only today's data
                today = datetime.now().date()
                if 'datetime' in data.columns:
                    data['date'] = pd.to_datetime(data['datetime']).dt.date
                    data = data[data['date'] == today]
                
                if not data.empty:
                    self.cache[cache_key] = (data, datetime.now())
                    logger.info(f"[INTRADAY] {symbol}: {len(data)} {interval} candles")
                    return data
                    
        except Exception as e:
            logger.debug(f"Intraday data failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data with extended caching"""
        cache_key = f"{symbol}_hist_{days}"
        
        # Check cache (24 hour cache for historical data)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 86400:
                return data
        
        # Aggressive rate limiting - 15 second delay
        time.sleep(15)
        
        try:
            # Create new session
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            ticker = yf.Ticker(f"{symbol}.NS", session=session)
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[HIST] {symbol}: {len(data)} days")
                return data
                
        except Exception as e:
            logger.warning(f"Historical data failed for {symbol}: {e}")
            
            # Try to use cached training data as fallback
            try:
                training_data = self.load_training_data()
                if symbol in training_data:
                    data = training_data[symbol].tail(days)
                    logger.info(f"[FALLBACK] Using training data for {symbol}")
                    return data
            except:
                pass
        
        return pd.DataFrame()
    
    def bulk_price_fetch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple prices efficiently"""
        prices = {}
        
        # Group symbols to reduce API calls
        symbol_string = " ".join([f"{s}.NS" for s in symbols])
        
        try:
            # Use yfinance download for bulk fetching
            data = yf.download(symbol_string, period="1d", interval="1m", group_by='ticker')
            
            for symbol in symbols:
                try:
                    symbol_data = data[f"{symbol}.NS"]
                    if not symbol_data.empty:
                        price = float(symbol_data['Close'].dropna().iloc[-1])
                        prices[symbol] = price
                        
                        # Cache the price
                        cache_key = f"{symbol}_live_price"
                        self.cache[cache_key] = (price, datetime.now())
                        
                except Exception as e:
                    logger.debug(f"Bulk price failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.debug(f"Bulk fetch failed: {e}")
            
            # Fallback to individual fetching with delays
            for symbol in symbols:
                price = self.get_live_price(symbol)
                if price:
                    prices[symbol] = price
                time.sleep(1)  # Small delay between individual requests
        
        logger.info(f"[BULK] Fetched prices for {len(prices)}/{len(symbols)} symbols")
        return prices
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return {'is_open': False, 'status': 'Weekend', 'next_open': 'Monday 9:15 AM'}
        
        # Market hours check
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now < market_start:
            return {'is_open': False, 'status': 'Pre-market', 'next_open': 'Today 9:15 AM'}
        elif now > market_end:
            return {'is_open': False, 'status': 'Post-market', 'next_open': 'Tomorrow 9:15 AM'}
        else:
            return {'is_open': True, 'status': 'Open', 'closes_at': '3:30 PM'}
    
    def clear_cache(self):
        """Clear old cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (data, timestamp) in self.cache.items():
            # Remove entries older than 1 hour
            if (current_time - timestamp).seconds > 3600:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            logger.info(f"Cleared {len(keys_to_remove)} old cache entries")
    
    def load_training_data(self, path: str = "training_data") -> Dict[str, pd.DataFrame]:
        """Load training data from disk"""
        import os
        try:
            if not os.path.exists(path):
                return {}
            
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return {}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if data is recent
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            if (datetime.now() - last_updated).days > 7:
                return {}
            
            # Load data files
            data_dict = {}
            for symbol in metadata['symbols']:
                file_path = os.path.join(path, f"{symbol}_data.csv")
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    data['date'] = pd.to_datetime(data['date'])
                    data_dict[symbol] = data
            
            logger.info(f"Loaded cached training data for {len(data_dict)} symbols")
            return data_dict
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return {}
    
    def bulk_fetch_training_data(self, symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """Bulk fetch training data for multiple symbols"""
        logger.info(f"Fetching {days} days of training data for {len(symbols)} symbols...")
        
        data_dict = {}
        successful = 0
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Fetching data for {symbol}...")
                
                self._rate_limit_check(symbol)
                
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period=f"{days}d", interval="1d")
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    data.columns = [col.lower() for col in data.columns]
                    
                    if len(data) >= 100:
                        data_dict[symbol] = data
                        successful += 1
                        logger.info(f"✅ {symbol}: {len(data)} days of data collected")
                    else:
                        logger.warning(f"⚠️ {symbol}: Insufficient data ({len(data)} days)")
                else:
                    logger.error(f"❌ {symbol}: Failed to fetch data")
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Data collection failed - {e}")
        
        logger.info(f"Training data collection complete: {successful}/{len(symbols)} successful")
        return data_dict
    
    def save_training_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "training_data"):
        """Save training data to disk"""
        import os
        try:
            os.makedirs(path, exist_ok=True)
            
            for symbol, data in data_dict.items():
                file_path = os.path.join(path, f"{symbol}_data.csv")
                data.to_csv(file_path, index=False)
                logger.info(f"Saved {symbol} data to {file_path}")
            
            # Save metadata
            metadata = {
                'symbols': list(data_dict.keys()),
                'data_points': {symbol: len(data) for symbol, data in data_dict.items()},
                'last_updated': datetime.now().isoformat(),
                'total_symbols': len(data_dict)
            }
            
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training data saved to {path} directory")
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

# Global instance
real_data_provider = RealMarketDataProvider()