"""
Working Stock Price Scraper - Gets real data
"""

import requests
import yfinance as yf
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Optional, Dict
import json

logger = logging.getLogger(__name__)

class WorkingScraper:
    """Simple working scraper that actually gets data"""
    
    def __init__(self):
        self.cache = {}
    
    def get_yahoo_price(self, symbol: str) -> Optional[float]:
        """Get price from Yahoo Finance - most reliable method"""
        try:
            # Simple yfinance call
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get recent data
            data = ticker.history(period="5d", interval="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
        except Exception as e:
            logger.debug(f"Yahoo failed for {symbol}: {e}")
        return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with caching"""
        cache_key = f"{symbol}_price"
        
        # Check 1-minute cache
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 60:
                return price
        
        # Get price from Yahoo
        price = self.get_yahoo_price(symbol)
        if price and price > 0:
            self.cache[cache_key] = (price, datetime.now())
            logger.info(f"[YAHOO] {symbol}: Rs.{price:.2f}")
            return price
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                return data
        except Exception as e:
            logger.debug(f"Historical data failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        now = datetime.now()
        
        if now.weekday() >= 5:
            return {'is_open': False, 'status': 'Weekend'}
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now < market_start:
            return {'is_open': False, 'status': 'Pre-market'}
        elif now > market_end:
            return {'is_open': False, 'status': 'Post-market'}
        else:
            return {'is_open': True, 'status': 'Open'}
    
    # Required compatibility methods
    def load_training_data(self, path: str = "training_data"):
        import os
        try:
            if not os.path.exists(path):
                return {}
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return {}
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            data_dict = {}
            for symbol in metadata['symbols']:
                file_path = os.path.join(path, f"{symbol}_data.csv")
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    data['date'] = pd.to_datetime(data['date'])
                    data_dict[symbol] = data
            return data_dict
        except:
            return {}
    
    def bulk_fetch_training_data(self, symbols: list, days: int = 365):
        data_dict = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, days)
            if not data.empty:
                data_dict[symbol] = data
            time.sleep(1)
        return data_dict
    
    def save_training_data(self, data_dict, path: str = "training_data"):
        import os
        try:
            os.makedirs(path, exist_ok=True)
            for symbol, data in data_dict.items():
                file_path = os.path.join(path, f"{symbol}_data.csv")
                data.to_csv(file_path, index=False)
            metadata = {
                'symbols': list(data_dict.keys()),
                'last_updated': datetime.now().isoformat(),
                'total_symbols': len(data_dict)
            }
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

# Global instance
working_scraper = WorkingScraper()