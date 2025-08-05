"""
Real NSE Web Scraper - No fake data, only real market prices
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import time
import logging
import json
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class RealNSEScraper:
    """Real NSE web scraper with multiple working methods"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        
    def get_nse_quote_api(self, symbol: str) -> Optional[float]:
        """Method 1: NSE Quote API with proper session"""
        try:
            # Create fresh session
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/get-quotes/equity',
                'X-Requested-With': 'XMLHttpRequest'
            })
            
            # Get main page first for cookies
            session.get('https://www.nseindia.com', timeout=15)
            time.sleep(2)
            
            # Get quote
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'priceInfo' in data and 'lastPrice' in data['priceInfo']:
                    return float(data['priceInfo']['lastPrice'])
                    
        except Exception as e:
            logger.debug(f"NSE API failed for {symbol}: {e}")
        return None
    
    def get_nse_live_api(self, symbol: str) -> Optional[float]:
        """Method 2: NSE Live API"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.nseindia.com/'
            }
            
            url = f"https://www.nseindia.com/api/live-analysis-variations?index=gainers"
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    for stock in data['data']:
                        if stock.get('symbol') == symbol:
                            return float(stock.get('ltp', 0))
                            
        except Exception as e:
            logger.debug(f"NSE Live API failed for {symbol}: {e}")
        return None
    
    def get_yahoo_finance(self, symbol: str) -> Optional[float]:
        """Method 3: Yahoo Finance with aggressive rate limiting"""
        try:
            import yfinance as yf
            
            # Wait to avoid rate limiting
            time.sleep(3)
            
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Try multiple approaches
            approaches = [
                lambda: ticker.info.get('currentPrice'),
                lambda: ticker.info.get('regularMarketPrice'),
                lambda: ticker.history(period="1d", interval="5m")['Close'].iloc[-1] if not ticker.history(period="1d", interval="5m").empty else None,
                lambda: ticker.history(period="2d", interval="1d")['Close'].iloc[-1] if not ticker.history(period="2d", interval="1d").empty else None
            ]
            
            for approach in approaches:
                try:
                    price = approach()
                    if price and price > 0:
                        return float(price)
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Yahoo Finance failed for {symbol}: {e}")
        return None
    
    def get_investing_com(self, symbol: str) -> Optional[float]:
        """Method 4: Investing.com scraping"""
        try:
            # Investing.com symbol mapping
            investing_symbols = {
                'SBIN': 'state-bank-of-india',
                'TCS': 'tata-consultancy-services',
                'HDFCBANK': 'hdfc-bank-ltd',
                'INFY': 'infosys-ltd',
                'ITC': 'itc-ltd',
                'RELIANCE': 'reliance-industries',
                'ICICIBANK': 'icici-bank-ltd',
                'AXISBANK': 'axis-bank-ltd'
            }
            
            if symbol not in investing_symbols:
                return None
            
            url = f"https://in.investing.com/equities/{investing_symbols[symbol]}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price elements
                selectors = [
                    'span[data-test="instrument-price-last"]',
                    '.text-2xl',
                    '.instrument-price_last__KQzyA',
                    'span.last-price-value'
                ]
                
                for selector in selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        text = elem.get_text().strip().replace(',', '')
                        match = re.search(r'[\d.]+', text)
                        if match:
                            return float(match.group())
                            
        except Exception as e:
            logger.debug(f"Investing.com failed for {symbol}: {e}")
        return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price trying all real methods"""
        cache_key = f"{symbol}_price"
        
        # Check 30-second cache
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 30:
                return price
        
        # Try all methods in order
        methods = [
            (self.get_yahoo_finance, "YAHOO"),
            (self.get_nse_quote_api, "NSE_API"),
            (self.get_nse_live_api, "NSE_LIVE"),
            (self.get_investing_com, "INVESTING")
        ]
        
        for method, source in methods:
            try:
                price = method(symbol)
                if price and price > 0:
                    self.cache[cache_key] = (price, datetime.now())
                    logger.info(f"[{source}] {symbol}: Rs.{price:.2f}")
                    return price
            except Exception as e:
                logger.debug(f"{source} method failed for {symbol}: {e}")
                continue
        
        logger.warning(f"All real data sources failed for {symbol}")
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                logger.info(f"[HIST] {symbol}: {len(data)} days")
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
real_nse_scraper = RealNSEScraper()