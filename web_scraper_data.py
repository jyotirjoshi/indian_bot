"""
Web scraper for real-time NSE data - 1 minute intervals
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class NSEWebScraper:
    """Web scraper for NSE data with 1-minute intervals"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with multiple fallbacks"""
        cache_key = f"{symbol}_live"
        
        # Check 2-minute cache
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 120:
                return price
        
        # Try methods in order - NSE/BSE first
        methods = [
            (self._get_nse_scrape_price, "NSE"),
            (self._get_bse_scrape_price, "BSE"),
            (self._get_yahoo_backup, "YF")
        ]
        
        for method, source in methods:
            try:
                price = method(symbol)
                if price and price > 0:
                    self.cache[cache_key] = (price, datetime.now())
                    logger.info(f"[{source}] {symbol}: Rs.{price:.2f}")
                    return price
            except Exception as e:
                logger.debug(f"{source} failed for {symbol}: {e}")
                continue
        
        logger.warning(f"All price sources failed for {symbol}")
        return None
    
    def _get_nse_scrape_price(self, symbol: str) -> Optional[float]:
        """Scrape price directly from NSE website"""
        try:
            url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive'
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price in various NSE elements
                price_selectors = [
                    'span[id*="lastPrice"]',
                    '.quoteLtp',
                    '#lastPrice',
                    'span.number.bold',
                    'div[data-field="lastPrice"]'
                ]
                
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        price_text = price_elem.get_text().strip().replace(',', '')
                        price_match = re.search(r'[\d.]+', price_text)
                        if price_match:
                            return float(price_match.group())
                            
        except Exception as e:
            logger.debug(f"NSE scraping failed for {symbol}: {e}")
        return None
    
    def _get_bse_scrape_price(self, symbol: str) -> Optional[float]:
        """Scrape price from BSE website"""
        try:
            # BSE symbol mapping
            bse_codes = {
                'SBIN': '500112',
                'TCS': '532540', 
                'HDFCBANK': '500180',
                'INFY': '500209',
                'ITC': '500875',
                'RELIANCE': '500325',
                'ICICIBANK': '532174',
                'AXISBANK': '532215'
            }
            
            if symbol not in bse_codes:
                return None
                
            bse_code = bse_codes[symbol]
            url = f"https://www.bseindia.com/stock-share-price/{symbol.lower()}/{bse_code}/"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price in BSE elements
                price_selectors = [
                    'span.CurrentRate',
                    '.quote-price',
                    'strong[id*="idcrval"]',
                    '.price-current',
                    'span.number'
                ]
                
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        price_text = price_elem.get_text().strip().replace(',', '')
                        price_match = re.search(r'[\d.]+', price_text)
                        if price_match:
                            return float(price_match.group())
                            
        except Exception as e:
            logger.debug(f"BSE scraping failed for {symbol}: {e}")
        return None
    
    def _get_yahoo_backup(self, symbol: str) -> Optional[float]:
        """Yahoo Finance as backup with session"""
        try:
            import yfinance as yf
            import requests
            
            # Create session with proper headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            ticker = yf.Ticker(f"{symbol}.NS", session=session)
            
            # Try info first
            try:
                info = ticker.info
                if 'currentPrice' in info and info['currentPrice']:
                    return float(info['currentPrice'])
            except:
                pass
            
            # Try recent history
            data = ticker.history(period="1d", interval="5m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
                
        except Exception as e:
            logger.debug(f"Yahoo backup failed for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data with web scraping"""
        cache_key = f"{symbol}_hist_{days}"
        
        # Check cache (1 hour for historical)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 3600:
                return data
        
        try:
            # Try NSE historical data
            data = self._get_nse_historical(symbol, days)
            if not data.empty:
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[HIST] {symbol}: {len(data)} days")
                return data
            
            # Fallback to Yahoo Finance
            data = self._get_yahoo_historical(symbol, days)
            if not data.empty:
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[YF-HIST] {symbol}: {len(data)} days")
                return data
                
        except Exception as e:
            logger.debug(f"Historical data failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def _get_nse_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical data from NSE"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://www.nseindia.com/api/historical/cm/equity"
            params = {
                'symbol': symbol,
                'series': '["EQ"]',
                'from': start_date.strftime('%d-%m-%Y'),
                'to': end_date.strftime('%d-%m-%Y')
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])
                    df['date'] = pd.to_datetime(df['CH_TIMESTAMP'])
                    df['open'] = df['CH_OPENING_PRICE']
                    df['high'] = df['CH_TRADE_HIGH_PRICE']
                    df['low'] = df['CH_TRADE_LOW_PRICE']
                    df['close'] = df['CH_CLOSING_PRICE']
                    df['volume'] = df['CH_TOT_TRADED_QTY']
                    
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date')
        except:
            pass
        return pd.DataFrame()
    
    def _get_yahoo_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Yahoo Finance historical as backup"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                return data
        except:
            pass
        return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Get intraday data with 1-minute intervals"""
        cache_key = f"{symbol}_intraday_{interval}"
        
        # Check cache (5 minute cache for intraday)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 300:
                return data
        
        try:
            # Try to get intraday from NSE
            data = self._get_nse_intraday(symbol)
            if not data.empty:
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[INTRADAY] {symbol}: {len(data)} points")
                return data
            
            # Fallback to Yahoo Finance
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval=interval)
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                self.cache[cache_key] = (data, datetime.now())
                return data
                
        except Exception as e:
            logger.debug(f"Intraday data failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def _get_nse_intraday(self, symbol: str) -> pd.DataFrame:
        """Get intraday data from NSE"""
        try:
            url = f"https://www.nseindia.com/api/chart-databyindex"
            params = {
                'index': f"{symbol}EQN",
                'indices': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'grapthData' in data:
                    df_data = []
                    for point in data['grapthData']:
                        df_data.append({
                            'datetime': datetime.fromtimestamp(point[0] / 1000),
                            'close': point[1]
                        })
                    
                    if df_data:
                        return pd.DataFrame(df_data)
        except:
            pass
        return pd.DataFrame()
    
    def bulk_price_fetch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple prices with 1-minute intervals"""
        prices = {}
        
        for symbol in symbols:
            price = self.get_live_price(symbol)
            if price:
                prices[symbol] = price
            time.sleep(1)  # 1 second between requests
        
        logger.info(f"[BULK] Fetched {len(prices)}/{len(symbols)} prices")
        return prices
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return {'is_open': False, 'status': 'Weekend'}
        
        # Market hours
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now < market_start:
            return {'is_open': False, 'status': 'Pre-market'}
        elif now > market_end:
            return {'is_open': False, 'status': 'Post-market'}
        else:
            return {'is_open': True, 'status': 'Open'}
    
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
                data = self.get_historical_data(symbol, days)
                
                if not data.empty and len(data) >= 100:
                    data_dict[symbol] = data
                    successful += 1
                    logger.info(f"✅ {symbol}: {len(data)} days of data collected")
                else:
                    logger.warning(f"⚠️ {symbol}: Insufficient data")
                
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
web_scraper = NSEWebScraper()