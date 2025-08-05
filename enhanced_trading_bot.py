"""
Enhanced Intraday Trading Bot with Real Data and ML
Capital: ₹50,000
Features: Intraday trading, real-time data, ML signals, auto position closure
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass
import json
import threading
import os
import json
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Import ML components and aggressive strategies
from ml_components import MLTradingEngine
from aggressive_strategies import AggressiveStrategies, ProfitMaximizer
from final_working_scraper import final_working_scraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    price: float
    timestamp: datetime
    charges: float
    net_amount: float

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    target: float
    entry_time: datetime

class RealDataProvider:
    """Real-time data provider using multiple sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache
        
    def get_nse_price(self, symbol: str) -> Optional[float]:
        """Get real NSE price via web scraping"""
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data['priceInfo']['lastPrice'])
        except Exception as e:
            logger.debug(f"NSE API failed for {symbol}: {e}")
        return None
    
    def get_yahoo_price(self, symbol: str) -> Optional[float]:
        """Get price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.debug(f"Yahoo Finance failed for {symbol}: {e}")
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with extended caching"""
        cache_key = f"{symbol}_price"
        
        # Check cache (extended cache time)
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 300:  # 5 minute cache
                return price
        
        # Rate limiting
        time.sleep(2)
        
        # Try Yahoo Finance with retry
        for attempt in range(2):
            try:
                price = self.get_yahoo_price(symbol)
                if price:
                    self.cache[cache_key] = (price, datetime.now())
                    logger.info(f"[PRICE] {symbol}: Rs.{price:.2f}")
                    return price
                time.sleep(3)
            except:
                time.sleep(5)
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data with aggressive rate limiting"""
        cache_key = f"{symbol}_hist_{days}"
        
        # Check cache first (extended cache)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 14400:  # 4 hour cache
                return data
        
        # Aggressive rate limiting
        time.sleep(5)  # 5 second delay
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                
                # Cache the data
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[DATA] Fetched {len(data)} days of historical data for {symbol}")
                return data
            else:
                # Try with cached training data as fallback
                training_data = self.load_training_data()
                if symbol in training_data:
                    data = training_data[symbol].tail(days)
                    logger.info(f"[FALLBACK] Using cached training data for {symbol}")
                    return data
                    
        except Exception as e:
            logger.warning(f"Historical data failed for {symbol}: {e}")
            
            # Fallback to cached training data
            try:
                training_data = self.load_training_data()
                if symbol in training_data:
                    data = training_data[symbol].tail(days)
                    logger.info(f"[FALLBACK] Using cached training data for {symbol}")
                    return data
            except:
                pass
        
        return pd.DataFrame()
    
    def bulk_fetch_training_data(self, symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """Bulk fetch training data for multiple symbols"""
        logger.info(f"Fetching {days} days of training data for {len(symbols)} symbols...")
        
        data_dict = {}
        successful = 0
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Fetching data for {symbol}...")
                
                # Get data with retry logic
                data = None
                for attempt in range(3):  # 3 attempts
                    try:
                        ticker = yf.Ticker(f"{symbol}.NS")
                        data = ticker.history(period=f"{days}d", interval="1d")
                        
                        if not data.empty:
                            break
                        else:
                            logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}")
                            time.sleep(2)
                            
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                        time.sleep(2)
                
                if data is not None and not data.empty:
                    # Clean and format data
                    data.reset_index(inplace=True)
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Validate data quality
                    if len(data) >= 100:  # Minimum 100 days for training
                        data_dict[symbol] = data
                        successful += 1
                        logger.info(f"✅ {symbol}: {len(data)} days of data collected")
                    else:
                        logger.warning(f"⚠️ {symbol}: Insufficient data ({len(data)} days)")
                else:
                    logger.error(f"❌ {symbol}: Failed to fetch data")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Data collection failed - {e}")
        
        logger.info(f"Training data collection complete: {successful}/{len(symbols)} successful")
        return data_dict
    
    def save_training_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "training_data"):
        """Save training data to disk for faster loading"""
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
    
    def load_training_data(self, path: str = "training_data") -> Dict[str, pd.DataFrame]:
        """Load training data from disk"""
        try:
            if not os.path.exists(path):
                return {}
            
            # Load metadata
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return {}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if data is recent (less than 7 days old)
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            if (datetime.now() - last_updated).days > 7:
                logger.info("Cached training data is older than 7 days, will fetch fresh data")
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

class CommissionCalculator:
    """Calculate realistic trading charges for Indian markets"""
    
    def __init__(self, broker_type: str = "discount"):
        self.broker_type = broker_type
        
        # Zerodha-like charges (discount broker)
        if broker_type == "discount":
            self.brokerage_rate = 0.0003  # 0.03% or ₹20 per trade, whichever is lower
            self.max_brokerage = 20.0
        else:  # Full-service broker
            self.brokerage_rate = 0.005  # 0.5%
            self.max_brokerage = 1000.0
        
        # Fixed charges
        self.stt_rate = 0.001  # 0.1% on sell side
        self.transaction_charges = 0.00345  # 0.345% NSE
        self.gst_rate = 0.18  # 18% on brokerage + transaction charges
        self.sebi_charges = 0.000001  # ₹1 per crore
        self.stamp_duty = 0.00003  # 0.003% on buy side
    
    def calculate_charges(self, action: str, quantity: int, price: float) -> Dict[str, float]:
        """Calculate all trading charges"""
        turnover = quantity * price
        
        # Brokerage
        brokerage = min(turnover * self.brokerage_rate, self.max_brokerage)
        
        # STT (only on sell)
        stt = turnover * self.stt_rate if action == "SELL" else 0
        
        # Transaction charges
        transaction_charges = turnover * self.transaction_charges
        
        # GST on brokerage + transaction charges
        gst = (brokerage + transaction_charges) * self.gst_rate
        
        # SEBI charges
        sebi_charges = turnover * self.sebi_charges
        
        # Stamp duty (only on buy)
        stamp_duty = turnover * self.stamp_duty if action == "BUY" else 0
        
        total_charges = brokerage + stt + transaction_charges + gst + sebi_charges + stamp_duty
        
        return {
            'brokerage': brokerage,
            'stt': stt,
            'transaction_charges': transaction_charges,
            'gst': gst,
            'sebi_charges': sebi_charges,
            'stamp_duty': stamp_duty,
            'total_charges': total_charges,
            'net_amount': turnover + total_charges if action == "BUY" else turnover - total_charges
        }

class TechnicalAnalyzer:
    """Technical analysis for trading signals"""
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals"""
        if len(data) < 30:
            return {'signal': 0, 'strength': 0, 'strategy': 'insufficient_data'}
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        macd, macd_signal, macd_hist = self.calculate_macd(data)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
        
        current_price = data['close'].iloc[-1]
        signals = []
        
        # RSI signals
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            signals.append(('RSI_OVERSOLD', 0.8))
        elif current_rsi > 70:
            signals.append(('RSI_OVERBOUGHT', -0.8))
        
        # MACD signals
        if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
            signals.append(('MACD_BULLISH_CROSSOVER', 0.7))
        elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
            signals.append(('MACD_BEARISH_CROSSOVER', -0.7))
        
        # Bollinger Bands signals
        if current_price <= bb_lower.iloc[-1]:
            signals.append(('BB_OVERSOLD', 0.6))
        elif current_price >= bb_upper.iloc[-1]:
            signals.append(('BB_OVERBOUGHT', -0.6))
        
        # Combine signals
        if not signals:
            return {'signal': 0, 'strength': 0, 'strategy': 'no_signal'}
        
        total_signal = sum(signal[1] for signal in signals)
        avg_strength = abs(total_signal) / len(signals)
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': avg_strength,
            'strategy': ', '.join([s[0] for s in signals])
        }

class RiskManager:
    """Risk management for position sizing and stop losses"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.max_risk_per_trade = 0.025  # 2.5% risk per trade for maximum profit
        self.max_portfolio_risk = 0.08  # 8% total portfolio risk
        self.max_positions = 5  # Maximum 5 positions for multiple trades
    
    def calculate_position_size(self, price: float, stop_loss: float, current_positions: int) -> int:
        """Calculate position size based on risk management"""
        if current_positions >= self.max_positions:
            return 0
        
        # Risk amount per trade
        risk_amount = self.capital * self.max_risk_per_trade
        
        # Risk per share
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate quantity
        quantity = int(risk_amount / risk_per_share)
        
        # Ensure minimum viable quantity and maximum allocation
        max_allocation = self.capital * 0.35  # Max 35% per trade for Rs.50000
        max_quantity = int(max_allocation / price)
        
        quantity = min(quantity, max_quantity)
        
        # Minimum trade size for profit maximization
        if quantity * price < 8000:  # Minimum Rs.8000 trade for meaningful profit
            return 0
        
        return quantity
    
    def calculate_stop_loss(self, price: float, action: str, volatility: float = 0.02) -> float:
        """Calculate stop loss for aggressive intraday trading"""
        stop_distance = price * max(volatility, 0.02)  # 2% stop for higher profit potential
        
        if action == "BUY":
            return price - stop_distance
        else:
            return price + stop_distance
    
    def calculate_target(self, price: float, stop_loss: float, action: str, risk_reward: float = 3.5) -> float:
        """Calculate aggressive target for maximum profit"""
        risk = abs(price - stop_loss)
        reward = risk * risk_reward  # 3.5:1 risk-reward for maximum profit
        
        if action == "BUY":
            return price + reward
        else:
            return price - reward

class MarketAnalyzer:
    """Pre and post market analysis for swing trading"""
    
    def __init__(self, data_provider, watchlist):
        self.data_provider = data_provider
        self.watchlist = watchlist
    
    def pre_market_analysis(self) -> Dict:
        """Analyze market before opening"""
        analysis = {
            'market_sentiment': 'NEUTRAL',
            'strong_stocks': [],
            'weak_stocks': [],
            'gap_ups': [],
            'gap_downs': [],
            'recommendations': []
        }
        
        logger.info("=== PRE-MARKET ANALYSIS ===")
        
        for symbol in self.watchlist:
            try:
                # Get 5-day data for trend analysis
                data = self.data_provider.get_historical_data(symbol, 5)
                if data.empty:
                    continue
                
                current_price = self.data_provider.get_live_price(symbol)
                if not current_price:
                    continue
                
                prev_close = data['close'].iloc[-2] if len(data) > 1 else data['close'].iloc[-1]
                gap_pct = ((current_price - prev_close) / prev_close) * 100
                
                # Identify gaps
                if gap_pct > 2:
                    analysis['gap_ups'].append({'symbol': symbol, 'gap': gap_pct})
                elif gap_pct < -2:
                    analysis['gap_downs'].append({'symbol': symbol, 'gap': gap_pct})
                
                # 5-day trend strength
                if len(data) >= 5:
                    trend_strength = ((data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]) * 100
                    if trend_strength > 5:
                        analysis['strong_stocks'].append({'symbol': symbol, 'strength': trend_strength})
                    elif trend_strength < -5:
                        analysis['weak_stocks'].append({'symbol': symbol, 'strength': trend_strength})
                
            except Exception as e:
                logger.debug(f"Pre-market analysis failed for {symbol}: {e}")
        
        # Generate recommendations
        if len(analysis['strong_stocks']) > len(analysis['weak_stocks']):
            analysis['market_sentiment'] = 'BULLISH'
            analysis['recommendations'].append('Focus on LONG positions')
        elif len(analysis['weak_stocks']) > len(analysis['strong_stocks']):
            analysis['market_sentiment'] = 'BEARISH'
            analysis['recommendations'].append('Consider SHORT positions or wait')
        
        if analysis['gap_ups']:
            analysis['recommendations'].append(f"Watch gap-up stocks: {[s['symbol'] for s in analysis['gap_ups'][:3]]}")
        
        # Log analysis
        logger.info(f"Market Sentiment: {analysis['market_sentiment']}")
        logger.info(f"Strong Stocks: {len(analysis['strong_stocks'])}, Weak Stocks: {len(analysis['weak_stocks'])}")
        logger.info(f"Gap Ups: {len(analysis['gap_ups'])}, Gap Downs: {len(analysis['gap_downs'])}")
        
        for rec in analysis['recommendations']:
            logger.info(f"Recommendation: {rec}")
        
        logger.info("=== PRE-MARKET ANALYSIS COMPLETE ===")
        
        # Save analysis to database
        self._save_analysis('PRE_MARKET', analysis)
        return analysis
    
    def post_market_analysis(self, positions: Dict, trades: List) -> Dict:
        """Analyze market after closing"""
        analysis = {
            'daily_performance': {},
            'position_analysis': {},
            'market_review': '',
            'tomorrow_plan': []
        }
        
        logger.info("=== POST-MARKET ANALYSIS ===")
        
        # Analyze daily performance
        daily_pnl = sum(trade.charges for trade in trades if trade.timestamp.date() == datetime.now().date())
        winning_trades = len([t for t in trades if hasattr(t, 'pnl') and getattr(t, 'pnl', 0) > 0])
        total_trades = len([t for t in trades if t.timestamp.date() == datetime.now().date()])
        
        analysis['daily_performance'] = {
            'total_pnl': daily_pnl,
            'trades_count': total_trades,
            'win_rate': (winning_trades / max(total_trades, 1)) * 100
        }
        
        # Analyze current positions
        for symbol, position in positions.items():
            pnl_pct = (position.unrealized_pnl / (position.avg_price * position.quantity)) * 100
            analysis['position_analysis'][symbol] = {
                'pnl_pct': pnl_pct,
                'status': 'WINNING' if pnl_pct > 0 else 'LOSING',
                'days_held': (datetime.now() - position.entry_time).days
            }
        
        # Generate tomorrow's plan
        if analysis['daily_performance']['win_rate'] > 60:
            analysis['tomorrow_plan'].append('Continue current strategy')
        else:
            analysis['tomorrow_plan'].append('Review and adjust position sizes')
        
        if len(positions) < 2:
            analysis['tomorrow_plan'].append('Look for new opportunities')
        elif len(positions) >= 3:
            analysis['tomorrow_plan'].append('Focus on managing existing positions')
        
        # Log analysis
        logger.info(f"Daily P&L: Rs.{daily_pnl:.2f}")
        logger.info(f"Trades: {total_trades}, Win Rate: {analysis['daily_performance']['win_rate']:.1f}%")
        logger.info(f"Active Positions: {len(positions)}")
        
        for plan in analysis['tomorrow_plan']:
            logger.info(f"Tomorrow: {plan}")
        
        logger.info("=== POST-MARKET ANALYSIS COMPLETE ===")
        
        # Save analysis to database
        self._save_analysis('POST_MARKET', analysis)
        return analysis
    
    def _save_analysis(self, analysis_type: str, analysis: Dict):
        """Save analysis to database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            if analysis_type == 'PRE_MARKET':
                cursor.execute('''
                INSERT INTO market_analysis (date, type, sentiment, strong_stocks, weak_stocks, recommendations, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().date(),
                    analysis_type,
                    analysis.get('market_sentiment', ''),
                    str([s['symbol'] for s in analysis.get('strong_stocks', [])]),
                    str([s['symbol'] for s in analysis.get('weak_stocks', [])]),
                    str(analysis.get('recommendations', [])),
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

class EnhancedTradingBot:
    """Enhanced Trading Bot with Real Data and Realistic Environment"""
    
    def __init__(self, capital: float = 50000):
        self.capital = capital
        self.available_cash = capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Trading parameters (initialize first)
        self.watchlist = ["SBIN", "TCS", "INFY", "ITC", "RELIANCE"]
        self.is_running = False
        
        # Initialize components with final working scraper
        self.data_provider = final_working_scraper
        self.commission_calc = CommissionCalculator("discount")
        self.technical_analyzer = TechnicalAnalyzer()
        self.risk_manager = RiskManager(capital)
        self.market_analyzer = MarketAnalyzer(self.data_provider, self.watchlist)
        self.ml_engine = MLTradingEngine(capital, self.watchlist)
        self.aggressive_strategies = AggressiveStrategies()
        self.profit_maximizer = ProfitMaximizer(capital)
        
        # ML training flag
        self.ml_trained = False
        
        # Database setup
        self.setup_database()
        
        logger.info(f"Enhanced Intraday ML Trading Bot initialized with Rs.{capital:,.2f} capital")
        
        # Initialize ML models
        self.initialize_ml_models()
    
    def initialize_ml_models(self):
        """Initialize and train ML models with automatic data fetching"""
        logger.info("🧠 Initializing ML models...")
        
        # Try to load pre-trained models
        if os.path.exists('ml_models') and self.ml_engine.load_models('ml_models'):
            logger.info("✅ Pre-trained ML models loaded successfully")
            self.ml_trained = True
            return
        
        logger.info("🔄 No pre-trained models found, starting training process...")
        
        # Step 1: Try to load cached training data
        data_dict = self.data_provider.load_training_data()
        
        # Step 2: If no cached data, fetch fresh data
        if not data_dict:
            logger.info("📊 Fetching fresh training data from market...")
            data_dict = self.data_provider.bulk_fetch_training_data(self.watchlist, days=365)
            
            # Save the fetched data for future use
            if data_dict:
                self.data_provider.save_training_data(data_dict)
        else:
            logger.info("📁 Using cached training data")
        
        # Step 3: Validate data quality
        if not data_dict:
            logger.error("❌ No training data available, ML features disabled")
            self.ml_trained = False
            return
        
        # Log data summary
        total_data_points = sum(len(data) for data in data_dict.values())
        logger.info(f"📈 Training data summary:")
        logger.info(f"   Symbols: {len(data_dict)}")
        logger.info(f"   Total data points: {total_data_points:,}")
        logger.info(f"   Average days per symbol: {total_data_points // len(data_dict)}")
        
        # Step 4: Train ML models
        logger.info("🏋️ Training ML models (this may take 2-5 minutes)...")
        
        try:
            self.ml_trained = self.ml_engine.train_models(data_dict)
            
            if self.ml_trained:
                # Save trained models
                logger.info("💾 Saving trained models...")
                self.ml_engine.save_models('ml_models')
                logger.info("✅ ML models trained and saved successfully!")
                
                # Log model capabilities
                logger.info("🎯 ML capabilities enabled:")
                logger.info("   ✓ LSTM price prediction")
                logger.info("   ✓ Random Forest/XGBoost classification")
                logger.info("   ✓ Reinforcement Learning position sizing")
                logger.info("   ✓ News sentiment analysis")
                logger.info("   ✓ Ensemble signal fusion")
            else:
                logger.warning("⚠️ ML model training failed, using traditional signals only")
                
        except Exception as e:
            logger.error(f"❌ ML training error: {e}")
            self.ml_trained = False
    
    def setup_database(self):
        """Setup SQLite database for trade tracking"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            action TEXT,
            quantity INTEGER,
            price REAL,
            charges REAL,
            net_amount REAL,
            timestamp DATETIME,
            pnl REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            quantity INTEGER,
            avg_price REAL,
            current_price REAL,
            unrealized_pnl REAL,
            stop_loss REAL,
            target REAL,
            entry_time DATETIME
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            type TEXT,
            sentiment TEXT,
            strong_stocks TEXT,
            weak_stocks TEXT,
            recommendations TEXT,
            timestamp DATETIME
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: Trade, pnl: float = 0):
        """Save trade to database"""
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (symbol, action, quantity, price, charges, net_amount, timestamp, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade.symbol, trade.action, trade.quantity, trade.price, 
              trade.charges, trade.net_amount, trade.timestamp, pnl))
        
        conn.commit()
        conn.close()
    
    def is_market_open(self) -> bool:
        """Check if market is open using enhanced provider"""
        market_status = self.data_provider.get_market_status()
        return market_status['is_open']
    
    def scan_opportunities(self) -> List[Dict]:
        """Scan for trading opportunities"""
        opportunities = []
        
        for symbol in self.watchlist:
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            try:
                # Get current price using web scraper
                current_price = self.data_provider.get_live_price(symbol)
                if not current_price:
                    logger.warning(f"No price data for {symbol}")
                    continue
                
                # Get historical data with web scraper
                hist_data = self.data_provider.get_historical_data(symbol, 90)
                if hist_data.empty:
                    logger.warning(f"No historical data for {symbol}")
                    continue
                
                # Generate technical signals
                technical_signals = self.technical_analyzer.generate_signals(hist_data)
                
                # Generate aggressive strategy signals
                aggressive_signals = self.aggressive_strategies.get_combined_signals(hist_data, current_price)
                
                # Get ML signals if trained
                ml_signals = {'signal': 0, 'confidence': 0.0}
                if self.ml_trained:
                    portfolio_value = self.available_cash + sum(pos.current_price * pos.quantity for pos in self.positions.values())
                    ml_signals = self.ml_engine.get_ml_signals(
                        symbol, hist_data, current_price, portfolio_value, len(self.positions)
                    )
                
                # Combine all signals for maximum profit
                signals_to_combine = [technical_signals]
                
                if aggressive_signals['strength'] > 0.5:
                    signals_to_combine.append(aggressive_signals)
                
                if self.ml_trained and ml_signals.get('confidence', 0) > 0.6:
                    signals_to_combine.append({
                        'signal': ml_signals['signal'],
                        'strength': ml_signals['confidence']
                    })
                
                # Weighted combination of all signals
                total_signal = sum(s['signal'] * s['strength'] for s in signals_to_combine)
                total_weight = sum(s['strength'] for s in signals_to_combine)
                
                if total_weight > 0:
                    final_signal = total_signal / total_weight
                    signal_strength = total_weight / len(signals_to_combine)
                    strategy_name = f"Multi-Strategy: {', '.join([s.get('strategy', 'Unknown') for s in signals_to_combine])}"
                else:
                    final_signal = 0
                    signal_strength = 0
                    strategy_name = "No Signal"
                
                # Lower threshold for more trading opportunities
                if final_signal > 0.1 and signal_strength > 0.3:
                    action = "BUY"
                    
                    # Calculate aggressive position sizing
                    volatility = hist_data['close'].pct_change().std() * np.sqrt(252)
                    quantity = self.profit_maximizer.calculate_aggressive_position_size(
                        current_price, signal_strength, volatility
                    )
                    
                    # Calculate dynamic stop loss
                    stop_loss = self.profit_maximizer.dynamic_stop_loss(
                        current_price, current_price, action, volatility
                    )
                    
                    if quantity > 0:
                        # Calculate multiple profit targets
                        profit_levels = self.profit_maximizer.profit_taking_levels(
                            current_price, action, signal_strength
                        )
                        target = profit_levels[1]['price'] if len(profit_levels) > 1 else current_price * 1.05
                        
                        opportunities.append({
                            'symbol': symbol,
                            'action': action,
                            'price': current_price,
                            'quantity': quantity,
                            'stop_loss': stop_loss,
                            'target': target,
                            'signal_strength': signal_strength,
                            'strategy': strategy_name
                        })
                        
                        logger.info(f"[OPPORTUNITY] {action} {symbol} @ Rs.{current_price:.2f} "
                                  f"(Qty: {quantity}, SL: Rs.{stop_loss:.2f}, Target: Rs.{target:.2f})")
                        logger.info(f"Strategy: {strategy_name}, Strength: {signal_strength:.2f}")
                        
                        if self.ml_trained and 'components' in ml_signals:
                            components = ml_signals['components']
                            logger.info(f"ML Components - Tech: {components.get('technical', 0):.2f}, "
                                      f"LSTM: {components.get('lstm', 0):.2f}, "
                                      f"Classifier: {components.get('ml_classifier', 0):.2f}, "
                                      f"Sentiment: {components.get('sentiment', 0):.2f}")
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return opportunities
    
    def execute_trade(self, opportunity: Dict) -> bool:
        """Execute a trade"""
        try:
            symbol = opportunity['symbol']
            action = opportunity['action']
            price = opportunity['price']
            quantity = opportunity['quantity']
            
            # Calculate charges
            charges_data = self.commission_calc.calculate_charges(action, quantity, price)
            total_charges = charges_data['total_charges']
            net_amount = charges_data['net_amount']
            
            # Check if we have enough cash
            if action == "BUY" and net_amount > self.available_cash:
                logger.warning(f"Insufficient cash for {symbol}. Required: Rs.{net_amount:.2f}, Available: Rs.{self.available_cash:.2f}")
                return False
            
            # Create trade
            trade = Trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                charges=total_charges,
                net_amount=net_amount
            )
            
            # Update cash
            if action == "BUY":
                self.available_cash -= net_amount
            else:
                self.available_cash += net_amount
            
            # Create/update position
            if action == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    unrealized_pnl=0,
                    stop_loss=opportunity['stop_loss'],
                    target=opportunity['target'],
                    entry_time=datetime.now()
                )
            
            # Save trade
            self.trades.append(trade)
            self.save_trade(trade)
            
            logger.info(f"[EXECUTED] {action} {quantity} {symbol} @ Rs.{price:.2f} "
                       f"(Charges: Rs.{total_charges:.2f}, Net: Rs.{net_amount:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def update_positions(self):
        """Update all positions with current prices"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            try:
                current_price = self.data_provider.get_live_price(symbol)
                if not current_price:
                    continue
                
                # Update position
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
                # Check exit conditions
                should_exit, reason = self.check_exit_conditions(position)
                if should_exit:
                    positions_to_close.append((symbol, reason))
                
                # Enhanced position logging with profit/loss status
                pnl_pct = (position.unrealized_pnl / (position.avg_price * position.quantity)) * 100
                pnl_str = f"+Rs.{position.unrealized_pnl:.2f}" if position.unrealized_pnl >= 0 else f"-Rs.{abs(position.unrealized_pnl):.2f}"
                status = "🟢 PROFIT" if position.unrealized_pnl > 0 else "🔴 LOSS" if position.unrealized_pnl < 0 else "⚪ BREAK-EVEN"
                
                logger.info(f"[LIVE] {symbol}: Rs.{position.avg_price:.2f} -> Rs.{current_price:.2f} | {status} {pnl_str} ({pnl_pct:+.2f}%)")
                logger.info(f"[TARGETS] SL: Rs.{position.stop_loss:.2f} | Target: Rs.{position.target:.2f}")
            
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
        
        # Log total unrealized P&L
        if self.positions:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_pnl_str = f"+Rs.{total_unrealized_pnl:.2f}" if total_unrealized_pnl >= 0 else f"-Rs.{abs(total_unrealized_pnl):.2f}"
            logger.info(f"[TOTAL UNREALIZED P&L] {total_pnl_str}")
        
        # Close positions that need to be closed
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def check_exit_conditions(self, position: Position) -> Tuple[bool, str]:
        """Check intraday exit conditions"""
        current_price = position.current_price
        current_time = datetime.now().time()
        
        # Force close all positions at 3:20 PM for intraday
        market_close_time = datetime.strptime('15:20', '%H:%M').time()
        
        # Stop loss hit
        if current_price <= position.stop_loss:
            return True, "Stop Loss"
        
        # Target hit
        if current_price >= position.target:
            return True, "Target"
        
        # Force close before market close for intraday
        if current_time >= market_close_time:
            return True, "Market Close"
        
        # Profit maximization: Take partial profits
        pnl_pct = ((current_price - position.avg_price) / position.avg_price) * 100
        
        # Take 50% profit at 2% gain
        if pnl_pct >= 2.0 and not hasattr(position, 'partial_profit_taken'):
            return True, "Partial Profit 2%"
        
        # Take remaining at 4% gain
        if pnl_pct >= 4.0:
            return True, "Full Profit 4%"
        
        return False, ""
    
    def close_position(self, symbol: str, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = position.current_price
        
        # Calculate charges for selling
        charges_data = self.commission_calc.calculate_charges("SELL", position.quantity, current_price)
        total_charges = charges_data['total_charges']
        net_amount = charges_data['net_amount']
        
        # Calculate P&L
        gross_pnl = (current_price - position.avg_price) * position.quantity
        net_pnl = gross_pnl - total_charges
        
        # Create closing trade
        trade = Trade(
            symbol=symbol,
            action="SELL",
            quantity=position.quantity,
            price=current_price,
            timestamp=datetime.now(),
            charges=total_charges,
            net_amount=net_amount
        )
        
        # Update cash and P&L
        self.available_cash += net_amount
        self.daily_pnl += net_pnl
        self.total_pnl += net_pnl
        
        # Save trade
        self.trades.append(trade)
        self.save_trade(trade, net_pnl)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"[CLOSED] {symbol} @ Rs.{current_price:.2f} - {reason} "
                   f"(P&L: Rs.{net_pnl:.2f}, Charges: Rs.{total_charges:.2f})")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_invested = sum(pos.avg_price * pos.quantity for pos in self.positions.values())
        total_current_value = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'capital': self.capital,
            'available_cash': self.available_cash,
            'invested_amount': total_invested,
            'current_value': total_current_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_pnl,
            'total_pnl': unrealized_pnl + self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'positions_count': len(self.positions),
            'trades_count': len(self.trades)
        }
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        current_time = datetime.now().time()
        
        # Pre-market analysis (8:00 AM - 9:15 AM)
        if current_time >= datetime.strptime('08:00', '%H:%M').time() and current_time < datetime.strptime('09:15', '%H:%M').time():
            if not hasattr(self, '_pre_market_done') or not self._pre_market_done:
                self.market_analyzer.pre_market_analysis()
                self._pre_market_done = True
            return
        
        # Post-market analysis (3:30 PM - 4:00 PM)
        if current_time >= datetime.strptime('15:30', '%H:%M').time() and current_time < datetime.strptime('16:00', '%H:%M').time():
            if not hasattr(self, '_post_market_done') or not self._post_market_done:
                self.market_analyzer.post_market_analysis(self.positions, self.trades)
                self._post_market_done = True
            return
        
        # Reset flags for next day
        if current_time < datetime.strptime('08:00', '%H:%M').time():
            self._pre_market_done = False
            self._post_market_done = False
        
        if not self.is_market_open():
            logger.info("Market is closed - waiting for market hours (9:15 AM - 3:30 PM)")
            return
        
        logger.info("=== TRADING CYCLE START ===")
        logger.info(f"Market Status: {'OPEN' if self.is_market_open() else 'CLOSED'}")
        
        # Update existing positions
        self.update_positions()
        
        # Scan for new opportunities
        logger.info("Scanning for trading opportunities...")
        opportunities = self.scan_opportunities()
        logger.info(f"Found {len(opportunities)} trading opportunities")
        
        # Execute multiple opportunities for maximum profit
        # Sort by signal strength for best opportunities first
        opportunities.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        for opportunity in opportunities[:4]:  # Take up to 4 new positions per cycle
            if len(self.positions) < self.risk_manager.max_positions:
                success = self.execute_trade(opportunity)
                if success:
                    logger.info(f"[PROFIT HUNT] Executed trade {len(self.positions)}/{self.risk_manager.max_positions}")
                time.sleep(0.5)  # Quick execution
        
        # Print portfolio summary
        summary = self.get_portfolio_summary()
        logger.info(f"Portfolio: Cash: Rs.{summary['available_cash']:.2f}, "
                   f"Invested: Rs.{summary['invested_amount']:.2f}, "
                   f"P&L: Rs.{summary['total_pnl']:.2f}")
        
        logger.info("=== TRADING CYCLE END ===")
    
    def start_trading(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info("Enhanced Intraday ML Trading Bot started")
        
        while self.is_running:
            try:
                self.run_trading_cycle()
                time.sleep(180)  # Wait 3 minutes between cycles with web scraper
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)
        
        self.is_running = False
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            self.close_position(symbol, "Bot Stopped")
        
        logger.info("Enhanced Intraday ML Trading Bot stopped")

if __name__ == "__main__":
    # Initialize and start the bot
    bot = EnhancedTradingBot(capital=50000)
    
    print("=" * 60)
    print("    ENHANCED INTRADAY ML TRADING BOT")
    print("=" * 60)
    print(f"Capital: Rs.{bot.capital:,}")
    print(f"Watchlist: {', '.join(bot.watchlist)}")
    print("Features:")
    print("✓ Real NSE/Yahoo Finance data")
    print("✓ Realistic commission calculations")
    print("✓ Proper risk management")
    print("✓ Technical analysis signals")
    print("✓ SQLite trade tracking")
    print("=" * 60)
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        bot.stop_trading()
        print("\nIntraday ML Trading Bot stopped successfully")