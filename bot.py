import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import threading
import json
import os
import requests
import pickle
import talib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from flask import Flask, render_template, jsonify
import schedule
import logging
from abc import ABC, abstractmethod
from collections import deque
from scipy import stats
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller, coint
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_KEY = "YOUR_ALICE_BLUE_API_KEY"
API_SECRET = "YOUR_ALICE_BLUE_API_SECRET"
BUDGET = 7000  # INR
RISK_APPETITE = 7  # Scale of 1-10
MAX_POSITIONS = 5  # Maximum simultaneous positions
MARKET_START_TIME = datetime.time(9, 15)
MARKET_END_TIME = datetime.time(15, 30)

# Indian Market Trading Costs (as percentages)
TRADING_COSTS = {
    'brokerage': 0.01,      # 0.01% brokerage
    'stt': 0.1,            # 0.1% STT (Securities Transaction Tax)
    'transaction_charges': 0.00325,  # NSE transaction charges
    'gst': 18,             # 18% GST on brokerage + transaction charges
    'sebi_charges': 0.0001, # SEBI charges
    'stamp_duty': 0.005,    # Stamp duty
    'slippage': 0.05       # 0.05% slippage
}

class AdvancedPositionSizer:
    """Advanced position sizing with multiple risk management techniques"""
    
    def __init__(self, risk_per_trade=0.02, max_portfolio_risk=0.10):
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.max_portfolio_risk = max_portfolio_risk  # 10% max portfolio risk
        self.kelly_fraction = 0.25  # Fraction of Kelly criterion to use
        self.volatility_target = 0.15  # Annual volatility target (15%)
        self.portfolio_heat = 0  # Current portfolio risk exposure
        
    def calculate_position_size(self, price, volatility, account_balance, current_positions, market_regime=None):
        """Calculate position size using multiple risk management techniques"""
        
        # Adjust risk based on risk appetite
        adjusted_risk = self.risk_per_trade * (RISK_APPETITE / 5)
        
        # 1. Fixed Fractional Risk
        risk_amount = account_balance * adjusted_risk
        
        # 2. Volatility Adjusted Position Sizing
        # Convert daily volatility to annual volatility
        annual_volatility = volatility * np.sqrt(252)
        
        # Scale position to target volatility
        volatility_adjustment = self.volatility_target / max(annual_volatility, 0.05)
        
        # 3. Kelly Criterion (simplified)
        # Estimate win rate and payoff ratio from recent trades
        # In a real implementation, this would come from actual trade history
        win_rate = 0.55  # Default win rate
        payoff_ratio = 1.5  # Average win / average loss
        
        kelly_percentage = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
        
        # Apply Kelly fraction (use fraction of full Kelly to reduce risk)
        kelly_amount = account_balance * kelly_percentage * self.kelly_fraction
        
        # 4. Market Regime Adjustment
        regime_adjustment = 1.0
        if market_regime:
            if market_regime == 'high_volatility':
                regime_adjustment = 0.7  # Reduce position size in high volatility
            elif market_regime == 'trending':
                regime_adjustment = 1.2  # Increase position size in trending markets
            elif market_regime == 'ranging':
                regime_adjustment = 0.9  # Slightly reduce in ranging markets
        
        # 5. Portfolio Heat Adjustment
        # Reduce position size if portfolio is already heavily exposed
        heat_adjustment = 1.0 - min(self.portfolio_heat / self.max_portfolio_risk, 0.5)
        
        # Calculate position value using the most conservative method
        position_value = min(
            risk_amount / volatility_adjustment,
            kelly_amount
        )
        
        # Apply adjustments
        position_value *= volatility_adjustment * regime_adjustment * heat_adjustment
        
        # Calculate position size
        position_size = position_value / price
        
        # Ensure minimum and maximum position sizes
        min_position = 1  # At least 1 share
        max_position = account_balance * 0.2 / price  # Max 20% of account in one position
        
        position_size = max(min_position, min(position_size, max_position))
        
        return int(position_size)
    
    def update_portfolio_heat(self, positions, account_balance):
        """Update the portfolio heat based on current positions"""
        total_risk = 0
        for symbol, position in positions.items():
            position_value = position['quantity'] * position['current_price']
            position_risk = position_value * position['stop_loss']  # Risk amount for this position
            total_risk += position_risk
        
        self.portfolio_heat = total_risk / account_balance

class AdvancedMarketDataSimulator:
    """Advanced market data simulator with realistic market microstructure"""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.data_cache = {}
        self.current_prices = {symbol: None for symbol in symbols}
        self.price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.order_book_depth = {symbol: {'bids': deque(maxlen=10), 'asks': deque(maxlen=10)} for symbol in symbols}
        self.market_regime = {symbol: 'normal' for symbol in symbols}
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize with historical data"""
        for symbol in self.symbols:
            # Generate realistic historical data
            np.random.seed(hash(symbol) % 10000)
            dates = pd.date_range(end=datetime.datetime.now(), periods=252)  # 1 year of data
            base_price = np.random.uniform(100, 5000)
            
            # Create a realistic price series with multiple regimes
            returns = np.zeros(252)
            
            # Generate different market regimes
            regime_changes = np.sort(np.random.choice(range(20, 232), 3, replace=False))
            
            # Low volatility regime
            returns[:regime_changes[0]] = np.random.normal(0.0005, 0.01, regime_changes[0])
            
            # High volatility regime
            returns[regime_changes[0]:regime_changes[1]] = np.random.normal(0.0002, 0.03, regime_changes[1] - regime_changes[0])
            
            # Trending regime
            trend_drift = 0.001 if np.random.random() > 0.5 else -0.001
            returns[regime_changes[1]:regime_changes[2]] = np.random.normal(trend_drift, 0.015, regime_changes[2] - regime_changes[1])
            
            # Normal regime
            returns[regime_changes[2]:] = np.random.normal(0.0005, 0.02, 252 - regime_changes[2])
            
            # Create price series
            price_series = [base_price]
            for ret in returns:
                price_series.append(price_series[-1] * (1 + ret))
            
            # Create OHLCV data
            data = pd.DataFrame({
                'date': dates,
                'open': price_series[1:],
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in price_series[1:]],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in price_series[1:]],
                'close': price_series[1:],
                'volume': np.random.randint(1000, 100000, 252)
            })
            
            self.data_cache[symbol] = data.set_index('date')
            self.current_prices[symbol] = price_series[-1]
            self.price_history[symbol] = deque(price_series[-20:], maxlen=100)
            
            # Initialize order book
            self._initialize_order_book(symbol)
    
    def _initialize_order_book(self, symbol):
        """Initialize order book for a symbol"""
        current_price = self.current_prices[symbol]
        
        # Generate realistic order book
        bid_spread = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2% bid-ask spread
        ask_price = current_price * (1 + bid_spread/2)
        bid_price = current_price * (1 - bid_spread/2)
        
        # Generate order book depth
        bid_sizes = np.random.lognormal(mean=8, sigma=1, size=10).astype(int)
        ask_sizes = np.random.lognormal(mean=8, sigma=1, size=10).astype(int)
        
        # Create bid levels (decreasing price)
        bid_prices = [bid_price * (1 - i*0.0005) for i in range(10)]
        
        # Create ask levels (increasing price)
        ask_prices = [ask_price * (1 + i*0.0005) for i in range(10)]
        
        # Populate order book
        for i in range(10):
            self.order_book_depth[symbol]['bids'].append({'price': bid_prices[i], 'size': bid_sizes[i]})
            self.order_book_depth[symbol]['asks'].append({'price': ask_prices[i], 'size': ask_sizes[i]})
    
    def get_historical_data(self, symbol, timeframe='1d', days=252):
        """Get historical data for a symbol"""
        logger.info(f"Fetching historical data for {symbol}")
        return self.data_cache[symbol].copy()
    
    def get_real_time_data(self, symbol):
        """Simulate real-time data for a symbol"""
        # Simulate price movement based on market regime
        current_price = self.current_prices[symbol]
        
        # Determine market regime
        if np.random.random() < 0.01:  # 1% chance to change regime
            regimes = ['low_volatility', 'high_volatility', 'trending', 'ranging', 'normal']
            self.market_regime[symbol] = np.random.choice(regimes)
        
        # Generate return based on market regime
        if self.market_regime[symbol] == 'low_volatility':
            change_percent = np.random.normal(0, 0.005)
        elif self.market_regime[symbol] == 'high_volatility':
            change_percent = np.random.normal(0, 0.03)
        elif self.market_regime[symbol] == 'trending':
            trend = 0.001 if np.random.random() > 0.5 else -0.001
            change_percent = np.random.normal(trend, 0.01)
        elif self.market_regime[symbol] == 'ranging':
            # Mean reversion
            mean_price = np.mean(list(self.price_history[symbol])[-20:])
            reversion_force = (mean_price - current_price) / current_price * 0.1
            change_percent = np.random.normal(reversion_force, 0.01)
        else:  # normal
            change_percent = np.random.normal(0.0005, 0.015)
        
        # Apply price change
        new_price = current_price * (1 + change_percent)
        
        # Update current price and history
        self.current_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Update order book
        self._update_order_book(symbol, new_price)
        
        # Calculate OHLC for this "tick"
        history = list(self.price_history[symbol])
        open_price = history[0]
        close_price = history[-1]
        high_price = max(history)
        low_price = min(history)
        
        return {
            'symbol': symbol,
            'price': new_price,
            'change': change_percent * 100,
            'volume': np.random.randint(100, 10000),
            'timestamp': datetime.datetime.now(),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'market_regime': self.market_regime[symbol],
            'order_book': {
                'bids': list(self.order_book_depth[symbol]['bids']),
                'asks': list(self.order_book_depth[symbol]['asks'])
            }
        }
    
    def _update_order_book(self, symbol, new_price):
        """Update order book based on new price"""
        # Shift order book
        for i in range(9, 0, -1):
            self.order_book_depth[symbol]['bids'][i] = self.order_book_depth[symbol]['bids'][i-1]
            self.order_book_depth[symbol]['asks'][i] = self.order_book_depth[symbol]['asks'][i-1]
        
        # Generate new best bid and ask
        bid_spread = np.random.uniform(0.0005, 0.002)
        ask_price = new_price * (1 + bid_spread/2)
        bid_price = new_price * (1 - bid_spread/2)
        
        # Generate new sizes
        bid_size = int(np.random.lognormal(mean=8, sigma=1))
        ask_size = int(np.random.lognormal(mean=8, sigma=1))
        
        # Update best levels
        self.order_book_depth[symbol]['bids'][0] = {'price': bid_price, 'size': bid_size}
        self.order_book_depth[symbol]['asks'][0] = {'price': ask_price, 'size': ask_size}

class UltraAdvancedAIRiskManager:
    """Ultra-advanced AI-based risk management using state-of-the-art techniques"""
    
    def __init__(self):
        # Ensemble of different model types
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.01, 
                                batch_size='auto', learning_rate='adaptive', max_iter=500, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        }
        
        self.lstm_model = None
        self.cnn_model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.trained = False
        self.model_weights = {model: 1.0/len(self.models) for model in self.models}
        self.feature_importance = {}
        self.market_regime_detector = MarketRegimeDetector()
        
    def _prepare_features(self, data):
        """Prepare advanced features for AI models"""
        df = data.copy()
        
        # Basic returns and volatility
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_change'] = df['volatility'].pct_change()
        
        # Price-based features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        # Volume-based features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        df['volume_volatility'] = df['volume_change'].rolling(20).std()
        
        # Technical indicators
        # Overlap Studies
        df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_6'] = talib.RSI(df['close'], timeperiod=6)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volatility Indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume Indicators
        df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Cycle Indicators
        df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
        df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
        df['ht_phasor_inphase'], df['ht_phasor_quadrature'] = talib.HT_PHASOR(df['close'])
        df['ht_sine'], df['ht_sinelead'] = talib.HT_SINE(df['close'])
        
        # Pattern Recognition
        df['cdl2crows'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
        df['cdl3blackcrows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        df['cdl3inside'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
        df['cdl3linestrike'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
        df['cdl3outside'] = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
        df['cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
        df['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        df['cdlabandonedbaby'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'])
        df['cdladvanceblock'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
        df['cdlbelthold'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
        df['cdlbreakaway'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
        df['cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
        df['cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
        df['cdlcounterattack'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
        df['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
        df['cdldoji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdldojistar'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdleveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
        df['cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdlhammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdlhangingman'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
        df['cdlharami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        df['cdlharamicross'] = talib.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
        df['cdlhighwave'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
        df['cdlhikkake'] = talib.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
        df['cdlhikkakemod'] = talib.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])
        df['cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])
        df['cdlidlelongwhite'] = talib.CDLIDLELONGWHITE(df['open'], df['high'], df['low'], df['close'])
        df['cdlinneck'] = talib.CDLINNECK(df['open'], df['high'], df['low'], df['close'])
        df['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdlkicking'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
        df['cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])
        df['cdlladderbottom'] = talib.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])
        df['cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdllongline'] = talib.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])
        df['cdlmarubozu'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
        df['cdlmatchinglow'] = talib.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])
        df['cdlmathold'] = talib.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'])
        df['cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdlmorningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdlonneck'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])
        df['cdlpiercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
        df['cdlrickshawman'] = talib.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])
        df['cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
        df['cdlseparatinglines'] = talib.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])
        df['cdlshootingstar'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdlshortline'] = talib.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])
        df['cdlspinningtop'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
        df['cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])
        df['cdlsticksandwich'] = talib.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])
        df['cdltakuri'] = talib.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])
        df['cdltasukigap'] = talib.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])
        df['cdlthrusting'] = talib.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])
        df['cdltristar'] = talib.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdlunique3river'] = talib.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])
        df['cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])
        df['cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])
        
        # Statistical features
        df['returns_skew'] = df['returns'].rolling(20).skew()
        df['returns_kurtosis'] = df['returns'].rolling(20).kurt()
        df['returns_zscore'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['returns'].rolling(20).std()
        
        # Trend features
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['trend_direction'] = np.sign(df['trend_strength'])
        df['trend_change'] = df['trend_direction'].diff()
        
        # Mean reversion features
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_width'] = (df['upper'] - df['lower']) / df['middle']
        df['bb_position'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
        
        # Price acceleration
        df['price_acceleration'] = df['returns'].diff()
        df['price_acceleration_change'] = df['price_acceleration'].diff()
        
        # Volume-price trend
        df['vpt'] = df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['vpt_sma'] = df['vpt'].rolling(20).mean()
        
        # Money flow index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Drop NA values
        df = df.dropna()
        
        return df
    
    def _create_lstm_model(self, input_shape):
        """Create advanced LSTM model with regularization"""
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False, kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(2))  # Predict stop_loss and take_profit
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def _create_cnn_model(self, input_shape):
        """Create 1D CNN model for time series"""
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(50, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(2))  # Predict stop_loss and take_profit
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def _prepare_lstm_data(self, data, look_back=30):
        """Prepare data for LSTM and CNN models"""
        features = data.drop(['date'], axis=1, errors='ignore').values
        X, y = [], []
        
        for i in range(look_back, len(features)):
            X.append(features[i-look_back:i])
            y.append([data['future_max_loss'].iloc[i], data['future_max_gain'].iloc[i]])
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data):
        """Train the AI models on historical data"""
        logger.info("Training ultra-advanced AI risk management models...")
        
        # Prepare features
        df = self._prepare_features(historical_data)
        
        # Target variables: future max gain and max loss
        df['future_max_gain'] = df['close'].rolling(5).max().pct_change(5).shift(-5)
        df['future_max_loss'] = df['close'].rolling(5).min().pct_change(5).shift(-5)
        
        # Drop NA values
        df = df.dropna()
        
        # Features and targets
        feature_cols = [col for col in df.columns if col not in ['date', 'future_max_gain', 'future_max_loss']]
        X = df[feature_cols]
        y_gain = df['future_max_gain']
        y_loss = df['future_max_loss']
        
        # Apply PCA for dimensionality reduction
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Split data
        X_train, X_test, y_gain_train, y_gain_test, y_loss_train, y_loss_test = train_test_split(
            X_pca, y_gain, y_loss, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train traditional models
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Train model for gain prediction
            model.fit(X_train, y_gain_train)
            gain_pred = model.predict(X_test)
            gain_mse = mean_squared_error(y_gain_test, gain_pred)
            
            # Train model for loss prediction
            model.fit(X_train, y_loss_train)
            loss_pred = model.predict(X_test)
            loss_mse = mean_squared_error(y_loss_test, loss_pred)
            
            # Average MSE
            avg_mse = (gain_mse + loss_mse) / 2
            model_scores[name] = avg_mse
            
            logger.info(f"{name} model trained with MSE: {avg_mse:.6f}")
        
        # Train deep learning models
        try:
            # Prepare data for LSTM and CNN
            X_lstm, y_lstm = self._prepare_lstm_data(df)
            
            # Split LSTM data
            split_idx = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
            
            # Create and train LSTM model
            self.lstm_model = self._create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            history = self.lstm_model.fit(
                X_train_lstm, y_train_lstm,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_lstm, y_test_lstm),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            lstm_pred = self.lstm_model.predict(X_test_lstm)
            lstm_mse = mean_squared_error(y_test_lstm, lstm_pred)
            model_scores['lstm'] = lstm_mse
            logger.info(f"LSTM model trained with MSE: {lstm_mse:.6f}")
            
            # Create and train CNN model
            self.cnn_model = self._create_cnn_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
            
            history = self.cnn_model.fit(
                X_train_lstm, y_train_lstm,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_lstm, y_test_lstm),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            cnn_pred = self.cnn_model.predict(X_test_lstm)
            cnn_mse = mean_squared_error(y_test_lstm, cnn_pred)
            model_scores['cnn'] = cnn_mse
            logger.info(f"CNN model trained with MSE: {cnn_mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error training deep learning models: {e}")
            self.lstm_model = None
            self.cnn_model = None
        
        # Calculate model weights based on performance (inverse of MSE)
        total_mse = sum(1/score for score in model_scores.values())
        self.model_weights = {model: (1/score)/total_mse for model, score in model_scores.items()}
        
        # Normalize weights
        weight_sum = sum(self.model_weights.values())
        self.model_weights = {model: weight/weight_sum for model, weight in self.model_weights.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        # Calculate feature importance from tree-based models
        feature_importance = np.zeros(len(feature_cols))
        
        for name in ['rf', 'gb', 'xgb', 'lgb']:
            if hasattr(self.models[name], 'feature_importances_'):
                # Map PCA components back to original features
                components = self.pca.components_
                importance = np.sum(np.abs(components * self.models[name].feature_importances_), axis=0)
                feature_importance += importance
        
        # Normalize feature importance
        feature_importance = feature_importance / feature_importance.sum()
        
        # Create feature importance dictionary
        self.feature_importance = {feature_cols[i]: feature_importance[i] for i in range(len(feature_cols))}
        
        # Get top features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top features: {top_features}")
        
        self.trained = True
        logger.info("Ultra-advanced AI risk management models trained successfully")
    
    def get_stop_loss_take_profit(self, current_data, historical_data):
        """Get AI-based stop-loss and take-profit levels using ensemble approach"""
        if not self.trained:
            # Default values if model not trained
            return -0.02, 0.03  # 2% stop loss, 3% take profit
        
        # Prepare features
        df = self._prepare_features(historical_data)
        
        # Add current data point
        current_df = pd.DataFrame([current_data])
        df = pd.concat([df, current_df], ignore_index=True)
        
        # Get the last row (current data)
        current_features = df.iloc[-1]
        
        # Prepare feature vector
        feature_cols = [col for col in df.columns if col not in ['date', 'future_max_gain', 'future_max_loss']]
        X = current_features[feature_cols].values.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA
        X_pca = self.pca.transform(X_scaled)
        
        # Get predictions from each model
        predictions = {}
        
        # Traditional models
        for name, model in self.models.items():
            # Predict gain
            model.fit(X_pca, df['future_max_gain'].iloc[:-1])  # Retrain on latest data
            gain_pred = model.predict(X_pca)[0]
            
            # Predict loss
            model.fit(X_pca, df['future_max_loss'].iloc[:-1])  # Retrain on latest data
            loss_pred = model.predict(X_pca)[0]
            
            predictions[name] = (loss_pred, gain_pred)
        
        # Deep learning models
        if self.lstm_model is not None:
            try:
                # Prepare LSTM data
                lstm_data = df[-30:].copy()  # Last 30 data points
                lstm_features = lstm_data[feature_cols].values
                X_lstm = lstm_features.reshape(1, lstm_features.shape[0], lstm_features.shape[1])
                
                lstm_pred = self.lstm_model.predict(X_lstm)[0]
                predictions['lstm'] = (lstm_pred[0], lstm_pred[1])
            except Exception as e:
                logger.error(f"Error getting LSTM prediction: {e}")
                predictions['lstm'] = (0.0, 0.0)
        else:
            predictions['lstm'] = (0.0, 0.0)
        
        if self.cnn_model is not None:
            try:
                # Prepare CNN data
                cnn_data = df[-30:].copy()  # Last 30 data points
                cnn_features = cnn_data[feature_cols].values
                X_cnn = cnn_features.reshape(1, cnn_features.shape[0], cnn_features.shape[1])
                
                cnn_pred = self.cnn_model.predict(X_cnn)[0]
                predictions['cnn'] = (cnn_pred[0], cnn_pred[1])
            except Exception as e:
                logger.error(f"Error getting CNN prediction: {e}")
                predictions['cnn'] = (0.0, 0.0)
        else:
            predictions['cnn'] = (0.0, 0.0)
        
        # Weighted ensemble prediction
        stop_loss = 0
        take_profit = 0
        
        for model, (loss, gain) in predictions.items():
            weight = self.model_weights.get(model, 0)
            stop_loss += loss * weight
            take_profit += gain * weight
        
        # Detect market regime
        market_regime = self.market_regime_detector.detect_market_regime(historical_data)
        
        # Adjust based on market regime
        if market_regime == 'high_volatility':
            # Wider stops in high volatility
            stop_loss *= 1.2
            take_profit *= 1.2
        elif market_regime == 'trending':
            # Tighter stops, wider targets in trending markets
            stop_loss *= 0.8
            take_profit *= 1.3
        elif market_regime == 'ranging':
            # Tighter stops and targets in ranging markets
            stop_loss *= 0.9
            take_profit *= 0.9
        
        # Adjust based on risk appetite
        risk_adjustment = (RISK_APPETITE - 5) * 0.2
        stop_loss = stop_loss * (1 + risk_adjustment)
        take_profit = take_profit * (1 + risk_adjustment)
        
        # Ensure reasonable values
        stop_loss = max(min(stop_loss, -0.008), -0.06)  # Between 0.8% and 6%
        take_profit = min(max(take_profit, 0.012), 0.08)  # Between 1.2% and 8%
        
        # Ensure risk-reward ratio is at least 1:1.5
        if take_profit / abs(stop_loss) < 1.5:
            take_profit = abs(stop_loss) * 1.5
        
        return stop_loss, take_profit
    
    def save_models(self, path):
        """Save trained models to disk"""
        if not self.trained:
            logger.warning("Models not trained yet. Cannot save.")
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save traditional models
        for name, model in self.models.items():
            with open(os.path.join(path, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save deep learning models
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(path, 'lstm_model.h5'))
        
        if self.cnn_model is not None:
            self.cnn_model.save(os.path.join(path, 'cnn_model.h5'))
        
        # Save scaler and PCA
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(path, 'pca.pkl'), 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Save model weights
        with open(os.path.join(path, 'model_weights.json'), 'w') as f:
            json.dump(self.model_weights, f)
        
        # Save feature importance
        with open(os.path.join(path, 'feature_importance.json'), 'w') as f:
            json.dump(self.feature_importance, f)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path):
        """Load trained models from disk"""
        if not os.path.exists(path):
            logger.error(f"Model path {path} does not exist")
            return False
        
        try:
            # Load traditional models
            for name in self.models.keys():
                model_path = os.path.join(path, f'{name}_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load deep learning models
            lstm_path = os.path.join(path, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path)
            else:
                self.lstm_model = None
            
            cnn_path = os.path.join(path, 'cnn_model.h5')
            if os.path.exists(cnn_path):
                self.cnn_model = load_model(cnn_path)
            else:
                self.cnn_model = None
            
            # Load scaler and PCA
            with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(os.path.join(path, 'pca.pkl'), 'rb') as f:
                self.pca = pickle.load(f)
            
            # Load model weights
            with open(os.path.join(path, 'model_weights.json'), 'r') as f:
                self.model_weights = json.load(f)
            
            # Load feature importance
            with open(os.path.join(path, 'feature_importance.json'), 'r') as f:
                self.feature_importance = json.load(f)
            
            self.trained = True
            logger.info(f"Models loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

class MarketRegimeDetector:
    """Detects market regimes using statistical and ML methods"""
    
    def __init__(self):
        self.regimes = ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']
        self.current_regime = 'normal'
        
    def detect_market_regime(self, data):
        """Detect the current market regime"""
        if len(data) < 50:
            return 'normal'
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Calculate volatility
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Calculate trend
        sma_20 = talib.SMA(data['close'], timeperiod=20).iloc[-1]
        sma_50 = talib.SMA(data['close'], timeperiod=50).iloc[-1]
        
        # Calculate ADX for trend strength
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
        
        # Detect regime based on indicators
        if volatility > 0.03:  # High volatility
            regime = 'high_volatility'
        elif volatility < 0.01:  # Low volatility
            regime = 'low_volatility'
        elif adx > 25:  # Strong trend
            if sma_20 > sma_50:
                regime = 'trending_up'
            else:
                regime = 'trending_down'
        else:  # Weak or no trend
            regime = 'ranging'
        
        self.current_regime = regime
        return regime

class AdvancedStrategySelector:
    """Advanced strategy selection using ensemble methods and market regime detection"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.trained = False
        self.strategy_names = []
        self.model_weights = {model: 1.0/len(self.models) for model in self.models}
        self.market_regime_detector = MarketRegimeDetector()
        self.regime_strategy_weights = {
            'trending_up': {'Trend Following': 0.6, 'Momentum': 0.3, 'Mean Reversion': 0.1},
            'trending_down': {'Trend Following': 0.6, 'Momentum': 0.2, 'Mean Reversion': 0.2},
            'ranging': {'Trend Following': 0.1, 'Momentum': 0.1, 'Mean Reversion': 0.8},
            'high_volatility': {'Trend Following': 0.3, 'Momentum': 0.4, 'Mean Reversion': 0.3},
            'low_volatility': {'Trend Following': 0.4, 'Momentum': 0.3, 'Mean Reversion': 0.3}
        }
    
    def _prepare_features(self, data):
        """Prepare features for strategy selection"""
        df = data.copy()
        
        # Calculate market regime features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Calculate trend strength
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Calculate momentum
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        # Calculate mean reversion potential
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_position'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
        
        # Calculate volume trend
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'] / df['volume_sma']
        
        # Calculate ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Calculate MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])
        
        # Drop NA values
        df = df.dropna()
        
        return df
    
    def train(self, historical_data, strategies):
        """Train the strategy selection model"""
        logger.info("Training advanced strategy selection model...")
        
        self.strategy_names = [strategy.get_name() for strategy in strategies]
        
        # Prepare features
        df = self._prepare_features(historical_data)
        
        # Generate signals from each strategy
        strategy_signals = {}
        for strategy in strategies:
            signals = strategy.generate_signals(historical_data)
            strategy_signals[strategy.get_name()] = signals['signal']
        
        # Create target variable: which strategy would have been most profitable
        # We'll use a more sophisticated approach with forward returns and risk-adjusted performance
        df['future_return'] = df['close'].pct_change(5).shift(-5)
        df['future_volatility'] = df['returns'].rolling(5).std().shift(-5)
        
        # Calculate risk-adjusted returns for each strategy
        strategy_sharpe = {}
        for name in self.strategy_names:
            strategy_returns = df['future_return'] * strategy_signals[name]
            strategy_sharpe[name] = strategy_returns / df['future_volatility']
        
        # Select best strategy for each time point based on Sharpe ratio
        best_strategy = []
        for i in range(len(df)):
            sharpe_ratios = {name: strategy_sharpe[name].iloc[i] for name in self.strategy_names}
            best = max(sharpe_ratios, key=sharpe_ratios.get)
            best_strategy.append(self.strategy_names.index(best))
        
        df['best_strategy'] = best_strategy
        
        # Features and target
        feature_cols = [col for col in df.columns if col not in ['date', 'future_return', 'future_volatility', 'best_strategy']]
        X = df[feature_cols]
        y = df['best_strategy']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models and evaluate
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            logger.info(f"{name} model trained with accuracy: {train_score:.4f} (train), {test_score:.4f} (test)")
            
            # Use test score as model performance metric
            model_scores[name] = test_score
        
        # Calculate model weights based on performance
        total_score = sum(model_scores.values())
        self.model_weights = {model: score/total_score for model, score in model_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        self.trained = True
    
    def select_strategy(self, current_data, historical_data):
        """Select the best strategy for current market conditions"""
        if not self.trained:
            # Equal weights if model not trained
            return {name: 1.0/len(self.strategy_names) for name in self.strategy_names}
        
        # Detect market regime
        market_regime = self.market_regime_detector.detect_market_regime(historical_data)
        
        # Prepare features
        df = self._prepare_features(historical_data)
        
        # Add current data point
        current_df = pd.DataFrame([current_data])
        df = pd.concat([df, current_df], ignore_index=True)
        
        # Get the last row (current data)
        current_features = df.iloc[-1]
        
        # Prepare feature vector
        feature_cols = [col for col in df.columns if col not in ['date', 'future_return', 'future_volatility', 'best_strategy']]
        X = current_features[feature_cols].values.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get strategy probabilities from each model
        strategy_probs = {}
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(X_scaled)[0]
                weight = self.model_weights.get(name, 0)
                
                # Apply model weight to probabilities
                for i, strategy in enumerate(self.strategy_names):
                    if strategy not in strategy_probs:
                        strategy_probs[strategy] = 0
                    strategy_probs[strategy] += probs[i] * weight
            except Exception as e:
                logger.error(f"Error getting {name} prediction: {e}")
        
        # Normalize probabilities
        total_prob = sum(strategy_probs.values())
        if total_prob > 0:
            strategy_probs = {strategy: prob/total_prob for strategy, prob in strategy_probs.items()}
        
        # Adjust based on market regime
        if market_regime in self.regime_strategy_weights:
            regime_weights = self.regime_strategy_weights[market_regime]
            
            # Blend model predictions with regime-based weights
            final_weights = {}
            for strategy in self.strategy_names:
                model_weight = strategy_probs.get(strategy, 1.0/len(self.strategy_names))
                regime_weight = regime_weights.get(strategy, 1.0/len(self.strategy_names))
                
                # 70% model weight, 30% regime weight
                final_weights[strategy] = 0.7 * model_weight + 0.3 * regime_weight
            
            # Normalize final weights
            total_weight = sum(final_weights.values())
            final_weights = {strategy: weight/total_weight for strategy, weight in final_weights.items()}
            
            return final_weights
        else:
            return strategy_probs
    
    def save_model(self, path):
        """Save trained model to disk"""
        if not self.trained:
            logger.warning("Model not trained yet. Cannot save.")
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(os.path.join(path, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save strategy names
        with open(os.path.join(path, 'strategy_names.json'), 'w') as f:
            json.dump(self.strategy_names, f)
        
        # Save model weights
        with open(os.path.join(path, 'model_weights.json'), 'w') as f:
            json.dump(self.model_weights, f)
        
        logger.info(f"Strategy selection model saved to {path}")
    
    def load_model(self, path):
        """Load trained model from disk"""
        if not os.path.exists(path):
            logger.error(f"Model path {path} does not exist")
            return False
        
        try:
            # Load models
            for name in self.models.keys():
                model_path = os.path.join(path, f'{name}_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load scaler
            with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load strategy names
            with open(os.path.join(path, 'strategy_names.json'), 'r') as f:
                self.strategy_names = json.load(f)
            
            # Load model weights
            with open(os.path.join(path, 'model_weights.json'), 'r') as f:
                self.model_weights = json.load(f)
            
            self.trained = True
            logger.info(f"Strategy selection model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading strategy selection model: {e}")
            return False

class AdvancedBacktester:
    """Advanced backtesting engine with walk-forward optimization and Monte Carlo simulation"""
    
    def __init__(self, data_fetcher, strategies, risk_manager, position_sizer, initial_balance=BUDGET):
        self.data_fetcher = data_fetcher
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.initial_balance = initial_balance
        self.results = {}
    
    def run_backtest(self, symbol, start_date, end_date, walk_forward=True):
        """Run backtest for a symbol over a date range with walk-forward optimization"""
        logger.info(f"Running advanced backtest for {symbol} from {start_date} to {end_date}")
        
        # Get historical data
        data = self.data_fetcher.get_historical_data(symbol, days=500)
        
        # Filter by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if len(data) < 100:
            logger.warning(f"Not enough data for backtesting {symbol}")
            return None
        
        if walk_forward:
            # Walk-forward optimization
            return self._walk_forward_optimization(symbol, data)
        else:
            # Simple backtest
            return self._simple_backtest(symbol, data)
    
    def _simple_backtest(self, symbol, data):
        """Simple backtest without walk-forward optimization"""
        # Initialize simulated broker
        broker = SimulatedBroker(self.initial_balance)
        
        # Train risk manager if not trained
        if not self.risk_manager.trained:
            self.risk_manager.train(data)
        
        # Initialize strategy selector if needed
        strategy_selector = AdvancedStrategySelector()
        strategy_selector.train(data, self.strategies)
        
        # Track performance
        portfolio_values = []
        positions = {}
        
        # Run backtest
        for i in range(50, len(data)):
            current_data = data.iloc[i]
            historical_data = data.iloc[:i]
            
            # Update existing positions
            for symbol_pos in list(positions.keys()):
                broker.update_position(symbol_pos, current_data['close'])
            
            # Check if we can open new positions
            current_positions = broker.get_positions()
            if len(current_positions) < MAX_POSITIONS and symbol not in current_positions:
                # Get strategy weights
                strategy_weights = strategy_selector.select_strategy(
                    current_data.to_dict(), historical_data
                )
                
                # Generate signals from all strategies
                signals = {}
                for strategy in self.strategies:
                    strategy_signals = strategy.generate_signals(historical_data)
                    latest_signal = strategy_signals.iloc[-1]
                    signals[strategy.get_name()] = latest_signal['signal']
                
                # Combine signals using strategy weights
                weighted_signal = 0
                for name, signal in signals.items():
                    weighted_signal += signal * strategy_weights.get(name, 0)
                
                # Make decision
                if weighted_signal > 0.5:  # Buy signal
                    # Calculate position size
                    volatility = historical_data['close'].pct_change().std()
                    position_size = self.position_sizer.calculate_position_size(
                        current_data['close'], volatility, broker.cash_balance,
                        current_positions, self.risk_manager.market_regime_detector.current_regime
                    )
                    
                    # Get AI-based stop-loss and take-profit
                    stop_loss, take_profit = self.risk_manager.get_stop_loss_take_profit(
                        current_data.to_dict(), historical_data
                    )
                    
                    # Open position
                    position = broker.open_position(
                        symbol=symbol,
                        quantity=position_size,
                        price=current_data['close'],
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if position:
                        positions[symbol] = position
            
            # Record portfolio value
            account_summary = broker.get_account_summary()
            portfolio_values.append({
                'date': current_data.name,
                'portfolio_value': account_summary['total_value'],
                'cash': account_summary['cash_balance'],
                'positions': account_summary['position_value']
            })
        
        # Close all positions at the end
        for symbol_pos in list(positions.keys()):
            broker.close_position(symbol_pos, data.iloc[-1]['close'])
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_balance) / self.initial_balance
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252)
        max_drawdown = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax() - 1).min()
        
        # Get trade history
        trade_history = broker.trade_history
        
        # Calculate win rate
        if trade_history:
            winning_trades = sum(1 for trade in trade_history if trade['final_pnl'] > 0)
            win_rate = winning_trades / len(trade_history) * 100
        else:
            win_rate = 0
        
        results = {
            'symbol': symbol,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'initial_balance': self.initial_balance,
            'final_balance': portfolio_df['portfolio_value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trade_history),
            'transaction_costs': broker.transaction_costs,
            'portfolio_values': portfolio_df,
            'trade_history': trade_history
        }
        
        self.results[symbol] = results
        logger.info(f"Backtest completed for {symbol}: Total return = {total_return:.2%}, Win rate = {win_rate:.2f}%")
        
        return results
    
    def _walk_forward_optimization(self, symbol, data):
        """Walk-forward optimization for more robust backtesting"""
        # Split data into training and testing periods
        train_size = int(len(data) * 0.7)  # 70% training, 30% testing
        
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        logger.info(f"Walk-forward optimization: Training on {len(train_data)} days, testing on {len(test_data)} days")
        
        # Initialize simulated broker
        broker = SimulatedBroker(self.initial_balance)
        
        # Train models on training data
        self.risk_manager.train(train_data)
        
        strategy_selector = AdvancedStrategySelector()
        strategy_selector.train(train_data, self.strategies)
        
        # Track performance
        portfolio_values = []
        positions = {}
        
        # Run backtest on test data
        for i in range(len(test_data)):
            current_data = test_data.iloc[i]
            historical_data = pd.concat([train_data, test_data.iloc[:i]])
            
            # Update existing positions
            for symbol_pos in list(positions.keys()):
                broker.update_position(symbol_pos, current_data['close'])
            
            # Check if we can open new positions
            current_positions = broker.get_positions()
            if len(current_positions) < MAX_POSITIONS and symbol not in current_positions:
                # Get strategy weights
                strategy_weights = strategy_selector.select_strategy(
                    current_data.to_dict(), historical_data
                )
                
                # Generate signals from all strategies
                signals = {}
                for strategy in self.strategies:
                    strategy_signals = strategy.generate_signals(historical_data)
                    latest_signal = strategy_signals.iloc[-1]
                    signals[strategy.get_name()] = latest_signal['signal']
                
                # Combine signals using strategy weights
                weighted_signal = 0
                for name, signal in signals.items():
                    weighted_signal += signal * strategy_weights.get(name, 0)
                
                # Make decision
                if weighted_signal > 0.5:  # Buy signal
                    # Calculate position size
                    volatility = historical_data['close'].pct_change().std()
                    position_size = self.position_sizer.calculate_position_size(
                        current_data['close'], volatility, broker.cash_balance,
                        current_positions, self.risk_manager.market_regime_detector.current_regime
                    )
                    
                    # Get AI-based stop-loss and take-profit
                    stop_loss, take_profit = self.risk_manager.get_stop_loss_take_profit(
                        current_data.to_dict(), historical_data
                    )
                    
                    # Open position
                    position = broker.open_position(
                        symbol=symbol,
                        quantity=position_size,
                        price=current_data['close'],
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if position:
                        positions[symbol] = position
            
            # Record portfolio value
            account_summary = broker.get_account_summary()
            portfolio_values.append({
                'date': current_data.name,
                'portfolio_value': account_summary['total_value'],
                'cash': account_summary['cash_balance'],
                'positions': account_summary['position_value']
            })
        
        # Close all positions at the end
        for symbol_pos in list(positions.keys()):
            broker.close_position(symbol_pos, test_data.iloc[-1]['close'])
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_balance) / self.initial_balance
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252)
        max_drawdown = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax() - 1).min()
        
        # Get trade history
        trade_history = broker.trade_history
        
        # Calculate win rate
        if trade_history:
            winning_trades = sum(1 for trade in trade_history if trade['final_pnl'] > 0)
            win_rate = winning_trades / len(trade_history) * 100
        else:
            win_rate = 0
        
        results = {
            'symbol': symbol,
            'start_date': test_data.index[0],
            'end_date': test_data.index[-1],
            'initial_balance': self.initial_balance,
            'final_balance': portfolio_df['portfolio_value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trade_history),
            'transaction_costs': broker.transaction_costs,
            'portfolio_values': portfolio_df,
            'trade_history': trade_history,
            'walk_forward': True
        }
        
        self.results[symbol] = results
        logger.info(f"Walk-forward backtest completed for {symbol}: Total return = {total_return:.2%}, Win rate = {win_rate:.2f}%")
        
        return results
    
    def run_monte_carlo_simulation(self, symbol, num_simulations=1000):
        """Run Monte Carlo simulation to assess strategy robustness"""
        if symbol not in self.results:
            logger.error(f"No backtest results found for {symbol}")
            return None
        
        results = self.results[symbol]
        portfolio_values = results['portfolio_values']
        
        # Extract returns
        returns = portfolio_values['portfolio_value'].pct_change().dropna()
        
        # Run Monte Carlo simulation
        simulation_results = []
        
        for i in range(num_simulations):
            # Generate random returns based on historical distribution
            random_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate portfolio values
            simulated_values = [self.initial_balance]
            for r in random_returns:
                simulated_values.append(simulated_values[-1] * (1 + r))
            
            # Calculate metrics
            simulated_values = pd.Series(simulated_values)
            total_return = (simulated_values.iloc[-1] - self.initial_balance) / self.initial_balance
            max_drawdown = (simulated_values / simulated_values.cummax() - 1).min()
            
            simulation_results.append({
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'final_value': simulated_values.iloc[-1]
            })
        
        # Calculate statistics
        simulation_df = pd.DataFrame(simulation_results)
        
        # Calculate percentiles
        return_percentiles = {
            '5th': simulation_df['total_return'].quantile(0.05),
            '25th': simulation_df['total_return'].quantile(0.25),
            '50th': simulation_df['total_return'].quantile(0.5),
            '75th': simulation_df['total_return'].quantile(0.75),
            '95th': simulation_df['total_return'].quantile(0.95)
        }
        
        drawdown_percentiles = {
            '5th': simulation_df['max_drawdown'].quantile(0.05),
            '25th': simulation_df['max_drawdown'].quantile(0.25),
            '50th': simulation_df['max_drawdown'].quantile(0.5),
            '75th': simulation_df['max_drawdown'].quantile(0.75),
            '95th': simulation_df['max_drawdown'].quantile(0.95)
        }
        
        # Calculate probability of profit
        prob_profit = (simulation_df['total_return'] > 0).mean()
        
        # Calculate probability of exceeding return threshold
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
        prob_exceed_threshold = {f"prob_exceed_{int(threshold*100)}%": (simulation_df['total_return'] > threshold).mean() 
                               for threshold in thresholds}
        
        monte_carlo_results = {
            'symbol': symbol,
            'num_simulations': num_simulations,
            'return_percentiles': return_percentiles,
            'drawdown_percentiles': drawdown_percentiles,
            'prob_profit': prob_profit,
            'prob_exceed_threshold': prob_exceed_threshold,
            'actual_return': results['total_return'],
            'actual_max_drawdown': results['max_drawdown']
        }
        
        logger.info(f"Monte Carlo simulation completed for {symbol}: Probability of profit = {prob_profit:.2%}")
        
        return monte_carlo_results
    
    def generate_backtest_report(self, symbol, include_monte_carlo=True):
        """Generate a detailed backtest report"""
        if symbol not in self.results:
            logger.error(f"No backtest results found for {symbol}")
            return None
        
        results = self.results[symbol]
        
        # Create performance chart
        plt.figure(figsize=(15, 10))
        
        # Portfolio value over time
        plt.subplot(3, 1, 1)
        plt.plot(results['portfolio_values']['date'], results['portfolio_values']['portfolio_value'])
        plt.title(f'Portfolio Value for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Value (INR)')
        plt.grid(True)
        
        # Drawdown chart
        plt.subplot(3, 1, 2)
        portfolio_df = results['portfolio_values']
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax() - 1) * 100
        plt.fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, color='red', alpha=0.3)
        plt.title(f'Drawdown for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Monte Carlo simulation results if available
        if include_monte_carlo:
            monte_carlo_results = self.run_monte_carlo_simulation(symbol)
            
            if monte_carlo_results:
                plt.subplot(3, 1, 3)
                returns = [monte_carlo_results['return_percentiles'][f'{p}th'] for p in ['5th', '25th', '50th', '75th', '95th']]
                percentiles = ['5th', '25th', '50th', '75th', '95th']
                
                plt.bar(percentiles, returns)
                plt.axhline(y=results['total_return'], color='r', linestyle='-', label='Actual Return')
                plt.title(f'Monte Carlo Simulation Return Distribution for {symbol}')
                plt.xlabel('Percentile')
                plt.ylabel('Return')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = f'backtest_{symbol}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path)
        plt.close()
        
        # Create text report
        report = f"""
        Advanced Backtest Report for {symbol}
        ==========================================
        Period: {results['start_date']} to {results['end_date']}
        Walk-Forward Optimization: {'Yes' if results.get('walk_forward', False) else 'No'}
        
        Performance Metrics:
        - Initial Balance: INR {results['initial_balance']:.2f}
        - Final Balance: INR {results['final_balance']:.2f}
        - Total Return: {results['total_return']:.2%}
        - Annualized Return: {results['annualized_return']:.2%}
        - Sharpe Ratio: {results['sharpe_ratio']:.2f}
        - Maximum Drawdown: {results['max_drawdown']:.2%}
        - Win Rate: {results['win_rate']:.2f}%
        - Number of Trades: {results['num_trades']}
        - Transaction Costs: INR {results['transaction_costs']:.2f}
        
        """
        
        # Add Monte Carlo results if available
        if include_monte_carlo and monte_carlo_results:
            report += f"""
        Monte Carlo Simulation Results ({monte_carlo_results['num_simulations']} simulations):
        - Probability of Profit: {monte_carlo_results['prob_profit']:.2%}
        
        Return Percentiles:
        - 5th Percentile: {monte_carlo_results['return_percentiles']['5th']:.2%}
        - 25th Percentile: {monte_carlo_results['return_percentiles']['25th']:.2%}
        - 50th Percentile (Median): {monte_carlo_results['return_percentiles']['50th']:.2%}
        - 75th Percentile: {monte_carlo_results['return_percentiles']['75th']:.2%}
        - 95th Percentile: {monte_carlo_results['return_percentiles']['95th']:.2%}
        
        Drawdown Percentiles:
        - 5th Percentile: {monte_carlo_results['drawdown_percentiles']['5th']:.2%}
        - 25th Percentile: {monte_carlo_results['drawdown_percentiles']['25th']:.2%}
        - 50th Percentile (Median): {monte_carlo_results['drawdown_percentiles']['50th']:.2%}
        - 75th Percentile: {monte_carlo_results['drawdown_percentiles']['75th']:.2%}
        - 95th Percentile: {monte_carlo_results['drawdown_percentiles']['95th']:.2%}
        
        Probability of Exceeding Return Thresholds:
        """
            
            for threshold, prob in monte_carlo_results['prob_exceed_threshold'].items():
                report += f"- {threshold}: {prob:.2%}\n"
        
        report += """
        Trade History:
        """
        
        for i, trade in enumerate(results['trade_history'][:10]):  # Show first 10 trades
            report += f"""
        Trade {i+1}:
        - Symbol: {trade['symbol']}
        - Entry: {trade['entry_price']:.2f} on {trade['timestamp'].strftime('%Y-%m-%d')}
        - Exit: {trade['exit_price']:.2f} on {trade['exit_timestamp'].strftime('%Y-%m-%d')}
        - Quantity: {trade['quantity']}
        - P&L: INR {trade['final_pnl']:.2f}
        """
        
        if len(results['trade_history']) > 10:
            report += f"... and {len(results['trade_history']) - 10} more trades."
        
        # Save report
        report_path = f'backtest_report_{symbol}_{datetime.datetime.now().strftime("%Y%m%d")}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Advanced backtest report saved to {report_path}")
        
        return {
            'chart_path': chart_path,
            'report_path': report_path,
            'results': results,
            'monte_carlo_results': monte_carlo_results if include_monte_carlo else None
        }

# Define the trading strategies
class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data):
        """Generate trading signals based on the strategy"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return the name of the strategy"""
        pass

class AdvancedTrendFollowingStrategy(Strategy):
    """Advanced trend following strategy with multiple indicators and dynamic parameters"""
    
    def __init__(self):
        self.fast_ma_period = 10
        self.slow_ma_period = 30
        self.signal_ma_period = 9
        self.rsi_period = 14
        self.adx_period = 14
        self.atr_period = 14
        self.bb_period = 20
        self.bb_deviation = 2
        
    def get_name(self):
        return "Advanced Trend Following"
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on advanced trend following indicators"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate moving averages
        signals['fast_ma'] = talib.SMA(data['close'], timeperiod=self.fast_ma_period)
        signals['slow_ma'] = talib.SMA(data['close'], timeperiod=self.slow_ma_period)
        signals['signal_ma'] = talib.SMA(data['close'], timeperiod=self.signal_ma_period)
        
        # Calculate MACD
        signals['macd'], signals['macdsignal'], signals['macdhist'] = talib.MACD(data['close'])
        
        # Calculate RSI
        signals['rsi'] = talib.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Calculate ADX for trend strength
        signals['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=self.adx_period)
        
        # Calculate ATR for volatility
        signals['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.atr_period)
        signals['atr_pct'] = signals['atr'] / data['close'] * 100
        
        # Calculate Bollinger Bands
        signals['upper'], signals['middle'], signals['lower'] = talib.BBANDS(
            data['close'], timeperiod=self.bb_period, nbdevup=self.bb_deviation, nbdevdn=self.bb_deviation
        )
        
        # Calculate price position relative to Bollinger Bands
        signals['bb_position'] = (data['close'] - signals['lower']) / (signals['upper'] - signals['lower'])
        
        # Generate signals
        signals['signal'] = 0
        
        # Strong uptrend conditions
        strong_uptrend = (
            (signals['fast_ma'] > signals['slow_ma']) & 
            (signals['macd'] > signals['macdsignal']) & 
            (signals['adx'] > 25) & 
            (signals['rsi'] > 50) & 
            (signals['rsi'] < 70) &
            (signals['bb_position'] > 0.5)
        )
        
        # Weak uptrend conditions
        weak_uptrend = (
            (signals['fast_ma'] > signals['slow_ma']) & 
            (signals['macd'] > signals['macdsignal']) & 
            (signals['adx'] > 20) & 
            (signals['rsi'] > 40) & 
            (signals['rsi'] < 60)
        )
        
        # Strong downtrend conditions
        strong_downtrend = (
            (signals['fast_ma'] < signals['slow_ma']) & 
            (signals['macd'] < signals['macdsignal']) & 
            (signals['adx'] > 25) & 
            (signals['rsi'] < 50) & 
            (signals['rsi'] > 30) &
            (signals['bb_position'] < 0.5)
        )
        
        # Weak downtrend conditions
        weak_downtrend = (
            (signals['fast_ma'] < signals['slow_ma']) & 
            (signals['macd'] < signals['macdsignal']) & 
            (signals['adx'] > 20) & 
            (signals['rsi'] < 60) & 
            (signals['rsi'] > 40)
        )
        
        # Assign signals
        signals.loc[strong_uptrend, 'signal'] = 2  # Strong buy
        signals.loc[weak_uptrend, 'signal'] = 1   # Weak buy
        signals.loc[strong_downtrend, 'signal'] = -2  # Strong sell
        signals.loc[weak_downtrend, 'signal'] = -1   # Weak sell
        
        return signals

class AdvancedMeanReversionStrategy(Strategy):
    """Advanced mean reversion strategy with multiple indicators and dynamic parameters"""
    
    def __init__(self):
        self.bb_period = 20
        self.bb_deviation = 2
        self.rsi_period = 14
        self.stoch_period = 14
        self.stoch_slow_period = 3
        self.williams_period = 14
        self.cci_period = 14
        
    def get_name(self):
        return "Advanced Mean Reversion"
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on advanced mean reversion indicators"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate Bollinger Bands
        signals['upper'], signals['middle'], signals['lower'] = talib.BBANDS(
            data['close'], timeperiod=self.bb_period, nbdevup=self.bb_deviation, nbdevdn=self.bb_deviation
        )
        
        # Calculate BB width and position
        signals['bb_width'] = (signals['upper'] - signals['lower']) / signals['middle']
        signals['bb_position'] = (data['close'] - signals['lower']) / (signals['upper'] - signals['lower'])
        
        # Calculate RSI
        signals['rsi'] = talib.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Calculate Stochastic Oscillator
        signals['slowk'], signals['slowd'] = talib.STOCH(
            data['high'], data['low'], data['close'], 
            fastk_period=self.stoch_period, slowk_period=self.stoch_slow_period
        )
        
        # Calculate Williams %R
        signals['williams_r'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=self.williams_period)
        
        # Calculate CCI
        signals['cci'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=self.cci_period)
        
        # Generate signals
        signals['signal'] = 0
        
        # Strong oversold conditions
        strong_oversold = (
            (data['close'] < signals['lower']) & 
            (signals['rsi'] < 30) & 
            (signals['slowk'] < 20) & 
            (signals['williams_r'] < -80) & 
            (signals['cci'] < -200)
        )
        
        # Weak oversold conditions
        weak_oversold = (
            (data['close'] < signals['lower']) & 
            (signals['rsi'] < 40) & 
            (signals['slowk'] < 30) & 
            (signals['williams_r'] < -60)
        )
        
        # Strong overbought conditions
        strong_overbought = (
            (data['close'] > signals['upper']) & 
            (signals['rsi'] > 70) & 
            (signals['slowk'] > 80) & 
            (signals['williams_r'] > -20) & 
            (signals['cci'] > 200)
        )
        
        # Weak overbought conditions
        weak_overbought = (
            (data['close'] > signals['upper']) & 
            (signals['rsi'] > 60) & 
            (signals['slowk'] > 70) & 
            (signals['williams_r'] > -40)
        )
        
        # Assign signals
        signals.loc[strong_oversold, 'signal'] = 2  # Strong buy
        signals.loc[weak_oversold, 'signal'] = 1   # Weak buy
        signals.loc[strong_overbought, 'signal'] = -2  # Strong sell
        signals.loc[weak_overbought, 'signal'] = -1   # Weak sell
        
        return signals

class AdvancedMomentumStrategy(Strategy):
    """Advanced momentum strategy with multiple indicators and dynamic parameters"""
    
    def __init__(self):
        self.rsi_period = 14
        self.stoch_period = 14
        self.stoch_slow_period = 3
        self.momentum_period = 10
        self.roc_period = 10
        self.cci_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
    def get_name(self):
        return "Advanced Momentum"
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on advanced momentum indicators"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate RSI
        signals['rsi'] = talib.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Calculate Stochastic Oscillator
        signals['slowk'], signals['slowd'] = talib.STOCH(
            data['high'], data['low'], data['close'], 
            fastk_period=self.stoch_period, slowk_period=self.stoch_slow_period
        )
        
        # Calculate Momentum
        signals['momentum'] = talib.MOM(data['close'], timeperiod=self.momentum_period)
        
        # Calculate Rate of Change
        signals['roc'] = talib.ROC(data['close'], timeperiod=self.roc_period)
        
        # Calculate CCI
        signals['cci'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=self.cci_period)
        
        # Calculate MACD
        signals['macd'], signals['macdsignal'], signals['macdhist'] = talib.MACD(
            data['close'], fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )
        
        # Generate signals
        signals['signal'] = 0
        
        # Strong bullish momentum
        strong_bullish = (
            (signals['rsi'] > 50) & 
            (signals['rsi'] < 70) & 
            (signals['slowk'] > signals['slowd']) & 
            (signals['slowk'] < 80) & 
            (signals['momentum'] > 0) & 
            (signals['roc'] > 0) & 
            (signals['cci'] > 0) & 
            (signals['cci'] < 100) & 
            (signals['macd'] > signals['macdsignal'])
        )
        
        # Weak bullish momentum
        weak_bullish = (
            (signals['rsi'] > 40) & 
            (signals['rsi'] < 60) & 
            (signals['slowk'] > signals['slowd']) & 
            (signals['slowk'] < 70) & 
            (signals['momentum'] > 0) & 
            (signals['roc'] > 0) & 
            (signals['cci'] > -50) & 
            (signals['cci'] < 50)
        )
        
        # Strong bearish momentum
        strong_bearish = (
            (signals['rsi'] < 50) & 
            (signals['rsi'] > 30) & 
            (signals['slowk'] < signals['slowd']) & 
            (signals['slowk'] > 20) & 
            (signals['momentum'] < 0) & 
            (signals['roc'] < 0) & 
            (signals['cci'] < 0) & 
            (signals['cci'] > -100) & 
            (signals['macd'] < signals['macdsignal'])
        )
        
        # Weak bearish momentum
        weak_bearish = (
            (signals['rsi'] < 60) & 
            (signals['rsi'] > 40) & 
            (signals['slowk'] < signals['slowd']) & 
            (signals['slowk'] > 30) & 
            (signals['momentum'] < 0) & 
            (signals['roc'] < 0) & 
            (signals['cci'] < 50) & 
            (signals['cci'] > -50)
        )
        
        # Assign signals
        signals.loc[strong_bullish, 'signal'] = 2  # Strong buy
        signals.loc[weak_bullish, 'signal'] = 1   # Weak buy
        signals.loc[strong_bearish, 'signal'] = -2  # Strong sell
        signals.loc[weak_bearish, 'signal'] = -1   # Weak sell
        
        return signals

class BreakoutStrategy(Strategy):
    """Breakout strategy with volatility adjustment"""
    
    def __init__(self):
        self.bb_period = 20
        self.bb_deviation = 2
        self.atr_period = 14
        self.adx_period = 14
        self.volume_period = 20
        
    def get_name(self):
        return "Breakout"
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on breakout patterns"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate Bollinger Bands
        signals['upper'], signals['middle'], signals['lower'] = talib.BBANDS(
            data['close'], timeperiod=self.bb_period, nbdevup=self.bb_deviation, nbdevdn=self.bb_deviation
        )
        
        # Calculate ATR for volatility
        signals['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.atr_period)
        signals['atr_pct'] = signals['atr'] / data['close'] * 100
        
        # Calculate ADX for trend strength
        signals['adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=self.adx_period)
        
        # Calculate volume SMA
        signals['volume_sma'] = talib.SMA(data['volume'], timeperiod=self.volume_period)
        signals['volume_ratio'] = data['volume'] / signals['volume_sma']
        
        # Calculate Donchian Channels
        signals['donchian_upper'] = data['high'].rolling(self.bb_period).max()
        signals['donchian_lower'] = data['low'].rolling(self.bb_period).min()
        signals['donchian_middle'] = (signals['donchian_upper'] + signals['donchian_lower']) / 2
        
        # Generate signals
        signals['signal'] = 0
        
        # Breakout above Bollinger Band or Donchian Channel with confirmation
        upper_breakout = (
            (data['close'] > signals['upper']) & 
            (data['close'] > signals['donchian_upper']) & 
            (signals['adx'] > 25) & 
            (signals['volume_ratio'] > 1.2) & 
            (signals['atr_pct'] > 0.01)  # Sufficient volatility
        )
        
        # Breakdown below Bollinger Band or Donchian Channel with confirmation
        lower_breakdown = (
            (data['close'] < signals['lower']) & 
            (data['close'] < signals['donchian_lower']) & 
            (signals['adx'] > 25) & 
            (signals['volume_ratio'] > 1.2) & 
            (signals['atr_pct'] > 0.01)  # Sufficient volatility
        )
        
        # Assign signals
        signals.loc[upper_breakout, 'signal'] = 1  # Buy
        signals.loc[lower_breakdown, 'signal'] = -1  # Sell
        
        return signals

class StatisticalArbitrageStrategy(Strategy):
    """Statistical arbitrage strategy using cointegration and mean reversion"""
    
    def __init__(self, lookback_period=60):
        self.lookback_period = lookback_period
        self.z_score_threshold = 2.0
        self.half_life = None
        
    def get_name(self):
        return "Statistical Arbitrage"
    
    def calculate_half_life(self, series):
        """Calculate half-life of mean reversion"""
        # Calculate lagged series and returns
        lagged_series = series.shift(1).dropna()
        returns = series.diff().dropna()
        
        # Align series
        lagged_series = lagged_series[returns.index]
        
        # Regress returns against lagged series
        X = sm.add_constant(lagged_series)
        model = sm.OLS(returns, X).fit()
        
        # Calculate half-life
        half_life = -np.log(2) / model.params[1]
        
        return half_life
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on statistical arbitrage"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Calculate returns
        signals['returns'] = data['close'].pct_change()
        
        # Calculate rolling mean and standard deviation
        signals['rolling_mean'] = data['close'].rolling(self.lookback_period).mean()
        signals['rolling_std'] = data['close'].rolling(self.lookback_period).std()
        
        # Calculate z-score
        signals['z_score'] = (data['close'] - signals['rolling_mean']) / signals['rolling_std']
        
        # Calculate half-life of mean reversion
        if len(data) > self.lookback_period * 2:
            self.half_life = self.calculate_half_life(data['close'])
        
        # Generate signals
        signals['signal'] = 0
        
        # Buy signal when z-score is below threshold (oversold)
        buy_signal = signals['z_score'] < -self.z_score_threshold
        
        # Sell signal when z-score is above threshold (overbought)
        sell_signal = signals['z_score'] > self.z_score_threshold
        
        # Assign signals
        signals.loc[buy_signal, 'signal'] = 1  # Buy
        signals.loc[sell_signal, 'signal'] = -1  # Sell
        
        return signals

class MultiTimeframeStrategy(Strategy):
    """Multi-timeframe strategy combining signals from different timeframes"""
    
    def __init__(self):
        self.short_tf = 5   # 5-period (short-term)
        self.medium_tf = 20  # 20-period (medium-term)
        self.long_tf = 50    # 50-period (long-term)
        
    def get_name(self):
        return "Multi-Timeframe"
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on multiple timeframes"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        
        # Short-term indicators
        signals['short_ma'] = talib.SMA(data['close'], timeperiod=self.short_tf)
        signals['short_rsi'] = talib.RSI(data['close'], timeperiod=self.short_tf)
        
        # Medium-term indicators
        signals['medium_ma'] = talib.SMA(data['close'], timeperiod=self.medium_tf)
        signals['medium_rsi'] = talib.RSI(data['close'], timeperiod=self.medium_tf)
        signals['medium_macd'], signals['medium_macdsignal'], _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Long-term indicators
        signals['long_ma'] = talib.SMA(data['close'], timeperiod=self.long_tf)
        signals['long_rsi'] = talib.RSI(data['close'], timeperiod=self.long_tf)
        
        # Generate signals for each timeframe
        signals['short_signal'] = 0
        signals['medium_signal'] = 0
        signals['long_signal'] = 0
        
        # Short-term signals
        short_buy = (data['close'] > signals['short_ma']) & (signals['short_rsi'] > 50)
        short_sell = (data['close'] < signals['short_ma']) & (signals['short_rsi'] < 50)
        signals.loc[short_buy, 'short_signal'] = 1
        signals.loc[short_sell, 'short_signal'] = -1
        
        # Medium-term signals
        medium_buy = (data['close'] > signals['medium_ma']) & (signals['medium_macd'] > signals['medium_macdsignal']) & (signals['medium_rsi'] > 50)
        medium_sell = (data['close'] < signals['medium_ma']) & (signals['medium_macd'] < signals['medium_macdsignal']) & (signals['medium_rsi'] < 50)
        signals.loc[medium_buy, 'medium_signal'] = 1
        signals.loc[medium_sell, 'medium_signal'] = -1
        
        # Long-term signals
        long_buy = (data['close'] > signals['long_ma']) & (signals['long_rsi'] > 50)
        long_sell = (data['close'] < signals['long_ma']) & (signals['long_rsi'] < 50)
        signals.loc[long_buy, 'long_signal'] = 1
        signals.loc[long_sell, 'long_signal'] = -1
        
        # Combine signals with different weights
        # Short-term: 20%, Medium-term: 50%, Long-term: 30%
        signals['signal'] = (
            0.2 * signals['short_signal'] + 
            0.5 * signals['medium_signal'] + 
            0.3 * signals['long_signal']
        )
        
        # Convert to integer signals
        signals.loc[signals['signal'] > 0.5, 'signal'] = 1
        signals.loc[signals['signal'] < -0.5, 'signal'] = -1
        signals.loc[(signals['signal'] >= -0.5) & (signals['signal'] <= 0.5), 'signal'] = 0
        
        return signals

class UltraAdvancedTradingBot:
    """Ultra-advanced trading bot with state-of-the-art AI and risk management"""
    
    def __init__(self, api_key, api_secret, symbols, simulation_mode=True):
        self.simulation_mode = simulation_mode
        
        if simulation_mode:
            logger.info("Initializing ultra-advanced trading bot in SIMULATION mode")
            self.data_fetcher = AdvancedMarketDataSimulator(symbols)
            self.broker = SimulatedBroker(BUDGET)
        else:
            logger.info("Initializing ultra-advanced trading bot in LIVE mode")
            self.data_fetcher = MarketDataFetcher(api_key, api_secret)
            self.broker = BrokerInterface(api_key, api_secret)
        
        self.risk_manager = UltraAdvancedAIRiskManager()
        self.position_sizer = AdvancedPositionSizer()
        self.performance_tracker = PerformanceTracker()
        self.backtester = AdvancedBacktester(self.data_fetcher, [], self.risk_manager, self.position_sizer)
        
        # Initialize strategies
        self.strategies = [
            AdvancedTrendFollowingStrategy(),
            AdvancedMeanReversionStrategy(),
            AdvancedMomentumStrategy(),
            BreakoutStrategy(),
            StatisticalArbitrageStrategy(),
            MultiTimeframeStrategy()
        ]
        
        # Update backtester with strategies
        self.backtester.strategies = self.strategies
        
        # Initialize strategy selector
        self.strategy_selector = AdvancedStrategySelector()
        
        # Symbols to trade
        self.symbols = symbols
        
        # Account balance
        self.account_balance = BUDGET
        
        # Flag to control bot operation
        self.running = False
        
        # Dashboard app
        self.dashboard_app = Flask(__name__)
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup the dashboard routes"""
        
        @self.dashboard_app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.dashboard_app.route('/api/status')
        def api_status():
            if self.simulation_mode:
                account_summary = self.broker.get_account_summary()
                positions = self.broker.get_positions()
                metrics = self.performance_tracker.metrics
                
                return jsonify({
                    'running': self.running,
                    'simulation_mode': self.simulation_mode,
                    'account_balance': account_summary['cash_balance'],
                    'portfolio_value': account_summary['total_value'],
                    'open_positions': len(positions),
                    'total_pnl': account_summary['total_pnl'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'transaction_costs': account_summary['transaction_costs']
                })
            else:
                positions = self.broker.get_positions()
                metrics = self.performance_tracker.metrics
                
                return jsonify({
                    'running': self.running,
                    'simulation_mode': self.simulation_mode,
                    'account_balance': self.account_balance,
                    'open_positions': len(positions),
                    'total_pnl': metrics['total_pnl'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades']
                })
        
        @self.dashboard_app.route('/api/positions')
        def api_positions():
            positions = self.broker.get_positions()
            return jsonify(positions)
        
        @self.dashboard_app.route('/api/performance')
        def api_performance():
            report = self.performance_tracker.get_daily_report()
            return jsonify(report)
        
        @self.dashboard_app.route('/api/backtest', methods=['POST'])
        def api_backtest():
            from flask import request
            data = request.json
            symbol = data.get('symbol')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            walk_forward = data.get('walk_forward', True)
            
            if not all([symbol, start_date, end_date]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            try:
                results = self.backtester.run_backtest(symbol, start_date, end_date, walk_forward=walk_forward)
                if results:
                    return jsonify({'success': True, 'results': results})
                else:
                    return jsonify({'error': 'Backtest failed'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.dashboard_app.route('/api/monte_carlo', methods=['POST'])
        def api_monte_carlo():
            from flask import request
            data = request.json
            symbol = data.get('symbol')
            num_simulations = data.get('num_simulations', 1000)
            
            if not symbol:
                return jsonify({'error': 'Missing symbol parameter'}), 400
            
            try:
                results = self.backtester.run_monte_carlo_simulation(symbol, num_simulations)
                if results:
                    return jsonify({'success': True, 'results': results})
                else:
                    return jsonify({'error': 'Monte Carlo simulation failed'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread"""
        def run_dashboard():
            self.dashboard_app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        logger.info("Dashboard started at http://127.0.0.1:5000")
    
    def initialize(self):
        """Initialize the trading bot"""
        logger.info("Initializing ultra-advanced trading bot...")
        
        # Fetch historical data for training AI models
        all_data = {}
        for symbol in self.symbols:
            data = self.data_fetcher.get_historical_data(symbol)
            all_data[symbol] = data
        
        # Train AI risk manager
        combined_data = pd.concat([df for df in all_data.values()])
        self.risk_manager.train(combined_data)
        
        # Train strategy selector
        self.strategy_selector.train(combined_data, self.strategies)
        
        # Try to load pre-trained models if available
        if os.path.exists('models/risk_manager'):
            self.risk_manager.load_models('models/risk_manager')
        
        if os.path.exists('models/strategy_selector'):
            self.strategy_selector.load_model('models/strategy_selector')
        
        # Start dashboard
        self.start_dashboard()
        
        logger.info("Ultra-advanced trading bot initialized successfully")
    
    def is_market_hours(self):
        """Check if current time is within market hours"""
        now = datetime.datetime.now().time()
        return MARKET_START_TIME <= now <= MARKET_END_TIME
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting ultra-advanced trading bot...")
        self.running = True
        
        # Schedule daily report
        schedule.every().day.at("15:45").do(self.generate_daily_report)
        
        try:
            while self.running:
                # Check if it's market hours
                if self.is_market_hours():
                    # Process each symbol
                    for symbol in self.symbols:
                        # Get real-time data
                        data = self.data_fetcher.get_real_time_data(symbol)
                        
                        # Update existing positions
                        self.broker.update_position(symbol, data['price'])
                        
                        # Update portfolio heat for position sizing
                        if self.simulation_mode:
                            self.position_sizer.update_portfolio_heat(
                                self.broker.get_positions(), 
                                self.broker.get_account_summary()['total_value']
                            )
                        
                        # Check if we can open new positions
                        current_positions = self.broker.get_positions()
                        if len(current_positions) < MAX_POSITIONS and symbol not in current_positions:
                            # Get historical data for signal generation
                            hist_data = self.data_fetcher.get_historical_data(symbol, days=252)
                            
                            # Get strategy weights
                            strategy_weights = self.strategy_selector.select_strategy(
                              data, hist_data
                            )
                            
                            # Generate signals from all strategies
                            signals = {}
                            signal_strengths = {}
                            
                            for strategy in self.strategies:
                                strategy_signals = strategy.generate_signals(hist_data)
                                latest_signal = strategy_signals.iloc[-1]
                                signals[strategy.get_name()] = latest_signal['signal']
                                
                                # Store signal strength if available
                                if 'signal_strength' in latest_signal:
                                    signal_strengths[strategy.get_name()] = latest_signal['signal_strength']
                                else:
                                    signal_strengths[strategy.get_name()] = abs(latest_signal['signal'])
                            
                            # Combine signals using strategy weights
                            weighted_signal = 0
                            weighted_strength = 0
                            
                            for name, signal in signals.items():
                                weight = strategy_weights.get(name, 0)
                                strength = signal_strengths.get(name, abs(signal))
                                weighted_signal += signal * weight
                                weighted_strength += strength * weight
                            
                            # Make decision based on weighted signal and strength
                            if weighted_signal > 0.5 and weighted_strength > 0.7:  # Strong buy signal
                                # Calculate position size
                                volatility = hist_data['close'].pct_change().std()
                                position_size = self.position_sizer.calculate_position_size(
                                    data['price'], volatility, 
                                    self.broker.cash_balance if self.simulation_mode else self.account_balance,
                                    current_positions, 
                                    self.risk_manager.market_regime_detector.current_regime
                                )
                                
                                # Get AI-based stop-loss and take-profit
                                stop_loss, take_profit = self.risk_manager.get_stop_loss_take_profit(
                                    data, hist_data
                                )
                                
                                # Open position
                                position = self.broker.open_position(
                                    symbol=symbol,
                                    quantity=position_size,
                                    price=data['price'],
                                    stop_loss=stop_loss,
                                    take_profit=take_profit
                                )
                                
                                if position:
                                    logger.info(f"Opened position for {symbol}: {position_size} shares at {data['price']}")
                                    logger.info(f"Strategy weights: {strategy_weights}")
                                    logger.info(f"Weighted signal: {weighted_signal:.2f}, strength: {weighted_strength:.2f}")
                        
                        # Small delay to prevent API rate limiting
                        time.sleep(1)
                else:
                    # Not market hours, wait and check again
                    time.sleep(60)
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.running = False
            logger.info("Trading bot stopped")
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping ultra-advanced trading bot...")
        self.running = False
    
    def generate_daily_report(self):
        """Generate and save daily performance report"""
        logger.info("Generating daily performance report...")
        
        # Get daily report
        report = self.performance_tracker.get_daily_report()
        
        # Generate performance chart
        chart_path = self.performance_tracker.generate_performance_chart()
        
        # Save report to file
        report_date = datetime.datetime.now().strftime('%Y-%m-%d')
        report_file = f"daily_report_{report_date}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Daily report saved to {report_file}")
        
        # Print report to console
        print("\n" + "="*50)
        print(f"DAILY TRADING REPORT - {report_date}")
        print("="*50)
        if self.simulation_mode:
            account_summary = self.broker.get_account_summary()
            print(f"Initial Balance: INR {account_summary['initial_balance']:.2f}")
            print(f"Current Balance: INR {account_summary['total_value']:.2f}")
            print(f"Transaction Costs: INR {account_summary['transaction_costs']:.2f}")
        print(f"Daily P&L: INR {report['daily_pnl']:.2f}")
        print(f"Total P&L: INR {report['total_pnl']:.2f}")
        print(f"Win Rate: {report['win_rate']:.2f}%")
        print(f"Total Trades: {report['total_trades']}")
        print(f"Max Drawdown: {report['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print("="*50 + "\n")
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving trained models...")
        
        # Save risk manager models
        self.risk_manager.save_models('models/risk_manager')
        
        # Save strategy selector model
        self.strategy_selector.save_model('models/strategy_selector')
        
        logger.info("All models saved successfully")
    
    def run_backtest(self, symbol, start_date, end_date, walk_forward=True):
        """Run backtest for a symbol"""
        return self.backtester.run_backtest(symbol, start_date, end_date, walk_forward=walk_forward)

# HTML template for dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra-Advanced Trading Bot Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .status-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            width: 23%;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        .neutral {
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .refresh-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            float: right;
        }
        .refresh-btn:hover {
            background-color: #45a049;
        }
        .tab-container {
            margin-top: 20px;
        }
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }
        .tab button:hover {
            background-color: #ddd;
        }
        .tab button.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
        }
        .backtest-form, .monte-carlo-form {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .simulation-indicator {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            margin-left: 10px;
        }
        .simulation-mode {
            background-color: #ffeb3b;
            color: #333;
        }
        .live-mode {
            background-color: #f44336;
            color: white;
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
        }
        .model-info {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .model-info h3 {
            margin-top: 0;
            color: #2e7d32;
        }
        .feature-importance {
            margin-top: 10px;
        }
        .feature-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .feature-bar {
            height: 20px;
            background-color: #4CAF50;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            Ultra-Advanced Trading Bot Dashboard
            <span class="simulation-indicator" id="mode-indicator">SIMULATION</span>
        </h1>
        <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
        
        <div class="status">
            <div class="status-card">
                <div>Bot Status</div>
                <div class="status-value neutral" id="bot-status">Loading...</div>
            </div>
            <div class="status-card">
                <div>Account Balance</div>
                <div class="status-value neutral" id="account-balance">Loading...</div>
            </div>
            <div class="status-card">
                <div>Portfolio Value</div>
                <div class="status-value neutral" id="portfolio-value">Loading...</div>
            </div>
            <div class="status-card">
                <div>Total P&L</div>
                <div class="status-value neutral" id="total-pnl">Loading...</div>
            </div>
            <div class="status-card">
                <div>Win Rate</div>
                <div class="status-value neutral" id="win-rate">Loading...</div>
            </div>
            <div class="status-card">
                <div>Total Trades</div>
                <div class="status-value neutral" id="total-trades">Loading...</div>
            </div>
            <div class="status-card">
                <div>Transaction Costs</div>
                <div class="status-value neutral" id="transaction-costs">Loading...</div>
            </div>
            <div class="status-card">
                <div>Open Positions</div>
                <div class="status-value neutral" id="open-positions">Loading...</div>
            </div>
        </div>
        
        <div class="tab-container">
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'positions-tab')">Positions</button>
                <button class="tablinks" onclick="openTab(event, 'performance-tab')">Performance</button>
                <button class="tablinks" onclick="openTab(event, 'backtest-tab')">Backtesting</button>
                <button class="tablinks" onclick="openTab(event, 'monte-carlo-tab')">Monte Carlo</button>
                <button class="tablinks" onclick="openTab(event, 'ai-models-tab')">AI Models</button>
            </div>
            
            <div id="positions-tab" class="tabcontent" style="display: block;">
                <h2>Open Positions</h2>
                <table id="positions-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th>Stop Loss</th>
                            <th>Take Profit</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="8">Loading positions...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div id="performance-tab" class="tabcontent">
                <h2>Performance Metrics</h2>
                <table id="performance-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Daily P&L</td>
                            <td id="daily-pnl">Loading...</td>
                        </tr>
                        <tr>
                            <td>Total P&L</td>
                            <td id="total-pnl-performance">Loading...</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td id="max-drawdown">Loading...</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td id="sharpe-ratio">Loading...</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td id="win-rate-performance">Loading...</td>
                        </tr>
                        <tr>
                            <td>Total Trades</td>
                            <td id="total-trades-performance">Loading...</td>
                        </tr>
                    </tbody>
                </table>
                
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
            
            <div id="backtest-tab" class="tabcontent">
                <h2>Advanced Backtesting</h2>
                <div class="backtest-form">
                    <div class="form-group">
                        <label for="backtest-symbol">Symbol</label>
                        <select id="backtest-symbol">
                            <option value="RELIANCE">RELIANCE</option>
                            <option value="TCS">TCS</option>
                            <option value="HDFCBANK">HDFCBANK</option>
                            <option value="INFY">INFY</option>
                            <option value="HINDUNILVR">HINDUNILVR</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="start-date">Start Date</label>
                        <input type="date" id="start-date" value="2023-01-01">
                    </div>
                    <div class="form-group">
                        <label for="end-date">End Date</label>
                        <input type="date" id="end-date" value="2023-12-31">
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="walk-forward" checked> Walk-Forward Optimization
                        </label>
                    </div>
                    <button class="btn" onclick="runBacktest()">Run Backtest</button>
                </div>
                <div id="backtest-results"></div>
            </div>
            
            <div id="monte-carlo-tab" class="tabcontent">
                <h2>Monte Carlo Simulation</h2>
                <div class="monte-carlo-form">
                    <div class="form-group">
                        <label for="monte-carlo-symbol">Symbol</label>
                        <select id="monte-carlo-symbol">
                            <option value="RELIANCE">RELIANCE</option>
                            <option value="TCS">TCS</option>
                            <option value="HDFCBANK">HDFCBANK</option>
                            <option value="INFY">INFY</option>
                            <option value="HINDUNILVR">HINDUNILVR</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="num-simulations">Number of Simulations</label>
                        <input type="number" id="num-simulations" value="1000" min="100" max="10000">
                    </div>
                    <button class="btn" onclick="runMonteCarlo()">Run Simulation</button>
                </div>
                <div id="monte-carlo-results"></div>
            </div>
            
            <div id="ai-models-tab" class="tabcontent">
                <h2>AI Model Information</h2>
                <div class="model-info">
                    <h3>Risk Management Models</h3>
                    <p>The trading bot uses an ensemble of advanced AI models for risk management:</p>
                    <ul>
                        <li>Random Forest</li>
                        <li>Gradient Boosting</li>
                        <li>XGBoost</li>
                        <li>LightGBM</li>
                        <li>Multi-Layer Perceptron</li>
                        <li>Support Vector Regression</li>
                        <li>Long Short-Term Memory (LSTM)</li>
                        <li>1D Convolutional Neural Network (CNN)</li>
                    </ul>
                    
                    <div class="feature-importance">
                        <h4>Top Features for Risk Management:</h4>
                        <div id="feature-importance-list">
                            <div class="feature-item">
                                <span>Feature 1</span>
                                <div class="feature-bar" style="width: 90%"></div>
                            </div>
                            <div class="feature-item">
                                <span>Feature 2</span>
                                <div class="feature-bar" style="width: 80%"></div>
                            </div>
                            <div class="feature-item">
                                <span>Feature 3</span>
                                <div class="feature-bar" style="width: 70%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="model-info">
                    <h3>Strategy Selection Models</h3>
                    <p>The trading bot uses an ensemble of classification models to select the optimal strategy based on market conditions:</p>
                    <ul>
                        <li>Random Forest Classifier</li>
                        <li>Gradient Boosting Classifier</li>
                        <li>XGBoost Classifier</li>
                        <li>LightGBM Classifier</li>
                    </ul>
                    
                    <p>The models are trained to recognize different market regimes and select the most appropriate strategy for current conditions.</p>
                </div>
                
                <div class="model-info">
                    <h3>Market Regime Detection</h3>
                    <p>The trading bot uses statistical methods to detect current market regimes:</p>
                    <ul>
                        <li>Trending Up</li>
                        <li>Trending Down</li>
                        <li>Ranging</li>
                        <li>High Volatility</li>
                        <li>Low Volatility</li>
                    </ul>
                    
                    <p>Each regime has different strategy weights and risk parameters to optimize performance.</p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        function refreshData() {
            // Fetch status
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('bot-status').textContent = data.running ? 'Running' : 'Stopped';
                    document.getElementById('bot-status').className = 'status-value ' + (data.running ? 'positive' : 'negative');
                    
                    document.getElementById('account-balance').textContent = 'INR ' + data.account_balance.toFixed(2);
                    
                    if (data.portfolio_value !== undefined) {
                        document.getElementById('portfolio-value').textContent = 'INR ' + data.portfolio_value.toFixed(2);
                    }
                    
                    const pnlElement = document.getElementById('total-pnl');
                    pnlElement.textContent = 'INR ' + data.total_pnl.toFixed(2);
                    pnlElement.className = 'status-value ' + (data.total_pnl >= 0 ? 'positive' : 'negative');
                    
                    document.getElementById('win-rate').textContent = data.win_rate.toFixed(2) + '%';
                    document.getElementById('total-trades').textContent = data.total_trades;
                    document.getElementById('open-positions').textContent = data.open_positions;
                    
                    if (data.transaction_costs !== undefined) {
                        document.getElementById('transaction-costs').textContent = 'INR ' + data.transaction_costs.toFixed(2);
                    }
                    
                    // Update mode indicator
                    const modeIndicator = document.getElementById('mode-indicator');
                    if (data.simulation_mode) {
                        modeIndicator.textContent = 'SIMULATION';
                        modeIndicator.className = 'simulation-indicator simulation-mode';
                    } else {
                        modeIndicator.textContent = 'LIVE';
                        modeIndicator.className = 'simulation-indicator live-mode';
                    }
                });
            
            // Fetch positions
            fetch('/api/positions')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.querySelector('#positions-table tbody');
                    tbody.innerHTML = '';
                    
                    if (Object.keys(data).length === 0) {
                        tbody.innerHTML = '<tr><td colspan="8">No open positions</td></tr>';
                        return;
                    }
                    
                    for (const symbol in data) {
                        const position = data[symbol];
                        const row = document.createElement('tr');
                        
                        const pnlClass = position.pnl >= 0 ? 'positive' : 'negative';
                        
                        row.innerHTML = `
                            <td>${symbol}</td>
                            <td>${position.quantity}</td>
                            <td>INR ${position.entry_price.toFixed(2)}</td>
                            <td>INR ${position.current_price.toFixed(2)}</td>
                            <td>${(position.stop_loss * 100).toFixed(2)}%</td>
                            <td>${(position.take_profit * 100).toFixed(2)}%</td>
                            <td class="${pnlClass}">INR ${position.pnl.toFixed(2)}</td>
                            <td class="${pnlClass}">${position.pnl_percent.toFixed(2)}%</td>
                        `;
                        
                        tbody.appendChild(row);
                    }
                });
            
            // Fetch performance
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    const dailyPnlElement = document.getElementById('daily-pnl');
                    dailyPnlElement.textContent = 'INR ' + data.daily_pnl.toFixed(2);
                    dailyPnlElement.className = data.daily_pnl >= 0 ? 'positive' : 'negative';
                    
                    const totalPnlElement = document.getElementById('total-pnl-performance');
                    totalPnlElement.textContent = 'INR ' + data.total_pnl.toFixed(2);
                    totalPnlElement.className = data.total_pnl >= 0 ? 'positive' : 'negative';
                    
                    document.getElementById('total-trades-performance').textContent = data.total_trades;
                    document.getElementById('win-rate-performance').textContent = data.win_rate.toFixed(2) + '%';
                    
                    const drawdownElement = document.getElementById('max-drawdown');
                    drawdownElement.textContent = data.max_drawdown.toFixed(2) + '%';
                    drawdownElement.className = data.max_drawdown <= 0 ? 'positive' : 'negative';
                    
                    document.getElementById('sharpe-ratio').textContent = data.sharpe_ratio.toFixed(2);
                    
                    // Update performance chart
                    updatePerformanceChart();
                });
        }
        
        function runBacktest() {
            const symbol = document.getElementById('backtest-symbol').value;
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const walkForward = document.getElementById('walk-forward').checked;
            
            if (!symbol || !startDate || !endDate) {
                alert('Please fill in all fields');
                return;
            }
            
            const resultsDiv = document.getElementById('backtest-results');
            resultsDiv.innerHTML = '<p>Running backtest...</p>';
            
            fetch('/api/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    start_date: startDate,
                    end_date: endDate,
                    walk_forward: walkForward
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const results = data.results;
                    resultsDiv.innerHTML = `
                        <h3>Backtest Results for ${symbol}</h3>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Initial Balance</td>
                                <td>INR ${results.initial_balance.toFixed(2)}</td>
                            </tr>
                            <tr>
                                <td>Final Balance</td>
                                <td>INR ${results.final_balance.toFixed(2)}</td>
                            </tr>
                            <tr>
                                <td>Total Return</td>
                                <td>${(results.total_return * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Annualized Return</td>
                                <td>${(results.annualized_return * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>${results.sharpe_ratio.toFixed(2)}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td>${(results.max_drawdown * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Win Rate</td>
                                <td>${results.win_rate.toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Number of Trades</td>
                                <td>${results.num_trades}</td>
                            </tr>
                            <tr>
                                <td>Transaction Costs</td>
                                <td>INR ${results.transaction_costs.toFixed(2)}</td>
                            </tr>
                            <tr>
                                <td>Walk-Forward Optimization</td>
                                <td>${results.walk_forward ? 'Yes' : 'No'}</td>
                            </tr>
                        </table>
                    `;
                } else {
                    resultsDiv.innerHTML = `<p class="negative">Error: ${data.error}</p>`;
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = `<p class="negative">Error: ${error.message}</p>`;
            });
        }
        
        function runMonteCarlo() {
            const symbol = document.getElementById('monte-carlo-symbol').value;
            const numSimulations = document.getElementById('num-simulations').value;
            
            if (!symbol || !numSimulations) {
                alert('Please fill in all fields');
                return;
            }
            
            const resultsDiv = document.getElementById('monte-carlo-results');
            resultsDiv.innerHTML = '<p>Running Monte Carlo simulation...</p>';
            
            fetch('/api/monte_carlo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    num_simulations: parseInt(numSimulations)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const results = data.results;
                    resultsDiv.innerHTML = `
                        <h3>Monte Carlo Simulation Results for ${symbol}</h3>
                        <p>Number of simulations: ${results.num_simulations}</p>
                        
                        <h4>Return Distribution</h4>
                        <table>
                            <tr>
                                <th>Percentile</th>
                                <th>Return</th>
                            </tr>
                            <tr>
                                <td>5th Percentile</td>
                                <td>${(results.return_percentiles['5th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>25th Percentile</td>
                                <td>${(results.return_percentiles['25th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>50th Percentile (Median)</td>
                                <td>${(results.return_percentiles['50th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>75th Percentile</td>
                                <td>${(results.return_percentiles['75th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>95th Percentile</td>
                                <td>${(results.return_percentiles['95th'] * 100).toFixed(2)}%</td>
                            </tr>
                        </table>
                        
                        <h4>Drawdown Distribution</h4>
                        <table>
                            <tr>
                                <th>Percentile</th>
                                <th>Drawdown</th>
                            </tr>
                            <tr>
                                <td>5th Percentile</td>
                                <td>${(results.drawdown_percentiles['5th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>25th Percentile</td>
                                <td>${(results.drawdown_percentiles['25th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>50th Percentile (Median)</td>
                                <td>${(results.drawdown_percentiles['50th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>75th Percentile</td>
                                <td>${(results.drawdown_percentiles['75th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>95th Percentile</td>
                                <td>${(results.drawdown_percentiles['95th'] * 100).toFixed(2)}%</td>
                            </tr>
                        </table>
                        
                        <h4>Probabilities</h4>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Probability</th>
                            </tr>
                            <tr>
                                <td>Probability of Profit</td>
                                <td>${(results.prob_profit * 100).toFixed(2)}%</td>
                            </tr>
                        `;
                    
                    // Add probability of exceeding thresholds
                    for (const [threshold, prob] of Object.entries(results.prob_exceed_threshold)) {
                        resultsDiv.innerHTML += `
                            <tr>
                                <td>${threshold.replace('_', ' ')}</td>
                                <td>${(prob * 100).toFixed(2)}%</td>
                            </tr>
                        `;
                    }
                    
                    resultsDiv.innerHTML += '</table>';
                    
                    // Compare with actual results
                    resultsDiv.innerHTML += `
                        <h4>Comparison with Actual Results</h4>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Actual</th>
                                <th>Median Simulation</th>
                            </tr>
                            <tr>
                                <td>Return</td>
                                <td>${(results.actual_return * 100).toFixed(2)}%</td>
                                <td>${(results.return_percentiles['50th'] * 100).toFixed(2)}%</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td>${(results.actual_max_drawdown * 100).toFixed(2)}%</td>
                                <td>${(results.drawdown_percentiles['50th'] * 100).toFixed(2)}%</td>
                            </tr>
                        </table>
                    `;
                } else {
                    resultsDiv.innerHTML = `<p class="negative">Error: ${data.error}</p>`;
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = `<p class="negative">Error: ${error.message}</p>`;
            });
        }
        
        function updatePerformanceChart() {
            // This would typically fetch performance data and update the chart
            // For now, we'll just create a placeholder chart
            const ctx = document.getElementById('performance-chart').getContext('2d');
            
            // Check if chart already exists
            if (window.performanceChart) {
                window.performanceChart.destroy();
            }
            
            // Create new chart
            window.performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [7000, 7200, 7100, 7500, 7800, 7600, 8000, 8200, 8500, 8300, 8700, 9000],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Portfolio Performance Over Time'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: 'Value (INR)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Month'
                            }
                        }
                    }
                }
            });
        }
        
        // Initial load and refresh every 30 seconds
        refreshData();
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
"""

def save_dashboard_template():
    """Save the dashboard template to a file"""
    os.makedirs('templates', exist_ok=True)
    with open('templates/dashboard.html', 'w') as f:
        f.write(DASHBOARD_TEMPLATE)

# Main execution
if __name__ == "__main__":
    # Save dashboard template
    save_dashboard_template()
    
    # Symbols to trade (example NSE stocks)
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
        'ICICIBANK', 'KOTAKBANK', 'LT', 'ITC', 'AXISBANK'
    ]
    
    # Create and initialize trading bot in simulation mode
    bot = UltraAdvancedTradingBot(API_KEY, API_SECRET, symbols, simulation_mode=True)
    bot.initialize()
    
    # Run the bot
    bot.run()