"""
ML/AI Components for Enhanced Trading Bot
Includes: LSTM, Random Forest, XGBoost, RL, Sentiment Analysis, Ensemble Methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests
from textblob import TextBlob
import yfinance as yf
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LSTMPricePredictor:
    """LSTM Neural Network for price prediction"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Use OHLCV data
        features = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.fit_transform(data[features].values)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 3])  # Close price
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train LSTM model"""
        try:
            if len(data) < self.sequence_length + 50:
                logger.warning("Insufficient data for LSTM training")
                return False
                
            X, y = self.prepare_data(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            
            self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.is_trained = True
            logger.info("LSTM model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return False
    
    def predict_price(self, data: pd.DataFrame) -> Optional[float]:
        """Predict next price using LSTM"""
        if not self.is_trained or self.model is None:
            return None
            
        try:
            features = ['open', 'high', 'low', 'close', 'volume']
            if len(data) < self.sequence_length:
                return None
                
            # Prepare last sequence
            scaled_data = self.scaler.transform(data[features].tail(self.sequence_length).values)
            X = scaled_data.reshape(1, self.sequence_length, len(features))
            
            # Predict
            prediction = self.model.predict(X, verbose=0)[0][0]
            
            # Inverse transform to get actual price
            dummy_array = np.zeros((1, len(features)))
            dummy_array[0, 3] = prediction
            actual_price = self.scaler.inverse_transform(dummy_array)[0, 3]
            
            return float(actual_price)
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None
    
    def save_model(self, path: str):
        """Save LSTM model"""
        if self.model:
            self.model.save(f"{path}_lstm.h5")
            joblib.dump(self.scaler, f"{path}_scaler.pkl")
    
    def load_model(self, path: str):
        """Load LSTM model"""
        try:
            self.model = tf.keras.models.load_model(f"{path}_lstm.h5")
            self.scaler = joblib.load(f"{path}_scaler.pkl")
            self.is_trained = True
            return True
        except:
            return False

class SignalClassifier:
    """Random Forest and XGBoost for signal classification"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for classification"""
        df = data.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df
    
    def create_labels(self, data: pd.DataFrame, forward_days: int = 3) -> pd.Series:
        """Create labels for classification (future returns)"""
        future_returns = data['close'].shift(-forward_days) / data['close'] - 1
        
        # Create labels: 0=SELL, 1=HOLD, 2=BUY
        labels = pd.Series(1, index=data.index)  # Default HOLD
        labels[future_returns < -0.02] = 0  # SELL if drops > 2%
        labels[future_returns > 0.02] = 2   # BUY if rises > 2%
        
        return labels
    
    def train(self, data: pd.DataFrame) -> bool:
        """Train classification models"""
        try:
            # Create features and labels
            df_features = self.create_features(data)
            labels = self.create_labels(data)
            
            # Select feature columns
            feature_cols = [col for col in df_features.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Remove NaN values
            df_clean = df_features[feature_cols].dropna()
            labels_clean = labels.loc[df_clean.index]
            
            if len(df_clean) < 100:
                logger.warning("Insufficient data for signal classification training")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(df_clean)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, labels_clean, test_size=0.2, random_state=42, stratify=labels_clean
            )
            
            # Train models
            self.rf_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            
            # Evaluate
            rf_score = self.rf_model.score(X_test, y_test)
            xgb_score = self.xgb_model.score(X_test, y_test)
            
            logger.info(f"Signal Classifier trained - RF: {rf_score:.3f}, XGB: {xgb_score:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Signal classification training failed: {e}")
            return False
    
    def predict_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """Predict trading signal"""
        if not self.is_trained:
            return {'signal': 1, 'confidence': 0.0, 'probabilities': [0.33, 0.34, 0.33]}
        
        try:
            df_features = self.create_features(data)
            feature_cols = [col for col in df_features.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Get last row features
            last_features = df_features[feature_cols].iloc[-1:].dropna(axis=1)
            
            if last_features.empty:
                return {'signal': 1, 'confidence': 0.0, 'probabilities': [0.33, 0.34, 0.33]}
            
            # Scale features
            X_scaled = self.scaler.transform(last_features)
            
            # Get predictions from both models
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
            
            # Ensemble prediction (average)
            avg_proba = (rf_proba + xgb_proba) / 2
            signal = np.argmax(avg_proba)
            confidence = np.max(avg_proba)
            
            return {
                'signal': int(signal),  # 0=SELL, 1=HOLD, 2=BUY
                'confidence': float(confidence),
                'probabilities': avg_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Signal prediction failed: {e}")
            return {'signal': 1, 'confidence': 0.0, 'probabilities': [0.33, 0.34, 0.33]}

class RLPositionSizer:
    """Reinforcement Learning for position sizing"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.state_size = 10
        self.action_size = 5  # 0%, 25%, 50%, 75%, 100% allocation
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural network for Q-learning"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def get_state(self, data: pd.DataFrame, portfolio_value: float, current_positions: int) -> np.ndarray:
        """Create state representation"""
        if len(data) < 20:
            return np.zeros(self.state_size)
        
        # Calculate features
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)
        rsi = self._calculate_rsi(data['close']).iloc[-1]
        
        state = np.array([
            portfolio_value / self.capital,  # Portfolio ratio
            current_positions / 3,  # Position ratio (max 3)
            volatility if not np.isnan(volatility) else 0,
            momentum if not np.isnan(momentum) else 0,
            rsi / 100 if not np.isnan(rsi) else 0.5,
            returns.iloc[-1] if not np.isnan(returns.iloc[-1]) else 0,
            returns.iloc[-5:].mean() if len(returns) >= 5 else 0,
            data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] if not np.isnan(data['volume'].rolling(20).mean().iloc[-1]) else 1,
            (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1],
            len(data) / 252  # Time factor
        ])
        
        return np.nan_to_num(state)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_position_size(self, data: pd.DataFrame, price: float, portfolio_value: float, current_positions: int) -> int:
        """Get position size using RL"""
        try:
            state = self.get_state(data, portfolio_value, current_positions)
            
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                q_values = self.model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            
            # Convert action to allocation percentage
            allocation_pct = [0, 0.25, 0.5, 0.75, 1.0][action]
            max_allocation = portfolio_value * 0.3  # Max 30% per trade
            allocation = max_allocation * allocation_pct
            
            quantity = int(allocation / price)
            
            # Minimum trade size
            if quantity * price < 1000:
                return 0
                
            return quantity
            
        except Exception as e:
            logger.error(f"RL position sizing failed: {e}")
            # Fallback to simple calculation
            return int((portfolio_value * 0.1) / price)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target = reward + 0.95 * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SentimentAnalyzer:
    """Sentiment analysis from news and social media"""
    
    def __init__(self):
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'
        ]
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment from news"""
        try:
            # Get news from Yahoo Finance
            ticker = yf.Ticker(f"{symbol}.NS")
            news = ticker.news
            
            if not news:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            sentiments = []
            for article in news[:5]:  # Analyze top 5 articles
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title} {summary}"
                
                if text.strip():
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            avg_sentiment = np.mean(sentiments)
            confidence = 1 - np.std(sentiments) if len(sentiments) > 1 else 0.5
            
            return {
                'sentiment': float(avg_sentiment),  # -1 to 1
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def get_market_sentiment(self) -> Dict[str, float]:
        """Get overall market sentiment"""
        try:
            # Analyze Nifty 50 sentiment
            nifty = yf.Ticker("^NSEI")
            news = nifty.news
            
            if not news:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            sentiments = []
            for article in news[:10]:
                title = article.get('title', '')
                if title:
                    blob = TextBlob(title)
                    sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            avg_sentiment = np.mean(sentiments)
            confidence = 1 - np.std(sentiments) if len(sentiments) > 1 else 0.5
            
            return {
                'sentiment': float(avg_sentiment),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}

class EnsembleSignalFusion:
    """Ensemble methods for signal fusion"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.4,
            'lstm': 0.25,
            'ml_classifier': 0.25,
            'sentiment': 0.1
        }
        self.confidence_threshold = 0.7
    
    def fuse_signals(self, signals: Dict) -> Dict[str, float]:
        """Fuse multiple signals using ensemble methods"""
        try:
            # Extract individual signals
            technical_signal = signals.get('technical', {})
            lstm_prediction = signals.get('lstm_prediction')
            ml_signal = signals.get('ml_classifier', {})
            sentiment = signals.get('sentiment', {})
            
            # Convert LSTM prediction to signal
            lstm_signal = 0
            if lstm_prediction and 'current_price' in signals:
                current_price = signals['current_price']
                price_change = (lstm_prediction - current_price) / current_price
                if price_change > 0.02:
                    lstm_signal = 1  # BUY
                elif price_change < -0.02:
                    lstm_signal = -1  # SELL
            
            # Convert ML classifier signal (0=SELL, 1=HOLD, 2=BUY)
            ml_classifier_signal = 0
            if ml_signal.get('signal') == 2:
                ml_classifier_signal = 1
            elif ml_signal.get('signal') == 0:
                ml_classifier_signal = -1
            
            # Convert sentiment to signal
            sentiment_signal = 0
            if sentiment.get('sentiment', 0) > 0.1:
                sentiment_signal = 0.5
            elif sentiment.get('sentiment', 0) < -0.1:
                sentiment_signal = -0.5
            
            # Technical signal (already normalized -1 to 1)
            tech_signal = technical_signal.get('signal', 0)
            
            # Weighted ensemble
            ensemble_signal = (
                self.weights['technical'] * tech_signal +
                self.weights['lstm'] * lstm_signal +
                self.weights['ml_classifier'] * ml_classifier_signal +
                self.weights['sentiment'] * sentiment_signal
            )
            
            # Calculate confidence
            confidences = []
            if technical_signal.get('strength'):
                confidences.append(technical_signal['strength'])
            if ml_signal.get('confidence'):
                confidences.append(ml_signal['confidence'])
            if sentiment.get('confidence'):
                confidences.append(sentiment['confidence'])
            
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Adjust signal strength based on agreement
            signal_agreement = self._calculate_agreement([
                tech_signal, lstm_signal, ml_classifier_signal, sentiment_signal
            ])
            
            final_confidence = avg_confidence * signal_agreement
            
            return {
                'signal': float(np.clip(ensemble_signal, -1, 1)),
                'confidence': float(final_confidence),
                'components': {
                    'technical': tech_signal,
                    'lstm': lstm_signal,
                    'ml_classifier': ml_classifier_signal,
                    'sentiment': sentiment_signal
                },
                'agreement': float(signal_agreement)
            }
            
        except Exception as e:
            logger.error(f"Signal fusion failed: {e}")
            return {'signal': 0, 'confidence': 0, 'components': {}, 'agreement': 0}
    
    def _calculate_agreement(self, signals: List[float]) -> float:
        """Calculate agreement between signals"""
        if not signals:
            return 0.0
        
        # Count positive, negative, and neutral signals
        positive = sum(1 for s in signals if s > 0.1)
        negative = sum(1 for s in signals if s < -0.1)
        neutral = len(signals) - positive - negative
        
        # Agreement is higher when signals point in same direction
        max_agreement = max(positive, negative, neutral)
        return max_agreement / len(signals)
    
    def should_trade(self, fused_signal: Dict) -> bool:
        """Determine if we should trade based on fused signal"""
        signal_strength = abs(fused_signal.get('signal', 0))
        confidence = fused_signal.get('confidence', 0)
        agreement = fused_signal.get('agreement', 0)
        
        return (signal_strength > 0.6 and 
                confidence > self.confidence_threshold and 
                agreement > 0.6)

class MLTradingEngine:
    """Main ML engine that coordinates all ML components"""
    
    def __init__(self, capital: float, watchlist: List[str]):
        self.capital = capital
        self.watchlist = watchlist
        
        # Initialize ML components
        self.lstm_predictor = LSTMPricePredictor()
        self.signal_classifier = SignalClassifier()
        self.rl_position_sizer = RLPositionSizer(capital)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ensemble_fusion = EnsembleSignalFusion()
        
        self.models_trained = False
        
    def train_models(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Train all ML models"""
        logger.info("Training ML models...")
        
        success_count = 0
        
        for symbol, data in data_dict.items():
            if len(data) < 100:
                continue
                
            # Train LSTM
            if self.lstm_predictor.train(data):
                success_count += 1
                
            # Train signal classifier
            if self.signal_classifier.train(data):
                success_count += 1
        
        self.models_trained = success_count > 0
        logger.info(f"ML models training completed. Success rate: {success_count}/{len(data_dict)*2}")
        
        return self.models_trained
    
    def get_ml_signals(self, symbol: str, data: pd.DataFrame, current_price: float, 
                      portfolio_value: float, current_positions: int) -> Dict:
        """Get comprehensive ML signals for a symbol"""
        if not self.models_trained:
            return {'signal': 0, 'confidence': 0.0}
        
        signals = {'current_price': current_price}
        
        try:
            # LSTM price prediction
            lstm_prediction = self.lstm_predictor.predict_price(data)
            if lstm_prediction:
                signals['lstm_prediction'] = lstm_prediction
            
            # ML signal classification
            ml_signal = self.signal_classifier.predict_signal(data)
            signals['ml_classifier'] = ml_signal
            
            # Sentiment analysis
            sentiment = self.sentiment_analyzer.get_news_sentiment(symbol)
            signals['sentiment'] = sentiment
            
            # Fuse all signals
            fused_signal = self.ensemble_fusion.fuse_signals(signals)
            
            # Get RL position size
            if fused_signal.get('signal', 0) != 0:
                position_size = self.rl_position_sizer.get_position_size(
                    data, current_price, portfolio_value, current_positions
                )
                fused_signal['position_size'] = position_size
            
            return fused_signal
            
        except Exception as e:
            logger.error(f"ML signal generation failed for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def save_models(self, path: str):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        self.lstm_predictor.save_model(f"{path}/lstm")
        joblib.dump(self.signal_classifier, f"{path}/signal_classifier.pkl")
        joblib.dump(self.rl_position_sizer, f"{path}/rl_position_sizer.pkl")
    
    def load_models(self, path: str) -> bool:
        """Load pre-trained models"""
        try:
            if self.lstm_predictor.load_model(f"{path}/lstm"):
                self.signal_classifier = joblib.load(f"{path}/signal_classifier.pkl")
                self.rl_position_sizer = joblib.load(f"{path}/rl_position_sizer.pkl")
                self.models_trained = True
                logger.info("ML models loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
        
        return False