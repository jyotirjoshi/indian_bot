"""
Aggressive Trading Strategies for Maximum Profit
Multiple strategies to maximize daily returns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AggressiveStrategies:
    """Advanced aggressive strategies for maximum profit"""
    
    def __init__(self):
        self.strategies = {
            'momentum_breakout': self.momentum_breakout,
            'volume_surge': self.volume_surge,
            'gap_trading': self.gap_trading,
            'scalping_signals': self.scalping_signals,
            'reversal_patterns': self.reversal_patterns
        }
    
    def momentum_breakout(self, data: pd.DataFrame, current_price: float) -> Dict:
        """High momentum breakout strategy"""
        if len(data) < 20:
            return {'signal': 0, 'strength': 0}
        
        # Calculate momentum indicators
        high_20 = data['high'].rolling(20).max().iloc[-1]
        low_20 = data['low'].rolling(20).min().iloc[-1]
        volume_avg = data['volume'].rolling(10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        signals = []
        
        # Breakout above 20-day high with volume
        if current_price > high_20 * 1.005 and current_volume > volume_avg * 1.5:
            signals.append(('MOMENTUM_BREAKOUT_BUY', 0.9))
        
        # Breakdown below 20-day low with volume
        elif current_price < low_20 * 0.995 and current_volume > volume_avg * 1.5:
            signals.append(('MOMENTUM_BREAKDOWN_SELL', 0.9))
        
        if not signals:
            return {'signal': 0, 'strength': 0}
        
        total_signal = sum(s[1] for s in signals if s[0].endswith('BUY')) - sum(s[1] for s in signals if s[0].endswith('SELL'))
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': abs(total_signal),
            'strategy': 'momentum_breakout'
        }
    
    def volume_surge(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Volume surge strategy for quick profits"""
        if len(data) < 10:
            return {'signal': 0, 'strength': 0}
        
        volume_avg = data['volume'].rolling(10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        signals = []
        
        # Massive volume surge with price increase
        if current_volume > volume_avg * 3 and price_change > 0.01:
            signals.append(('VOLUME_SURGE_BUY', 0.85))
        
        # Volume surge with price decrease
        elif current_volume > volume_avg * 3 and price_change < -0.01:
            signals.append(('VOLUME_SURGE_SELL', 0.85))
        
        if not signals:
            return {'signal': 0, 'strength': 0}
        
        total_signal = sum(s[1] for s in signals if s[0].endswith('BUY')) - sum(s[1] for s in signals if s[0].endswith('SELL'))
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': abs(total_signal),
            'strategy': 'volume_surge'
        }
    
    def gap_trading(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Gap trading for quick profits"""
        if len(data) < 2:
            return {'signal': 0, 'strength': 0}
        
        prev_close = data['close'].iloc[-2]
        gap_pct = (current_price - prev_close) / prev_close
        
        signals = []
        
        # Gap up - trade continuation
        if gap_pct > 0.02:
            signals.append(('GAP_UP_CONTINUATION', 0.8))
        
        # Gap down - trade reversal
        elif gap_pct < -0.02:
            signals.append(('GAP_DOWN_REVERSAL', 0.8))
        
        if not signals:
            return {'signal': 0, 'strength': 0}
        
        total_signal = sum(s[1] for s in signals)
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': abs(total_signal),
            'strategy': 'gap_trading'
        }
    
    def scalping_signals(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Quick scalping signals for frequent trades"""
        if len(data) < 5:
            return {'signal': 0, 'strength': 0}
        
        # Short-term moving averages
        ema_3 = data['close'].ewm(span=3).mean().iloc[-1]
        ema_8 = data['close'].ewm(span=8).mean().iloc[-1]
        
        signals = []
        
        # Quick EMA crossover
        if current_price > ema_3 > ema_8:
            signals.append(('SCALP_BUY', 0.7))
        elif current_price < ema_3 < ema_8:
            signals.append(('SCALP_SELL', 0.7))
        
        # Price momentum
        recent_prices = data['close'].tail(3).tolist()
        if len(recent_prices) >= 3:
            if all(recent_prices[i] < recent_prices[i+1] for i in range(len(recent_prices)-1)):
                signals.append(('MOMENTUM_BUY', 0.6))
            elif all(recent_prices[i] > recent_prices[i+1] for i in range(len(recent_prices)-1)):
                signals.append(('MOMENTUM_SELL', 0.6))
        
        if not signals:
            return {'signal': 0, 'strength': 0}
        
        total_signal = sum(s[1] for s in signals if s[0].endswith('BUY')) - sum(s[1] for s in signals if s[0].endswith('SELL'))
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': abs(total_signal),
            'strategy': 'scalping'
        }
    
    def reversal_patterns(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Reversal pattern detection for contrarian trades"""
        if len(data) < 10:
            return {'signal': 0, 'strength': 0}
        
        # Calculate RSI for oversold/overbought
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        signals = []
        
        # Extreme oversold reversal
        if current_rsi < 25:
            signals.append(('OVERSOLD_REVERSAL', 0.8))
        
        # Extreme overbought reversal
        elif current_rsi > 75:
            signals.append(('OVERBOUGHT_REVERSAL', -0.8))
        
        # Hammer pattern (simplified)
        if len(data) >= 2:
            prev_candle = data.iloc[-2]
            current_candle = data.iloc[-1]
            
            # Bullish hammer
            if (current_candle['close'] > current_candle['open'] and 
                (current_candle['open'] - current_candle['low']) > 2 * (current_candle['close'] - current_candle['open'])):
                signals.append(('HAMMER_REVERSAL', 0.7))
        
        if not signals:
            return {'signal': 0, 'strength': 0}
        
        total_signal = sum(signal[1] for signal in signals)
        
        return {
            'signal': np.clip(total_signal, -1, 1),
            'strength': abs(total_signal),
            'strategy': 'reversal_patterns'
        }
    
    def get_combined_signals(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Combine all aggressive strategies"""
        all_signals = []
        strategy_results = []
        
        for strategy_name, strategy_func in self.strategies.items():
            try:
                result = strategy_func(data, current_price)
                if result['strength'] > 0.5:  # Only consider strong signals
                    all_signals.append(result['signal'] * result['strength'])
                    strategy_results.append(f"{strategy_name}({result['strength']:.2f})")
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
        
        if not all_signals:
            return {'signal': 0, 'strength': 0, 'strategy': 'no_signal'}
        
        # Weighted average of all signals
        combined_signal = np.mean(all_signals)
        combined_strength = np.mean([abs(s) for s in all_signals])
        
        return {
            'signal': np.clip(combined_signal, -1, 1),
            'strength': combined_strength,
            'strategy': f"Combined: {', '.join(strategy_results)}"
        }

class ProfitMaximizer:
    """Profit maximization techniques"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.aggressive_strategies = AggressiveStrategies()
    
    def calculate_aggressive_position_size(self, price: float, signal_strength: float, volatility: float) -> int:
        """Calculate position size for maximum profit"""
        # Base allocation increases with signal strength
        base_allocation = self.capital * (0.15 + 0.25 * signal_strength)  # 15-40% allocation
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_factor = max(0.5, 1 - volatility)
        adjusted_allocation = base_allocation * volatility_factor
        
        quantity = int(adjusted_allocation / price)
        
        # Ensure minimum profitable trade
        if quantity * price < 8000:
            return 0
        
        return quantity
    
    def dynamic_stop_loss(self, entry_price: float, current_price: float, action: str, volatility: float) -> float:
        """Dynamic stop loss that adjusts with market conditions"""
        base_stop_pct = 0.015 + (volatility * 0.5)  # 1.5% + volatility adjustment
        
        if action == "BUY":
            # Trailing stop for profits
            if current_price > entry_price * 1.02:  # If 2% profit
                return current_price * (1 - base_stop_pct * 0.7)  # Tighter trailing stop
            else:
                return entry_price * (1 - base_stop_pct)
        else:
            if current_price < entry_price * 0.98:  # If 2% profit on short
                return current_price * (1 + base_stop_pct * 0.7)
            else:
                return entry_price * (1 + base_stop_pct)
    
    def profit_taking_levels(self, entry_price: float, action: str, signal_strength: float) -> List[Dict]:
        """Multiple profit taking levels for maximum extraction"""
        levels = []
        
        # Base target multiplier based on signal strength
        base_multiplier = 2.5 + (signal_strength * 2)  # 2.5x to 4.5x
        
        if action == "BUY":
            # Level 1: Quick profit (50% position)
            levels.append({
                'price': entry_price * (1 + 0.02),
                'quantity_pct': 0.5,
                'level': 'Quick_Profit'
            })
            
            # Level 2: Main target (30% position)
            levels.append({
                'price': entry_price * (1 + 0.04 * base_multiplier),
                'quantity_pct': 0.3,
                'level': 'Main_Target'
            })
            
            # Level 3: Moon shot (20% position)
            levels.append({
                'price': entry_price * (1 + 0.08 * base_multiplier),
                'quantity_pct': 0.2,
                'level': 'Moon_Shot'
            })
        
        return levels