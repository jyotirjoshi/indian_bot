"""
Debug the scanning process to find the exact issue
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_symbol_scanning():
    """Debug each step of symbol scanning"""
    watchlist = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
    
    for symbol in watchlist:
        print(f"\n{'='*50}")
        print(f"DEBUGGING {symbol}")
        print(f"{'='*50}")
        
        # Step 1: Get current price
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
                print(f"✅ Current Price: Rs.{current_price:.2f}")
            else:
                print(f"❌ No current price data")
                continue
        except Exception as e:
            print(f"❌ Price fetch error: {e}")
            continue
        
        # Step 2: Get historical data
        try:
            hist_data = ticker.history(period="90d", interval="1d")
            if not hist_data.empty:
                hist_data.reset_index(inplace=True)
                hist_data.columns = [col.lower() for col in hist_data.columns]
                print(f"✅ Historical Data: {len(hist_data)} days")
            else:
                print(f"❌ No historical data")
                continue
        except Exception as e:
            print(f"❌ Historical data error: {e}")
            continue
        
        # Step 3: Calculate RSI
        try:
            delta = hist_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            print(f"✅ RSI: {current_rsi:.2f}")
            
            # Check RSI signals
            if current_rsi < 30:
                print(f"🟢 RSI OVERSOLD SIGNAL!")
            elif current_rsi > 70:
                print(f"🔴 RSI OVERBOUGHT SIGNAL!")
            else:
                print(f"⚪ RSI Neutral")
                
        except Exception as e:
            print(f"❌ RSI calculation error: {e}")
            continue
        
        # Step 4: Calculate MACD
        try:
            exp1 = hist_data['close'].ewm(span=12).mean()
            exp2 = hist_data['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                print(f"🟢 MACD BULLISH CROSSOVER!")
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                print(f"🔴 MACD BEARISH CROSSOVER!")
            else:
                print(f"⚪ MACD Neutral")
                
        except Exception as e:
            print(f"❌ MACD calculation error: {e}")
        
        # Step 5: Calculate Bollinger Bands
        try:
            sma = hist_data['close'].rolling(window=20).mean()
            std = hist_data['close'].rolling(window=20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            if current_price <= lower.iloc[-1]:
                print(f"🟢 BOLLINGER OVERSOLD!")
            elif current_price >= upper.iloc[-1]:
                print(f"🔴 BOLLINGER OVERBOUGHT!")
            else:
                print(f"⚪ Bollinger Neutral")
                
        except Exception as e:
            print(f"❌ Bollinger calculation error: {e}")
        
        # Step 6: Position sizing test
        try:
            capital = 50000
            risk_per_trade = 0.025
            risk_amount = capital * risk_per_trade
            stop_distance = current_price * 0.02
            stop_loss = current_price - stop_distance
            risk_per_share = abs(current_price - stop_loss)
            quantity = int(risk_amount / risk_per_share)
            max_allocation = capital * 0.35
            max_quantity = int(max_allocation / current_price)
            final_quantity = min(quantity, max_quantity)
            trade_value = final_quantity * current_price
            
            print(f"📊 POSITION SIZING:")
            print(f"   Risk Amount: Rs.{risk_amount:.2f}")
            print(f"   Stop Loss: Rs.{stop_loss:.2f}")
            print(f"   Risk per Share: Rs.{risk_per_share:.2f}")
            print(f"   Calculated Quantity: {quantity}")
            print(f"   Max Quantity: {max_quantity}")
            print(f"   Final Quantity: {final_quantity}")
            print(f"   Trade Value: Rs.{trade_value:.2f}")
            
            if trade_value < 8000:
                print(f"❌ Trade value too small (< Rs.8000)")
            else:
                print(f"✅ Trade size acceptable")
                
        except Exception as e:
            print(f"❌ Position sizing error: {e}")
        
        time.sleep(2)  # Rate limiting

if __name__ == "__main__":
    debug_symbol_scanning()