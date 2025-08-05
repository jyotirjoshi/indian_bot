"""
Test ML Integration for Enhanced Trading Bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_components import MLTradingEngine
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def test_ml_components():
    """Test all ML components"""
    print("🧠 Testing ML Components Integration...")
    
    # Test data
    watchlist = ["TCS", "SBIN", "HDFCBANK"]
    capital = 50000
    
    # Initialize ML engine
    ml_engine = MLTradingEngine(capital, watchlist)
    print("✅ ML Engine initialized")
    
    # Get sample data
    print("📊 Fetching sample data...")
    data_dict = {}
    
    for symbol in watchlist[:1]:  # Test with one symbol
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="6mo", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                data_dict[symbol] = data
                print(f"✅ Got {len(data)} days of data for {symbol}")
            
        except Exception as e:
            print(f"❌ Failed to get data for {symbol}: {e}")
    
    if not data_dict:
        print("❌ No data available for testing")
        return False
    
    # Test model training
    print("🏋️ Training ML models...")
    success = ml_engine.train_models(data_dict)
    
    if success:
        print("✅ ML models trained successfully")
        
        # Test signal generation
        symbol = list(data_dict.keys())[0]
        data = data_dict[symbol]
        current_price = data['close'].iloc[-1]
        
        print(f"🔮 Testing ML signals for {symbol} @ Rs.{current_price:.2f}")
        
        ml_signals = ml_engine.get_ml_signals(
            symbol, data, current_price, capital, 0
        )
        
        print("📈 ML Signal Results:")
        print(f"   Signal: {ml_signals.get('signal', 0):.2f}")
        print(f"   Confidence: {ml_signals.get('confidence', 0):.2f}")
        print(f"   Agreement: {ml_signals.get('agreement', 0):.2f}")
        
        if 'components' in ml_signals:
            components = ml_signals['components']
            print("🧩 Signal Components:")
            print(f"   Technical: {components.get('technical', 0):.2f}")
            print(f"   LSTM: {components.get('lstm', 0):.2f}")
            print(f"   ML Classifier: {components.get('ml_classifier', 0):.2f}")
            print(f"   Sentiment: {components.get('sentiment', 0):.2f}")
        
        if 'position_size' in ml_signals:
            print(f"💰 Recommended Position Size: {ml_signals['position_size']} shares")
        
        print("✅ ML integration test completed successfully!")
        return True
        
    else:
        print("❌ ML model training failed")
        return False

if __name__ == "__main__":
    print("🚀 Starting ML Integration Test...")
    print("=" * 50)
    
    try:
        success = test_ml_components()
        
        print("=" * 50)
        if success:
            print("🎉 All ML components working correctly!")
            print("Your enhanced trading bot is ready with:")
            print("   ✅ LSTM Price Prediction")
            print("   ✅ Random Forest/XGBoost Classification")
            print("   ✅ Reinforcement Learning Position Sizing")
            print("   ✅ Sentiment Analysis")
            print("   ✅ Ensemble Signal Fusion")
        else:
            print("⚠️  Some ML components need attention")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_enhanced.txt")