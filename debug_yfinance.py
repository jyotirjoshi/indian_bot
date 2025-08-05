"""
Debug yfinance to see what's happening
"""

import yfinance as yf
import requests
import time

def debug_yfinance():
    """Debug yfinance step by step"""
    
    print("🔍 DEBUGGING YFINANCE")
    print("=" * 30)
    
    symbol = "SBIN"
    
    print(f"Testing symbol: {symbol}")
    print()
    
    # Test 1: Basic ticker creation
    print("1. Creating ticker...")
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        print("✅ Ticker created successfully")
    except Exception as e:
        print(f"❌ Ticker creation failed: {e}")
        return
    
    # Test 2: Get info
    print("\n2. Getting info...")
    try:
        info = ticker.info
        print(f"✅ Info retrieved: {len(info)} fields")
        
        # Check for price fields
        price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
        for field in price_fields:
            if field in info:
                print(f"   {field}: {info[field]}")
            else:
                print(f"   {field}: Not found")
                
    except Exception as e:
        print(f"❌ Info failed: {e}")
    
    # Test 3: Get history
    print("\n3. Getting history...")
    try:
        data = ticker.history(period="5d", interval="1d")
        if not data.empty:
            print(f"✅ History retrieved: {len(data)} rows")
            print(f"   Latest close: {data['Close'].iloc[-1]:.2f}")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        else:
            print("❌ History is empty")
    except Exception as e:
        print(f"❌ History failed: {e}")
    
    # Test 4: Different periods
    print("\n4. Testing different periods...")
    periods = ["1d", "2d", "5d"]
    for period in periods:
        try:
            data = ticker.history(period=period, interval="1d")
            if not data.empty:
                print(f"✅ {period}: {len(data)} rows, latest: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {period}: Empty")
        except Exception as e:
            print(f"❌ {period}: {e}")
    
    # Test 5: Network connectivity
    print("\n5. Testing network...")
    try:
        response = requests.get("https://finance.yahoo.com", timeout=10)
        print(f"✅ Yahoo Finance accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Network issue: {e}")

if __name__ == "__main__":
    debug_yfinance()