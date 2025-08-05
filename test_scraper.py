"""
Web Scraper Tester - Test if scraper is getting real data
"""

import logging
from datetime import datetime
from real_nse_scraper import RealNSEScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraper():
    """Test the web scraper with all methods"""
    
    print("🧪 TESTING WEB SCRAPER FOR REAL DATA")
    print("=" * 60)
    
    scraper = RealNSEScraper()
    test_symbols = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
    
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Testing {len(test_symbols)} symbols")
    print(f"🏪 Market Status: {scraper.get_market_status()['status']}")
    print()
    
    # Test each method individually
    methods_to_test = [
        ("Yahoo Finance", scraper.get_yahoo_finance),
        ("NSE Quote API", scraper.get_nse_quote_api),
        ("NSE Live API", scraper.get_nse_live_api),
        ("Investing.com", scraper.get_investing_com)
    ]
    
    results = {}
    
    for method_name, method_func in methods_to_test:
        print(f"🔍 Testing {method_name}:")
        print("-" * 40)
        
        method_results = {}
        successful = 0
        
        for symbol in test_symbols:
            try:
                price = method_func(symbol)
                if price and price > 0:
                    method_results[symbol] = price
                    successful += 1
                    print(f"  ✅ {symbol:10}: Rs.{price:8.2f}")
                else:
                    print(f"  ❌ {symbol:10}: No data")
            except Exception as e:
                print(f"  ❌ {symbol:10}: Error - {e}")
        
        results[method_name] = method_results
        success_rate = (successful / len(test_symbols)) * 100
        print(f"  📈 Success Rate: {success_rate:.1f}% ({successful}/{len(test_symbols)})")
        print()
    
    # Test combined method
    print("🔄 Testing Combined Method (get_live_price):")
    print("-" * 50)
    
    combined_results = {}
    combined_successful = 0
    
    for symbol in test_symbols:
        try:
            price = scraper.get_live_price(symbol)
            if price and price > 0:
                combined_results[symbol] = price
                combined_successful += 1
                print(f"  ✅ {symbol:10}: Rs.{price:8.2f}")
            else:
                print(f"  ❌ {symbol:10}: No data")
        except Exception as e:
            print(f"  ❌ {symbol:10}: Error - {e}")
    
    combined_success_rate = (combined_successful / len(test_symbols)) * 100
    print(f"  📈 Combined Success Rate: {combined_success_rate:.1f}% ({combined_successful}/{len(test_symbols)})")
    print()
    
    # Summary
    print("📋 SUMMARY:")
    print("=" * 60)
    
    for method_name, method_results in results.items():
        success_count = len(method_results)
        success_rate = (success_count / len(test_symbols)) * 100
        status = "🟢 WORKING" if success_rate > 50 else "🟡 PARTIAL" if success_rate > 0 else "🔴 FAILED"
        print(f"{method_name:15}: {status} ({success_rate:.1f}%)")
    
    print(f"{'Combined Method':15}: {'🟢 WORKING' if combined_success_rate > 50 else '🟡 PARTIAL' if combined_success_rate > 0 else '🔴 FAILED'} ({combined_success_rate:.1f}%)")
    
    # Recommendations
    print()
    print("💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if combined_success_rate >= 80:
        print("✅ Scraper is working well! Bot can trade with real data.")
    elif combined_success_rate >= 50:
        print("⚠️  Scraper is partially working. Some trades may be missed.")
    else:
        print("❌ Scraper needs improvement. Bot may not find trading opportunities.")
    
    # Best performing method
    best_method = max(results.keys(), key=lambda x: len(results[x]))
    best_success_rate = (len(results[best_method]) / len(test_symbols)) * 100
    print(f"🏆 Best Method: {best_method} ({best_success_rate:.1f}% success)")
    
    return combined_success_rate >= 50

def test_historical_data():
    """Test historical data fetching"""
    print()
    print("📈 TESTING HISTORICAL DATA:")
    print("-" * 40)
    
    scraper = RealNSEScraper()
    test_symbol = "SBIN"
    
    try:
        data = scraper.get_historical_data(test_symbol, 30)
        if not data.empty:
            print(f"✅ Historical data for {test_symbol}: {len(data)} days")
            print(f"   Latest close: Rs.{data['close'].iloc[-1]:.2f}")
            print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")
        else:
            print(f"❌ No historical data for {test_symbol}")
    except Exception as e:
        print(f"❌ Historical data error: {e}")

if __name__ == "__main__":
    try:
        # Test live prices
        scraper_working = test_scraper()
        
        # Test historical data
        test_historical_data()
        
        print()
        print("🎯 FINAL RESULT:")
        print("=" * 30)
        if scraper_working:
            print("✅ SCRAPER IS READY FOR TRADING!")
            print("   Your bot can now get real market data.")
        else:
            print("❌ SCRAPER NEEDS FIXING!")
            print("   Bot may not work properly without real data.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()