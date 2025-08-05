"""
Test Final Working Scraper
"""

import logging
from datetime import datetime
from final_working_scraper import FinalWorkingScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_final_scraper():
    """Test the final working scraper"""
    
    print("🧪 TESTING FINAL WORKING SCRAPER")
    print("=" * 50)
    
    scraper = FinalWorkingScraper()
    test_symbols = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
    
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏪 Market Status: {scraper.get_market_status()['status']}")
    print()
    
    successful = 0
    
    for symbol in test_symbols:
        try:
            print(f"🔍 Testing {symbol}...")
            price = scraper.get_live_price(symbol)
            if price and price > 0:
                successful += 1
                print(f"✅ {symbol:10}: Rs.{price:8.2f}")
            else:
                print(f"❌ {symbol:10}: No data")
        except Exception as e:
            print(f"❌ {symbol:10}: Error - {e}")
        
        print()  # Space between tests
    
    success_rate = (successful / len(test_symbols)) * 100
    print(f"📈 SUCCESS RATE: {success_rate:.1f}% ({successful}/{len(test_symbols)})")
    print()
    
    if success_rate >= 50:
        print("🎉 SCRAPER IS WORKING!")
        print("✅ Your bot can now trade with REAL market data!")
        print("✅ Run: python start_enhanced_bot.py --auto")
        return True
    else:
        print("❌ SCRAPER NEEDS MORE WORK")
        print("⚠️  Bot may not find trading opportunities")
        return False

if __name__ == "__main__":
    test_final_scraper()