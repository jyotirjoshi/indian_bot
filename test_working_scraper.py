"""
Test the working scraper
"""

import logging
from datetime import datetime
from working_scraper import WorkingScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_working_scraper():
    """Test the simple working scraper"""
    
    print("🧪 TESTING WORKING SCRAPER")
    print("=" * 40)
    
    scraper = WorkingScraper()
    test_symbols = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
    
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏪 Market Status: {scraper.get_market_status()['status']}")
    print()
    
    successful = 0
    
    for symbol in test_symbols:
        try:
            price = scraper.get_live_price(symbol)
            if price and price > 0:
                successful += 1
                print(f"✅ {symbol:10}: Rs.{price:8.2f}")
            else:
                print(f"❌ {symbol:10}: No data")
        except Exception as e:
            print(f"❌ {symbol:10}: Error - {e}")
    
    success_rate = (successful / len(test_symbols)) * 100
    print(f"\n📈 Success Rate: {success_rate:.1f}% ({successful}/{len(test_symbols)})")
    
    if success_rate >= 50:
        print("✅ SCRAPER IS WORKING! Bot can trade with real data.")
        return True
    else:
        print("❌ SCRAPER FAILED! Need to fix data sources.")
        return False

if __name__ == "__main__":
    test_working_scraper()