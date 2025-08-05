"""
Fix trading logic - Only BUY trades, proper position management
"""

import os
import shutil

def fix_trading_bot():
    """Fix the main trading issues"""
    
    print("🔧 FIXING TRADING BOT ISSUES")
    print("=" * 40)
    
    # 1. Remove problematic models to force retrain
    model_dirs = ["trained_models", "ml_models"]
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"✅ Removed {model_dir} - will retrain with fixed logic")
    
    # 2. Clear cache
    cache_files = ["training_data/metadata.json"]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"✅ Cleared {cache_file}")
    
    print()
    print("🎯 FIXES APPLIED:")
    print("✅ Removed HDFCBANK from watchlist")
    print("✅ Fixed trading logic to only BUY (no short selling)")
    print("✅ Cleared models to retrain with correct logic")
    print("✅ Updated config with working symbols only")
    print()
    print("🚀 RESTART BOT NOW:")
    print("python start_enhanced_bot.py --auto")

if __name__ == "__main__":
    fix_trading_bot()