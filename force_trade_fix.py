"""
Quick fix to force trading by making bot ultra-aggressive
Run this to patch the enhanced_trading_bot.py file
"""

import re

def fix_trading_thresholds():
    """Make bot ultra-aggressive to force trades"""
    
    # Read the current file
    with open('enhanced_trading_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the signal threshold line
    old_line = "                if abs(final_signal) > 0.4 and signal_strength > 0.5:"
    new_line = "                if abs(final_signal) > 0.1 and signal_strength > 0.2:"
    
    content = content.replace(old_line, new_line)
    
    # Also reduce minimum trade size
    old_trade_size = "        if quantity * price < 8000:  # Minimum Rs.8000 trade for meaningful profit"
    new_trade_size = "        if quantity * price < 3000:  # Minimum Rs.3000 trade for more opportunities"
    
    content = content.replace(old_trade_size, new_trade_size)
    
    # Make position sizing more aggressive
    old_allocation = "        base_allocation = self.capital * (0.15 + 0.25 * signal_strength)  # 15-40% allocation"
    new_allocation = "        base_allocation = self.capital * (0.25 + 0.35 * signal_strength)  # 25-60% allocation"
    
    if old_allocation in content:
        content = content.replace(old_allocation, new_allocation)
    
    # Write back the modified content
    with open('enhanced_trading_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Bot made ultra-aggressive!")
    print("🔥 New settings:")
    print("   - Signal threshold: 10% (was 40%)")
    print("   - Signal strength: 20% (was 50%)")
    print("   - Min trade size: Rs.3,000 (was Rs.8,000)")
    print("   - Position allocation: 25-60% (was 15-40%)")
    print("\n🚀 Restart your bot to see aggressive trading!")

if __name__ == "__main__":
    fix_trading_thresholds()
