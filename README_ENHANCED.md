# Enhanced Trading Bot - Real Data & Realistic Environment

## 🎯 Overview
This enhanced trading bot is specifically designed for your ₹50,000 capital with real market data, proper commission calculations, and realistic trading environment.

## ✨ Key Features

### 🔄 Real Data Sources
- **NSE API**: Direct NSE price feeds
- **Yahoo Finance**: Backup data source
- **Web Scraping**: Fallback for real-time prices
- **No Fake Data**: Completely removed synthetic data generation

### 💰 Realistic Trading Environment
- **Accurate Commissions**: Zerodha-like discount broker charges
  - Brokerage: 0.03% or ₹20 max per trade
  - STT: 0.1% on sell side
  - Transaction charges: 0.345%
  - GST: 18% on brokerage + transaction charges
  - Stamp duty: 0.003% on buy side
  - SEBI charges: ₹1 per crore

### 🛡️ Risk Management (Optimized for ₹50k)
- **Maximum 3 positions** simultaneously
- **2% risk per trade** (₹1,000 max risk)
- **30% max allocation per trade** (₹15,000 max)
- **5% daily loss limit** (₹2,500 max daily loss)
- **Minimum trade size**: ₹1,000

### 📊 Technical Analysis
- **RSI**: Oversold/Overbought signals
- **MACD**: Trend reversal detection
- **Bollinger Bands**: Volatility-based signals
- **Multi-timeframe confirmation**

### 📈 Real-time Dashboard
- Live portfolio tracking
- Position monitoring
- Trade history
- Performance statistics
- P&L visualization

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 2. Run the Bot (Windows)
```bash
start_bot.bat
```

### 3. Or Run Manually
```bash
python start_enhanced_bot.py
```

## 📁 File Structure
```
models/
├── enhanced_trading_bot.py      # Main trading bot
├── trading_dashboard.py         # Real-time dashboard
├── start_enhanced_bot.py        # Startup manager
├── trading_config.json          # Configuration file
├── requirements_enhanced.txt    # Dependencies
├── start_bot.bat               # Windows launcher
└── README_ENHANCED.md          # This file
```

## ⚙️ Configuration

### Capital & Risk Settings
```json
{
    "capital": 50000,
    "risk_management": {
        "max_risk_per_trade": 0.02,    // 2% per trade
        "max_positions": 3,            // Max 3 positions
        "max_allocation_per_trade": 0.3 // 30% max per trade
    }
}
```

### Watchlist (Liquid Stocks)
```json
{
    "watchlist": [
        "SBIN",      // State Bank of India
        "TCS",       // Tata Consultancy Services
        "HDFCBANK",  // HDFC Bank
        "INFY",      // Infosys
        "ITC",       // ITC Limited
        "RELIANCE"   // Reliance Industries
    ]
}
```

## 📊 Trading Strategy

### Signal Generation
1. **RSI < 30**: Oversold (Buy signal)
2. **RSI > 70**: Overbought (Sell signal)
3. **MACD Bullish Crossover**: Buy signal
4. **MACD Bearish Crossover**: Sell signal
5. **Price at Bollinger Lower Band**: Buy signal
6. **Price at Bollinger Upper Band**: Sell signal

### Position Management
- **Stop Loss**: 1.5% minimum or based on volatility
- **Target**: 2:1 risk-reward ratio
- **Exit**: Stop loss, target, or market close (3:25 PM)

## 💡 Usage Examples

### Interactive Mode
```bash
python start_enhanced_bot.py
```
Choose options:
1. Start Trading Bot
2. Start Dashboard
3. Show Status
4. Stop Bot
5. Exit

### Auto Mode
```bash
python start_enhanced_bot.py --auto
```
Automatically starts bot and dashboard.

### Status Check
```bash
python start_enhanced_bot.py --status
```
Shows current portfolio status.

## 📈 Expected Performance

### Conservative Estimates (₹50,000 capital)
- **Daily Target**: ₹500-1,000 (1-2%)
- **Monthly Target**: ₹10,000-15,000 (20-30%)
- **Maximum Daily Loss**: ₹2,500 (5%)
- **Win Rate Target**: 60-70%

### Risk Metrics
- **Maximum Drawdown**: 10%
- **Sharpe Ratio Target**: > 1.5
- **Maximum Positions**: 3
- **Average Trade Size**: ₹8,000-12,000

## 🔧 Customization

### Adding New Stocks
Edit `trading_config.json`:
```json
{
    "watchlist": [
        "SBIN", "TCS", "HDFCBANK", 
        "YOURNEWSTOCK"  // Add here
    ]
}
```

### Adjusting Risk
```json
{
    "risk_management": {
        "max_risk_per_trade": 0.015,  // Reduce to 1.5%
        "max_positions": 2            // Reduce to 2 positions
    }
}
```

### Changing Broker Charges
```json
{
    "charges": {
        "brokerage_rate": 0.0005,     // 0.05% for full-service
        "max_brokerage": 100.0        // ₹100 max
    }
}
```

## 📊 Dashboard Features

### Portfolio Summary
- Total capital and available cash
- Current positions and P&L
- Daily and total performance

### Position Tracking
- Real-time position updates
- Unrealized P&L calculation
- Stop loss and target monitoring

### Trade History
- Complete trade log
- Commission breakdown
- Performance analysis

### Statistics
- Win rate calculation
- Average profit/loss
- Risk metrics

## ⚠️ Important Notes

### Market Hours
- **Trading**: 9:15 AM - 3:30 PM (IST)
- **Position Exit**: All positions closed by 3:25 PM
- **Weekend**: Bot inactive on Saturday/Sunday

### Data Sources
- Primary: NSE API (real-time)
- Backup: Yahoo Finance
- Cache: 1-minute duration
- Fallback: Last known price

### Risk Warnings
- **Past performance doesn't guarantee future results**
- **Start with paper trading to test strategies**
- **Monitor positions actively during market hours**
- **Keep emergency stop-loss ready**

## 🐛 Troubleshooting

### Common Issues

#### 1. "No price data available"
- Check internet connection
- Verify stock symbols are correct
- NSE API might be down, will use Yahoo Finance

#### 2. "Insufficient cash for trade"
- Check available cash in dashboard
- Reduce position size in config
- Close existing positions

#### 3. "Database locked"
- Close dashboard and restart
- Delete `trading_bot.db` if corrupted

#### 4. "Import errors"
- Run: `pip install -r requirements_enhanced.txt`
- Check Python version (3.8+ required)

### Support
- Check logs in `trading_bot.log`
- Review configuration in `trading_config.json`
- Monitor dashboard for real-time status

## 📞 Next Steps

### Phase 1: Testing (Week 1-2)
1. Run in paper trading mode
2. Monitor signal quality
3. Adjust parameters based on performance

### Phase 2: Live Trading (Week 3+)
1. Start with smaller capital (₹10,000)
2. Gradually increase to full ₹50,000
3. Monitor and optimize continuously

### Phase 3: Enhancement
1. Add more technical indicators
2. Implement machine learning models
3. Add sector rotation strategies

---

## 🎯 Ready to Start?

1. **Install**: Run `pip install -r requirements_enhanced.txt`
2. **Configure**: Edit `trading_config.json` if needed
3. **Launch**: Double-click `start_bot.bat` or run `python start_enhanced_bot.py`
4. **Monitor**: Use the dashboard to track performance
5. **Optimize**: Adjust settings based on results

**Good luck with your trading! 🚀📈**