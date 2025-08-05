"""
Enhanced Trading Bot Startup Script
Integrates the trading bot with dashboard and monitoring
"""

import json
import threading
import time
import sys
import os
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trading_bot import EnhancedTradingBot
from trading_dashboard import TradingDashboard

class TradingBotManager:
    """Manager for the enhanced trading bot with dashboard"""
    
    def __init__(self, config_path: str = "trading_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.bot = None
        self.dashboard = None
        self.bot_thread = None
        self.dashboard_thread = None
        
        # Setup logging
        self.setup_logging()
        
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠ Configuration file {self.config_path} not found, using defaults")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"⚠ Error parsing configuration: {e}, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "capital": 50000,
            "broker_type": "discount",
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "max_portfolio_risk": 0.05,
                "max_positions": 3
            },
            "watchlist": ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'trading_bot.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def start_bot(self):
        """Start the trading bot in a separate thread"""
        try:
            print("🔧 Initializing trading bot...")
            self.bot = EnhancedTradingBot(capital=self.config['capital'])
            
            # Update bot configuration after initialization
            if 'watchlist' in self.config:
                self.bot.watchlist = self.config['watchlist']
                print(f"📊 Watchlist updated: {', '.join(self.config['watchlist'])}")
            
            if 'risk_management' in self.config:
                risk_config = self.config['risk_management']
                self.bot.risk_manager.max_risk_per_trade = risk_config.get('max_risk_per_trade', 0.02)
                self.bot.risk_manager.max_positions = risk_config.get('max_positions', 3)
                print(f"⚖️ Risk settings updated: {risk_config.get('max_risk_per_trade', 0.02)*100}% per trade, max {risk_config.get('max_positions', 3)} positions")
            
            # Start bot in separate thread
            print("🚀 Starting trading bot thread...")
            self.bot_thread = threading.Thread(target=self.bot.start_trading, daemon=True)
            self.bot_thread.start()
            
            print("✓ Trading bot started successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start trading bot: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread"""
        try:
            def run_dashboard():
                self.dashboard = TradingDashboard()
                self.dashboard.run()
            
            self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            self.dashboard_thread.start()
            
            print("✓ Dashboard started successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start dashboard: {e}")
            return False
    
    def collect_training_data(self):
        """Collect training data for ML models"""
        print("\n📊 Starting training data collection...")
        try:
            from collect_training_data import TrainingDataCollector
            
            collector = TrainingDataCollector()
            
            # Collect data
            data_dict = collector.collect_all_data(days=365)
            
            if data_dict:
                # Validate and save
                validated_data = collector.validate_data_quality(data_dict)
                if validated_data:
                    collector.save_data(validated_data)
                    collector.generate_summary_report(validated_data)
                    print("✅ Training data collection completed successfully!")
                else:
                    print("❌ No data passed validation")
            else:
                print("❌ No data collected")
                
        except ImportError:
            print("❌ Training data collector not found")
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot:
            self.bot.stop_trading()
            print("✓ Trading bot stopped")
    
    def get_status(self) -> dict:
        """Get current status of the bot"""
        if not self.bot:
            return {"status": "not_running"}
        
        summary = self.bot.get_portfolio_summary()
        return {
            "status": "running" if self.bot.is_running else "stopped",
            "capital": summary['capital'],
            "available_cash": summary['available_cash'],
            "total_pnl": summary['total_pnl'],
            "positions": summary['positions_count'],
            "trades": summary['trades_count']
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("    ENHANCED TRADING BOT STATUS")
        print("="*60)
        print(f"Status: {status.get('status', 'Unknown').upper()}")
        
        if status['status'] == 'running':
            print(f"Capital: Rs.{status['capital']:,.2f}")
            print(f"Available Cash: Rs.{status['available_cash']:,.2f}")
            print(f"Total P&L: Rs.{status['total_pnl']:,.2f}")
            print(f"Open Positions: {status['positions']}")
            print(f"Total Trades: {status['trades']}")
            
            # Live Prices Section
            print("\n" + "-" * 25 + " LIVE PRICES " + "-" * 25)
            watchlist = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE"]
            for symbol in watchlist:
                try:
                    price = self.bot.data_provider.get_live_price(symbol)
                    if price:
                        print(f"{symbol:10}: Rs.{price:8.2f}")
                    else:
                        print(f"{symbol:10}: Rs.{'N/A':>8}")
                except:
                    print(f"{symbol:10}: Rs.{'ERROR':>8}")
            
            # Active Positions Section
            if self.bot.positions:
                print("\n" + "-" * 23 + " ACTIVE POSITIONS " + "-" * 23)
                for symbol, position in self.bot.positions.items():
                    pnl = position.unrealized_pnl
                    pnl_pct = (pnl / (position.avg_price * position.quantity)) * 100
                    status_icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    print(f"{symbol:8} | {position.quantity:3} @ Rs.{position.avg_price:7.2f} | {status_icon} Rs.{pnl:+7.2f} ({pnl_pct:+5.1f}%)")
        
        print("="*60)
    
    def run_interactive(self):
        """Run in interactive mode"""
        print("\\n" + "="*60)
        print("    ENHANCED TRADING BOT - INTERACTIVE MODE")
        print("="*60)
        print(f"Capital: Rs.{self.config['capital']:,}")
        print(f"Watchlist: {', '.join(self.config['watchlist'])}")
        print("="*60)
        
        while True:
            print("\\nOptions:")
            print("1. Start Trading Bot")
            print("2. Start Dashboard")
            print("3. Collect Training Data")
            print("4. Show Status")
            print("5. Stop Bot")
            print("6. Exit")
            
            try:
                choice = input("\\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    if not self.bot or not self.bot.is_running:
                        self.start_bot()
                    else:
                        print("⚠ Bot is already running")
                
                elif choice == '2':
                    if not self.dashboard_thread or not self.dashboard_thread.is_alive():
                        self.start_dashboard()
                        time.sleep(2)  # Give dashboard time to start
                    else:
                        print("⚠ Dashboard is already running")
                
                elif choice == '3':
                    self.collect_training_data()
                
                elif choice == '4':
                    self.print_status()
                
                elif choice == '5':
                    self.stop_bot()
                
                elif choice == '6':
                    self.stop_bot()
                    print("\\n👋 Goodbye!")
                    break
                
                else:
                    print("⚠ Invalid choice. Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\\n\\n⚠ Interrupted by user")
                self.stop_bot()
                break
            except Exception as e:
                print(f"⚠ Error: {e}")

def main():
    """Main function"""
    print("\\n🚀 Starting Enhanced Trading Bot Manager...")
    
    # Check if required files exist
    required_files = [
        "enhanced_trading_bot.py",
        "trading_dashboard.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"⚠ Missing required files: {', '.join(missing_files)}")
        return
    
    # Create manager
    manager = TradingBotManager()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            # Auto mode: start bot and dashboard automatically
            print("🤖 Starting in auto mode...")
            manager.start_bot()
            time.sleep(2)
            manager.start_dashboard()
            
            try:
                while True:
                    time.sleep(10)
                    manager.print_status()
            except KeyboardInterrupt:
                print("\\n⚠ Stopping...")
                manager.stop_bot()
        
        elif sys.argv[1] == '--status':
            # Status mode: just show status and exit
            manager.print_status()
            return
    
    else:
        # Interactive mode
        manager.run_interactive()

if __name__ == "__main__":
    main()