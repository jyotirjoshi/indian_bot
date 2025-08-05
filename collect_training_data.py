"""
Training Data Collection Script
Fetches and prepares data for ML model training
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingDataCollector:
    """Dedicated class for collecting and preparing training data"""
    
    def __init__(self):
        self.watchlist = ["SBIN", "TCS", "HDFCBANK", "INFY", "ITC", "RELIANCE", "ICICIBANK", "AXISBANK"]
        self.data_dir = "training_data"
        
    def fetch_symbol_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch data for a single symbol with retry logic"""
        logger.info(f"📊 Fetching {days} days of data for {symbol}...")
        
        for attempt in range(3):
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period=f"{days}d", interval="1d")
                
                if not data.empty:
                    # Clean and format data
                    data.reset_index(inplace=True)
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Add derived features
                    data['returns'] = data['close'].pct_change()
                    data['volatility'] = data['returns'].rolling(20).std()
                    data['volume_ma'] = data['volume'].rolling(20).mean()
                    
                    logger.info(f"✅ {symbol}: {len(data)} days collected")
                    return data
                else:
                    logger.warning(f"⚠️ {symbol}: Empty data on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Attempt {attempt + 1} failed - {e}")
                
            time.sleep(2)  # Wait before retry
        
        logger.error(f"❌ {symbol}: Failed to fetch data after 3 attempts")
        return pd.DataFrame()
    
    def collect_all_data(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Collect data for all symbols"""
        logger.info(f"🚀 Starting data collection for {len(self.watchlist)} symbols...")
        logger.info(f"📅 Fetching {days} days of historical data")
        
        data_dict = {}
        successful = 0
        
        for i, symbol in enumerate(self.watchlist):
            logger.info(f"[{i+1}/{len(self.watchlist)}] Processing {symbol}...")
            
            data = self.fetch_symbol_data(symbol, days)
            
            if not data.empty and len(data) >= 100:
                data_dict[symbol] = data
                successful += 1
                
                # Log data quality
                logger.info(f"📈 {symbol} data quality:")
                logger.info(f"   • Date range: {data['date'].min().date()} to {data['date'].max().date()}")
                logger.info(f"   • Data points: {len(data)}")
                logger.info(f"   • Price range: Rs.{data['close'].min():.2f} - Rs.{data['close'].max():.2f}")
                logger.info(f"   • Avg volume: {data['volume'].mean():,.0f}")
            else:
                logger.error(f"❌ {symbol}: Insufficient data quality")
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"📊 Data collection summary:")
        logger.info(f"   • Successful: {successful}/{len(self.watchlist)} symbols")
        logger.info(f"   • Total data points: {sum(len(data) for data in data_dict.values()):,}")
        
        return data_dict
    
    def validate_data_quality(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean data quality"""
        logger.info("🔍 Validating data quality...")
        
        validated_data = {}
        
        for symbol, data in data_dict.items():
            issues = []
            
            # Check for missing values
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if missing_pct > 5:
                issues.append(f"High missing values: {missing_pct:.1f}%")
            
            # Check for price anomalies
            price_changes = data['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.2).sum()  # >20% daily changes
            if extreme_changes > len(data) * 0.05:  # >5% of days
                issues.append(f"Extreme price changes: {extreme_changes} days")
            
            # Check data recency
            last_date = pd.to_datetime(data['date'].max())
            days_old = (datetime.now() - last_date).days
            if days_old > 7:
                issues.append(f"Data is {days_old} days old")
            
            # Check minimum data requirement
            if len(data) < 100:
                issues.append(f"Insufficient data: {len(data)} days")
            
            if not issues:
                validated_data[symbol] = data
                logger.info(f"✅ {symbol}: Data quality passed")
            else:
                logger.warning(f"⚠️ {symbol}: Quality issues - {', '.join(issues)}")
        
        logger.info(f"🎯 Data validation complete: {len(validated_data)}/{len(data_dict)} symbols passed")
        return validated_data
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Save collected data to disk"""
        logger.info(f"💾 Saving data to {self.data_dir}...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Save individual CSV files
        for symbol, data in data_dict.items():
            file_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
            data.to_csv(file_path, index=False)
            logger.info(f"📁 Saved {symbol} data ({len(data)} rows)")
        
        # Create metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'symbols': list(data_dict.keys()),
            'data_points': {symbol: len(data) for symbol, data in data_dict.items()},
            'total_symbols': len(data_dict),
            'total_data_points': sum(len(data) for data in data_dict.values()),
            'date_ranges': {
                symbol: {
                    'start': data['date'].min().isoformat(),
                    'end': data['date'].max().isoformat()
                } for symbol, data in data_dict.items()
            }
        }
        
        # Save metadata
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Training data saved successfully!")
        logger.info(f"   • Location: {os.path.abspath(self.data_dir)}")
        logger.info(f"   • Files: {len(data_dict)} CSV files + metadata.json")
    
    def generate_summary_report(self, data_dict: Dict[str, pd.DataFrame]):
        """Generate a summary report of collected data"""
        logger.info("📋 Generating data summary report...")
        
        report = []
        report.append("=" * 60)
        report.append("         TRAINING DATA COLLECTION REPORT")
        report.append("=" * 60)
        report.append(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Symbols: {len(data_dict)}")
        report.append("")
        
        # Symbol-wise summary
        report.append("SYMBOL-WISE DATA SUMMARY:")
        report.append("-" * 40)
        
        total_points = 0
        for symbol, data in data_dict.items():
            start_date = data['date'].min().strftime('%Y-%m-%d')
            end_date = data['date'].max().strftime('%Y-%m-%d')
            data_points = len(data)
            total_points += data_points
            
            price_range = f"Rs.{data['close'].min():.2f} - Rs.{data['close'].max():.2f}"
            avg_volume = f"{data['volume'].mean():,.0f}"
            
            report.append(f"{symbol:10} | {data_points:3d} days | {start_date} to {end_date}")
            report.append(f"{'':10} | Price: {price_range} | Avg Vol: {avg_volume}")
            report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append("-" * 20)
        report.append(f"Total Data Points: {total_points:,}")
        report.append(f"Average per Symbol: {total_points // len(data_dict):,}")
        report.append(f"Data Quality: {len(data_dict)}/{len(self.watchlist)} symbols collected")
        report.append("")
        
        # ML readiness
        report.append("ML TRAINING READINESS:")
        report.append("-" * 22)
        lstm_ready = sum(1 for data in data_dict.values() if len(data) >= 60)
        classifier_ready = sum(1 for data in data_dict.values() if len(data) >= 100)
        
        report.append(f"LSTM Training: {lstm_ready}/{len(data_dict)} symbols ready (need 60+ days)")
        report.append(f"Classifier Training: {classifier_ready}/{len(data_dict)} symbols ready (need 100+ days)")
        report.append("")
        
        if lstm_ready >= 3 and classifier_ready >= 3:
            report.append("🎉 READY FOR ML TRAINING!")
        else:
            report.append("⚠️  Need more data for optimal ML training")
        
        report.append("=" * 60)
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report to file
        with open(os.path.join(self.data_dir, 'collection_report.txt'), 'w') as f:
            f.write(report_text)
        
        logger.info(f"📄 Report saved to {self.data_dir}/collection_report.txt")

def main():
    """Main data collection process"""
    print("🚀 Starting Training Data Collection...")
    print("=" * 50)
    
    collector = TrainingDataCollector()
    
    try:
        # Step 1: Collect data
        data_dict = collector.collect_all_data(days=365)
        
        if not data_dict:
            logger.error("❌ No data collected. Check internet connection and try again.")
            return
        
        # Step 2: Validate data quality
        validated_data = collector.validate_data_quality(data_dict)
        
        if not validated_data:
            logger.error("❌ No data passed validation. Check data sources.")
            return
        
        # Step 3: Save data
        collector.save_data(validated_data)
        
        # Step 4: Generate report
        collector.generate_summary_report(validated_data)
        
        print("\n🎉 Data collection completed successfully!")
        print("Your ML trading bot is now ready for training.")
        
    except KeyboardInterrupt:
        logger.info("❌ Data collection interrupted by user")
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")

if __name__ == "__main__":
    main()