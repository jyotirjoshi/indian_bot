"""
Trading Bot Dashboard for Real-time Monitoring
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import time
import threading
from typing import Dict, List
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class TradingDashboard:
    """Real-time trading dashboard"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.root = tk.Tk()
        self.root.title("Enhanced Trading Bot Dashboard")
        self.root.geometry("1200x800")
        
        # Variables for real-time updates
        self.update_interval = 5000  # 5 seconds
        self.is_running = False
        
        self.setup_ui()
        self.start_updates()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Portfolio Summary Frame
        portfolio_frame = ttk.LabelFrame(main_frame, text="Portfolio Summary", padding="10")
        portfolio_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        # Portfolio labels
        self.capital_label = ttk.Label(portfolio_frame, text="Capital: Rs.0", font=("Arial", 12, "bold"))
        self.capital_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.cash_label = ttk.Label(portfolio_frame, text="Available Cash: Rs.0")
        self.cash_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.invested_label = ttk.Label(portfolio_frame, text="Invested: Rs.0")
        self.invested_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        self.pnl_label = ttk.Label(portfolio_frame, text="Total P&L: Rs.0", font=("Arial", 12, "bold"))
        self.pnl_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        
        self.daily_pnl_label = ttk.Label(portfolio_frame, text="Daily P&L: Rs.0")
        self.daily_pnl_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        self.positions_label = ttk.Label(portfolio_frame, text="Positions: 0")
        self.positions_label.grid(row=1, column=2, sticky=tk.W, padx=(0, 20))
        
        # Current Positions Frame
        positions_frame = ttk.LabelFrame(main_frame, text="Current Positions", padding="10")
        positions_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        positions_frame.columnconfigure(0, weight=1)
        positions_frame.rowconfigure(0, weight=1)
        
        # Positions treeview
        self.positions_tree = ttk.Treeview(positions_frame, columns=("Symbol", "Qty", "Avg Price", "Current Price", "P&L", "P&L%"), show="headings", height=8)
        
        # Define headings
        self.positions_tree.heading("Symbol", text="Symbol")
        self.positions_tree.heading("Qty", text="Quantity")
        self.positions_tree.heading("Avg Price", text="Avg Price")
        self.positions_tree.heading("Current Price", text="Current Price")
        self.positions_tree.heading("P&L", text="P&L (Rs.)")
        self.positions_tree.heading("P&L%", text="P&L %")
        
        # Configure column widths
        self.positions_tree.column("Symbol", width=80)
        self.positions_tree.column("Qty", width=80)
        self.positions_tree.column("Avg Price", width=100)
        self.positions_tree.column("Current Price", width=100)
        self.positions_tree.column("P&L", width=100)
        self.positions_tree.column("P&L%", width=80)
        
        self.positions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for positions
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        positions_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        # Recent Trades Frame
        trades_frame = ttk.LabelFrame(main_frame, text="Recent Trades", padding="10")
        trades_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        trades_frame.columnconfigure(0, weight=1)
        trades_frame.rowconfigure(0, weight=1)
        
        # Trades treeview
        self.trades_tree = ttk.Treeview(trades_frame, columns=("Time", "Symbol", "Action", "Qty", "Price", "P&L"), show="headings", height=8)
        
        # Define headings
        self.trades_tree.heading("Time", text="Time")
        self.trades_tree.heading("Symbol", text="Symbol")
        self.trades_tree.heading("Action", text="Action")
        self.trades_tree.heading("Qty", text="Qty")
        self.trades_tree.heading("Price", text="Price")
        self.trades_tree.heading("P&L", text="P&L")
        
        # Configure column widths
        self.trades_tree.column("Time", width=80)
        self.trades_tree.column("Symbol", width=80)
        self.trades_tree.column("Action", width=60)
        self.trades_tree.column("Qty", width=60)
        self.trades_tree.column("Price", width=80)
        self.trades_tree.column("P&L", width=80)
        
        self.trades_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for trades
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        trades_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.trades_tree.configure(yscrollcommand=trades_scrollbar.set)
        
        # Statistics Frame
        stats_frame = ttk.LabelFrame(main_frame, text="Trading Statistics", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.total_trades_label = ttk.Label(stats_frame, text="Total Trades: 0")
        self.total_trades_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.win_rate_label = ttk.Label(stats_frame, text="Win Rate: 0%")
        self.win_rate_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.avg_profit_label = ttk.Label(stats_frame, text="Avg Profit: Rs.0")
        self.avg_profit_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        self.avg_loss_label = ttk.Label(stats_frame, text="Avg Loss: Rs.0")
        self.avg_loss_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.refresh_button = ttk.Button(control_frame, text="Refresh", command=self.update_dashboard)
        self.refresh_button.grid(row=0, column=0, padx=(0, 10))
        
        self.export_button = ttk.Button(control_frame, text="Export Trades", command=self.export_trades)
        self.export_button.grid(row=0, column=1, padx=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def get_portfolio_data(self) -> Dict:
        """Get portfolio data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent trades
            trades_df = pd.read_sql_query("""
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 20
            """, conn)
            
            # Get current positions (this would need to be updated by the bot)
            try:
                positions_df = pd.read_sql_query("SELECT * FROM positions", conn)
            except:
                positions_df = pd.DataFrame()
            
            conn.close()
            
            # Calculate portfolio metrics
            total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
            daily_trades = trades_df[trades_df['timestamp'].str.contains(datetime.now().strftime('%Y-%m-%d'))] if not trades_df.empty else pd.DataFrame()
            daily_pnl = daily_trades['pnl'].sum() if not daily_trades.empty else 0
            
            return {
                'trades': trades_df,
                'positions': positions_df,
                'total_pnl': total_pnl,
                'daily_pnl': daily_pnl,
                'total_trades': len(trades_df)
            }
            
        except Exception as e:
            print(f"Error getting portfolio data: {e}")
            return {
                'trades': pd.DataFrame(),
                'positions': pd.DataFrame(),
                'total_pnl': 0,
                'daily_pnl': 0,
                'total_trades': 0
            }
    
    def update_dashboard(self):
        """Update dashboard with latest data"""
        try:
            self.status_var.set("Updating...")
            data = self.get_portfolio_data()
            
            # Update portfolio summary
            capital = 50000  # From config
            invested = 0
            current_value = 0
            
            if not data['positions'].empty:
                invested = (data['positions']['quantity'] * data['positions']['avg_price']).sum()
                current_value = (data['positions']['quantity'] * data['positions']['current_price']).sum()
            
            available_cash = capital - invested + data['total_pnl']
            
            self.capital_label.config(text=f"Capital: Rs.{capital:,.2f}")
            self.cash_label.config(text=f"Available Cash: Rs.{available_cash:,.2f}")
            self.invested_label.config(text=f"Invested: Rs.{invested:,.2f}")
            self.pnl_label.config(text=f"Total P&L: Rs.{data['total_pnl']:,.2f}")
            self.daily_pnl_label.config(text=f"Daily P&L: Rs.{data['daily_pnl']:,.2f}")
            self.positions_label.config(text=f"Positions: {len(data['positions'])}")
            
            # Color code P&L
            pnl_color = "green" if data['total_pnl'] >= 0 else "red"
            daily_pnl_color = "green" if data['daily_pnl'] >= 0 else "red"
            self.pnl_label.config(foreground=pnl_color)
            self.daily_pnl_label.config(foreground=daily_pnl_color)
            
            # Update positions table
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            if not data['positions'].empty:
                for _, position in data['positions'].iterrows():
                    pnl = (position['current_price'] - position['avg_price']) * position['quantity']
                    pnl_pct = (pnl / (position['avg_price'] * position['quantity'])) * 100
                    
                    self.positions_tree.insert("", "end", values=(
                        position['symbol'],
                        int(position['quantity']),
                        f"Rs.{position['avg_price']:.2f}",
                        f"Rs.{position['current_price']:.2f}",
                        f"Rs.{pnl:.2f}",
                        f"{pnl_pct:.1f}%"
                    ))
            
            # Update trades table
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)
            
            if not data['trades'].empty:
                for _, trade in data['trades'].head(10).iterrows():
                    timestamp = pd.to_datetime(trade['timestamp']).strftime('%H:%M')
                    pnl_str = f"Rs.{trade['pnl']:.2f}" if trade['pnl'] != 0 else "-"
                    
                    self.trades_tree.insert("", "end", values=(
                        timestamp,
                        trade['symbol'],
                        trade['action'],
                        int(trade['quantity']),
                        f"Rs.{trade['price']:.2f}",
                        pnl_str
                    ))
            
            # Update statistics
            if not data['trades'].empty:
                completed_trades = data['trades'][data['trades']['pnl'] != 0]
                if not completed_trades.empty:
                    winning_trades = completed_trades[completed_trades['pnl'] > 0]
                    losing_trades = completed_trades[completed_trades['pnl'] < 0]
                    
                    win_rate = (len(winning_trades) / len(completed_trades)) * 100
                    avg_profit = winning_trades['pnl'].mean() if not winning_trades.empty else 0
                    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
                    
                    self.total_trades_label.config(text=f"Total Trades: {len(completed_trades)}")
                    self.win_rate_label.config(text=f"Win Rate: {win_rate:.1f}%")
                    self.avg_profit_label.config(text=f"Avg Profit: Rs.{avg_profit:.2f}")
                    self.avg_loss_label.config(text=f"Avg Loss: Rs.{avg_loss:.2f}")
            
            self.status_var.set(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Dashboard update error: {e}")
    
    def export_trades(self):
        """Export trades to CSV"""
        try:
            data = self.get_portfolio_data()
            if not data['trades'].empty:
                filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data['trades'].to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Trades exported to {filename}")
            else:
                messagebox.showwarning("No Data", "No trades to export")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export trades: {str(e)}")
    
    def start_updates(self):
        """Start automatic updates"""
        self.is_running = True
        self.update_dashboard()
        self.schedule_update()
    
    def schedule_update(self):
        """Schedule next update"""
        if self.is_running:
            self.root.after(self.update_interval, self.auto_update)
    
    def auto_update(self):
        """Automatic update callback"""
        self.update_dashboard()
        self.schedule_update()
    
    def run(self):
        """Run the dashboard"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()