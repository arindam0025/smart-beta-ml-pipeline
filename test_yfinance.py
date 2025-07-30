"""
Direct test of yfinance functionality
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test basic yfinance functionality
print("Testing yfinance directly...")

# Test single ticker
try:
    print("\n1. Testing single ticker (AAPL)...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False)
    print(f"AAPL data shape: {aapl.shape}")
    print(f"Columns: {list(aapl.columns)}")
    print(f"Sample data:\n{aapl.head()}")
    
    if 'Adj Close' in aapl.columns:
        adj_close = aapl['Adj Close']
        print(f"Adj Close shape: {adj_close.shape}")
        print(f"Recent prices: {adj_close.tail()}")
    
except Exception as e:
    print(f"Error with single ticker: {e}")

# Test multiple tickers
try:
    print("\n2. Testing multiple tickers...")
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
    print(f"Multi-ticker data shape: {data.shape}")
    print(f"Column structure: {data.columns}")
    
    if hasattr(data.columns, 'levels'):
        print(f"Level 0: {data.columns.levels[0]}")
        print(f"Level 1: {data.columns.levels[1]}")
        
        # Try to extract AAPL data
        if 'AAPL' in data.columns.levels[0]:
            aapl_data = data['AAPL']
            print(f"AAPL from multi: {aapl_data.shape}")
            if 'Adj Close' in aapl_data.columns:
                print(f"AAPL Adj Close: {aapl_data['Adj Close'].tail()}")
    
except Exception as e:
    print(f"Error with multiple tickers: {e}")

# Test SPY specifically (our benchmark)
try:
    print("\n3. Testing SPY (benchmark)...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    print(f"SPY data shape: {spy.shape}")
    print(f"SPY columns: {list(spy.columns)}")
    
    if 'Adj Close' in spy.columns:
        spy_prices = spy['Adj Close']
        print(f"SPY Adj Close shape: {spy_prices.shape}")
        print(f"SPY recent prices: {spy_prices.tail()}")
        
        # Calculate returns
        spy_returns = spy_prices.pct_change().dropna()
        print(f"SPY returns shape: {spy_returns.shape}")
        print(f"SPY mean daily return: {spy_returns.mean():.6f}")
        print(f"SPY daily volatility: {spy_returns.std():.6f}")
    
except Exception as e:
    print(f"Error with SPY: {e}")

print("\n=== yfinance test completed ===")
