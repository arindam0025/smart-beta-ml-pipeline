"""
Test updated yfinance functionality with auto_adjust=False
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing updated yfinance functionality...")

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Test with auto_adjust=False to get Adj Close column
try:
    print("\n1. Testing SPY with auto_adjust=False...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
    print(f"SPY data shape: {spy.shape}")
    print(f"SPY columns: {list(spy.columns)}")
    
    # Check for MultiIndex
    if isinstance(spy.columns, pd.MultiIndex):
        print(f"MultiIndex levels: {spy.columns.levels}")
        # Extract Adj Close
        if ('Adj Close', 'SPY') in spy.columns:
            adj_close = spy[('Adj Close', 'SPY')]
            print(f"Adj Close found: {adj_close.shape}")
        else:
            print("Adj Close not found in expected format")
    else:
        if 'Adj Close' in spy.columns:
            adj_close = spy['Adj Close']
            print(f"Adj Close found: {adj_close.shape}")
        else:
            print("Adj Close column not found")

except Exception as e:
    print(f"Error: {e}")

# Test multiple tickers
try:
    print("\n2. Testing multiple tickers...")
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, 
                      group_by='ticker', auto_adjust=False)
    print(f"Multi data shape: {data.shape}")
    print(f"Column structure: {data.columns}")
    
    # Extract AAPL Adj Close
    if 'AAPL' in data.columns.levels[0]:
        aapl_data = data['AAPL']
        if 'Adj Close' in aapl_data.columns:
            aapl_adj = aapl_data['Adj Close']
            print(f"AAPL Adj Close: {aapl_adj.shape}")
        else:
            print("AAPL Adj Close not found")

except Exception as e:
    print(f"Error with multiple tickers: {e}")

print("\n=== Test completed ===")
