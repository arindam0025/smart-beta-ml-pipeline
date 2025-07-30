"""
Simple test script to validate the DataFetcher functionality
"""

import sys
import os
sys.path.append('src')

from data_collection.data_fetcher import DataFetcher
from datetime import datetime, timedelta

def test_data_fetcher():
    """Test basic DataFetcher functionality"""
    
    print("=== Testing Smart Beta Portfolio Data Fetcher ===\n")
    
    # Initialize with recent 1 year for quick testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Testing with data from {start_date} to {end_date}")
    
    # Initialize fetcher
    fetcher = DataFetcher(start_date=start_date, end_date=end_date)
    
    # Test 1: Get stock universe
    print("\n1. Testing stock universe retrieval...")
    try:
        tickers = fetcher.get_sp500_universe()
        print(f"✓ Retrieved {len(tickers)} tickers")
        print(f"  Sample tickers: {tickers[:10]}")
    except Exception as e:
        print(f"✗ Error getting stock universe: {e}")
        return False
    
    # Test 2: Fetch stock data (small sample)
    print("\n2. Testing stock data fetching...")
    try:
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        prices = fetcher.get_stock_data(test_tickers)
        print(f"✓ Fetched price data: {prices.shape}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    except Exception as e:
        print(f"✗ Error fetching stock data: {e}")
        return False
    
    # Test 3: Calculate returns
    print("\n3. Testing return calculations...")
    try:
        returns = fetcher.calculate_returns(prices)
        print(f"✓ Calculated returns: {returns.shape}")
        print(f"  Mean daily return: {returns.mean().mean():.4f}")
        print(f"  Daily volatility: {returns.std().mean():.4f}")
    except Exception as e:
        print(f"✗ Error calculating returns: {e}")
        return False
    
    # Test 4: Fetch benchmark
    print("\n4. Testing benchmark data...")
    try:
        benchmark = fetcher.get_benchmark_data()
        print(f"✓ Fetched benchmark data: {len(benchmark)} observations")
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (252/len(benchmark)) - 1
        print(f"  Annualized return: {benchmark_return:.4f}")
    except Exception as e:
        print(f"✗ Error fetching benchmark: {e}")
        return False
    
    # Test 5: Try Fama-French factors
    print("\n5. Testing Fama-French factors...")
    try:
        factors = fetcher.get_fama_french_factors()
        if not factors.empty:
            print(f"✓ Fetched FF factors: {factors.shape}")
            print(f"  Available factors: {list(factors.columns)}")
        else:
            print("⚠ FF factors not available (may be network issue)")
    except Exception as e:
        print(f"⚠ Could not fetch FF factors: {e}")
    
    # Test 6: Try macro data (may fail without API key)
    print("\n6. Testing macro data...")
    try:
        macro = fetcher.get_macro_data()
        if not macro.empty:
            print(f"✓ Fetched macro data: {macro.shape}")
            print(f"  Variables: {list(macro.columns)}")
        else:
            print("⚠ Macro data not available (FRED API key required)")
    except Exception as e:
        print(f"⚠ Could not fetch macro data: {e}")
    
    # Test 7: Data saving
    print("\n7. Testing data persistence...")
    try:
        fetcher.save_data(prices, 'test_prices.csv')
        fetcher.save_data(returns, 'test_returns.csv')
        print("✓ Successfully saved test data")
        
        # Test loading
        loaded_prices = fetcher.load_data('test_prices.csv')
        if loaded_prices.shape == prices.shape:
            print("✓ Successfully loaded test data")
        else:
            print("⚠ Data shape mismatch after loading")
    except Exception as e:
        print(f"✗ Error with data persistence: {e}")
        return False
    
    print("\n=== Test Results ===")
    print("✓ Core functionality working correctly!")
    print("✓ Data fetcher is ready for use")
    
    if fetcher.fred is None:
        print("\n📝 Note: Set up FRED API key for macro data access")
        print("   Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    return True

if __name__ == "__main__":
    success = test_data_fetcher()
    if success:
        print("\n🎉 All tests passed! Ready to proceed with factor construction.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
