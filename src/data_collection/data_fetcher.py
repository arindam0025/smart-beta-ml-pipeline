"""
Data Collection Module for Smart Beta Portfolio Strategy

This module handles fetching and preprocessing of:
1. Stock price data from Yahoo Finance
2. Macroeconomic data from FRED
3. Fama-French factor data
4. Market cap and fundamental data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from fredapi import Fred
import warnings
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import *

warnings.filterwarnings('ignore')

class DataFetcher:
    """
    Comprehensive data fetching class for financial and macroeconomic data
    """
    
    def __init__(self, start_date: str = DATA_START_DATE, end_date: str = DATA_END_DATE):
        """
        Initialize DataFetcher
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.logger = self._setup_logger()
        
        # Initialize FRED API if key is available
        if FRED_API_KEY:
            self.fred = Fred(api_key=FRED_API_KEY)
        else:
            self.fred = None
            self.logger.warning("FRED API key not found. Macro data fetching will be limited.")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_sp500_universe(self, min_market_cap: float = MIN_MARKET_CAP) -> List[str]:
        """
        Get S&P 500 universe with market cap filtering
        
        Args:
            min_market_cap: Minimum market cap threshold
            
        Returns:
            List of ticker symbols
        """
        try:
            # Get S&P 500 list from Wikipedia
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_df = tables[0]
            
            tickers = sp500_df['Symbol'].tolist()
            
            # Clean ticker symbols (handle special characters)
            cleaned_tickers = []
            for ticker in tickers:
                # Replace common problematic characters
                cleaned_ticker = ticker.replace('.', '-')
                cleaned_tickers.append(cleaned_ticker)
            
            self.logger.info(f"Retrieved {len(cleaned_tickers)} S&P 500 tickers")
            return cleaned_tickers[:UNIVERSE_SIZE]  # Limit to configured universe size
        
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 universe: {e}")
            # Fallback to common large-cap stocks
            fallback_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX', 
                'ADBE', 'CRM', 'CMCSA', 'XOM', 'NKE', 'WFC', 'TMO', 'ABT', 'VZ'
            ]
            return fallback_tickers
    
    def get_stock_data(self, tickers: List[str], 
                      price_type: str = 'Adj Close') -> pd.DataFrame:
        """
        Fetch stock price data from Yahoo Finance
        
        Args:
            tickers: List of ticker symbols
            price_type: Type of price data ('Adj Close', 'Close', etc.)
            
        Returns:
            DataFrame with stock prices
        """
        try:
            self.logger.info(f"Fetching stock data for {len(tickers)} tickers...")
            
            # Fetch data in smaller batches to avoid API limits
            batch_size = 10
            all_prices = []
            
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]
                try:
                    batch_data = yf.download(
                        batch_tickers,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        group_by='ticker',
                        auto_adjust=False  # Keep original columns including Adj Close
                    )
                    
                    # Handle MultiIndex structure from new yfinance API
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        prices_list = []
                        
                        # Extract data for each ticker
                        for ticker in batch_tickers:
                            try:
                                # Check if ticker exists in level 1 (for single ticker) or level 0 (for multiple)
                                if len(batch_tickers) == 1:
                                    # Single ticker: columns are (price_type, ticker)
                                    if (price_type, ticker) in batch_data.columns:
                                        price_series = batch_data[(price_type, ticker)]
                                    elif ('Adj Close', ticker) in batch_data.columns:
                                        price_series = batch_data[('Adj Close', ticker)]
                                    else:
                                        continue
                                else:
                                    # Multiple tickers: columns are (ticker, price_type)
                                    if ticker in batch_data.columns.levels[0]:
                                        ticker_data = batch_data[ticker]
                                        if price_type in ticker_data.columns:
                                            price_series = ticker_data[price_type]
                                        elif 'Adj Close' in ticker_data.columns:
                                            price_series = ticker_data['Adj Close']
                                        else:
                                            continue
                                    else:
                                        continue
                                
                                price_series.name = ticker
                                prices_list.append(price_series)
                                
                            except Exception as e:
                                self.logger.debug(f"Skipping {ticker}: {e}")
                                continue
                        
                        if prices_list:
                            prices = pd.concat(prices_list, axis=1)
                        else:
                            continue
                    else:
                        # Fallback for non-MultiIndex (shouldn't happen with new API)
                        if price_type in batch_data.columns:
                            prices = batch_data[[price_type]]
                            prices.columns = batch_tickers[:1]
                        else:
                            continue
                    
                    all_prices.append(prices)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching batch {i//batch_size + 1}: {e}")
                    continue
            
            if not all_prices:
                raise ValueError("No data could be fetched")
            
            # Combine all batches
            combined_prices = pd.concat(all_prices, axis=1)
            
            # Clean data
            combined_prices = combined_prices.dropna(how='all')  # Remove rows with all NaN
            # Keep stocks with reasonable amount of data (at least 60 days)
            min_observations = min(60, len(combined_prices) // 4)
            combined_prices = combined_prices.loc[:, combined_prices.notna().sum() > min_observations]
            
            self.logger.info(f"Successfully fetched data for {len(combined_prices.columns)} stocks")
            return combined_prices
        
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}")
            raise
    
    def get_benchmark_data(self, benchmark: str = BENCHMARK) -> pd.Series:
        """
        Fetch benchmark data (e.g., SPY)
        
        Args:
            benchmark: Benchmark ticker symbol
            
        Returns:
            Series with benchmark prices
        """
        try:
            benchmark_data = yf.download(
                benchmark,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False
            )
            
            # Handle MultiIndex structure from new yfinance API
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                # Multi-level columns: (price_type, ticker)
                if ('Adj Close', benchmark) in benchmark_data.columns:
                    prices = benchmark_data[('Adj Close', benchmark)]
                elif ('Close', benchmark) in benchmark_data.columns:
                    prices = benchmark_data[('Close', benchmark)]
                else:
                    # Take the first available price column
                    prices = benchmark_data.iloc[:, 0]
            else:
                # Single-level columns (fallback)
                if 'Adj Close' in benchmark_data.columns:
                    prices = benchmark_data['Adj Close']
                elif 'Close' in benchmark_data.columns:
                    prices = benchmark_data['Close']
                else:
                    # Take the last column as fallback
                    prices = benchmark_data.iloc[:, -1]
            
            # Ensure it's a Series
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            
            prices.name = benchmark
            self.logger.info(f"Fetched benchmark data for {benchmark}")
            return prices
        
        except Exception as e:
            self.logger.error(f"Error fetching benchmark data: {e}")
            raise
    
    def get_fama_french_factors(self) -> pd.DataFrame:
        """
        Fetch Fama-French factor data from Ken French's data library
        
        Returns:
            DataFrame with factor returns
        """
        try:
            # Fama-French 5 factors + momentum
            ff5_factors = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', 
                                                start=self.start_date, 
                                                end=self.end_date)[0]
            
            momentum = pdr.get_data_famafrench('F-F_Momentum_Factor', 
                                             start=self.start_date, 
                                             end=self.end_date)[0]
            
            # Combine factors
            factors = pd.concat([ff5_factors, momentum], axis=1)
            
            # Convert from percentage to decimal
            factors = factors / 100
            
            self.logger.info("Successfully fetched Fama-French factors")
            return factors
        
        except Exception as e:
            self.logger.error(f"Error fetching Fama-French factors: {e}")
            # Return empty DataFrame if fetch fails
            return pd.DataFrame()
    
    def get_macro_data(self) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED
        
        Returns:
            DataFrame with macro indicators
        """
        if not self.fred:
            self.logger.warning("FRED API not available")
            return pd.DataFrame()
        
        try:
            macro_data = {}
            
            for variable in MACRO_VARIABLES:
                try:
                    data = self.fred.get_series(variable, 
                                              start=self.start_date, 
                                              end=self.end_date)
                    macro_data[variable] = data
                    self.logger.info(f"Fetched {variable}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not fetch {variable}: {e}")
                    continue
            
            if not macro_data:
                return pd.DataFrame()
            
            # Combine into DataFrame
            macro_df = pd.DataFrame(macro_data)
            
            # Forward fill missing values
            macro_df = macro_df.fillna(method='ffill')
            
            # Resample to monthly frequency
            macro_df = macro_df.resample('M').last()
            
            self.logger.info(f"Successfully fetched {len(macro_df.columns)} macro variables")
            return macro_df
        
        except Exception as e:
            self.logger.error(f"Error fetching macro data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame, 
                         method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: DataFrame with price data
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with returns
        """
        try:
            if method == 'log':
                returns = np.log(prices / prices.shift(1))
            else:
                returns = prices.pct_change()
            
            returns = returns.dropna()
            self.logger.info(f"Calculated {method} returns")
            return returns
        
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            raise
    
    def get_risk_free_rate(self) -> pd.Series:
        """
        Get risk-free rate (3-month Treasury rate)
        
        Returns:
            Series with risk-free rates
        """
        if not self.fred:
            # Use constant rate if FRED not available
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            return pd.Series(BACKTEST_PARAMS['risk_free_rate'] / 252, index=dates)
        
        try:
            rf_rate = self.fred.get_series('DGS3MO', start=self.start_date, end=self.end_date)
            rf_rate = rf_rate / 100 / 252  # Convert to daily decimal
            rf_rate = rf_rate.fillna(method='ffill')
            
            self.logger.info("Fetched risk-free rate data")
            return rf_rate
        
        except Exception as e:
            self.logger.warning(f"Error fetching risk-free rate: {e}")
            # Fallback to constant rate
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            return pd.Series(BACKTEST_PARAMS['risk_free_rate'] / 252, index=dates)
    
    def save_data(self, data: pd.DataFrame, filename: str, 
                  directory: str = PROCESSED_DATA_DIR) -> None:
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Output filename
            directory: Output directory
        """
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            data.to_csv(filepath)
            self.logger.info(f"Saved data to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def load_data(self, filename: str, 
                  directory: str = PROCESSED_DATA_DIR) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filename: Input filename
            directory: Input directory
            
        Returns:
            DataFrame with loaded data
        """
        try:
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded data from {filepath}")
            return data
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required data for the strategy
        
        Returns:
            Dictionary containing all datasets
        """
        self.logger.info("Starting comprehensive data fetch...")
        
        data = {}
        
        try:
            # 1. Get stock universe
            tickers = self.get_sp500_universe()
            
            # 2. Fetch stock prices
            prices = self.get_stock_data(tickers)
            data['prices'] = prices
            
            # 3. Calculate returns
            returns = self.calculate_returns(prices)
            data['returns'] = returns
            
            # 4. Fetch benchmark
            benchmark = self.get_benchmark_data()
            data['benchmark'] = benchmark
            
            # 5. Fetch factors
            factors = self.get_fama_french_factors()
            if not factors.empty:
                data['factors'] = factors
            
            # 6. Fetch macro data
            macro = self.get_macro_data()
            if not macro.empty:
                data['macro'] = macro
            
            # 7. Fetch risk-free rate
            rf_rate = self.get_risk_free_rate()
            data['risk_free_rate'] = rf_rate
            
            self.logger.info("Completed comprehensive data fetch")
            return data
        
        except Exception as e:
            self.logger.error(f"Error in comprehensive data fetch: {e}")
            raise


def main():
    """
    Example usage of DataFetcher
    """
    # Initialize fetcher
    fetcher = DataFetcher()
    
    # Fetch all data
    all_data = fetcher.fetch_all_data()
    
    # Save data
    for key, data in all_data.items():
        if isinstance(data, pd.DataFrame):
            fetcher.save_data(data, f"{key}.csv")
        elif isinstance(data, pd.Series):
            fetcher.save_data(data.to_frame(), f"{key}.csv")
    
    print("Data fetching completed successfully!")


if __name__ == "__main__":
    main()
