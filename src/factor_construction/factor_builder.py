"""
FactorBuilder module for constructing financial factors for the smart beta portfolio.
Implements common factors like Value, Momentum, Quality, Low Volatility, etc.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorBuilder:
    """
    A comprehensive factor builder for constructing various financial factors
    used in smart beta portfolio strategies.
    """
    
    def __init__(self, price_data, return_data=None, fundamental_data=None):
        """
        Initialize FactorBuilder with price and optional fundamental data.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with stock prices (columns: tickers, index: dates)
        return_data : pd.DataFrame, optional
            DataFrame with stock returns (columns: tickers, index: dates)
        fundamental_data : pd.DataFrame, optional
            DataFrame with fundamental data for stocks
        """
        self.price_data = price_data
        self.return_data = return_data if return_data is not None else price_data.pct_change()
        self.fundamental_data = fundamental_data
        self.factors = pd.DataFrame(index=price_data.index)
        
    def calculate_momentum_factor(self, lookback_periods=[1, 3, 6, 12]):
        """
        Calculate momentum factors for different lookback periods.
        
        Parameters:
        -----------
        lookback_periods : list
            List of lookback periods in months
            
        Returns:
        --------
        pd.DataFrame: Momentum factors for each period
        """
        logger.info("Calculating momentum factors...")
        
        momentum_factors = {}
        
        for period in lookback_periods:
            # Calculate cumulative returns over the lookback period
            lookback_days = period * 21  # Approximate trading days per month
            momentum = self.return_data.rolling(window=lookback_days, min_periods=int(lookback_days*0.7)).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            momentum_factors[f'momentum_{period}m'] = momentum
            
        return pd.concat(momentum_factors, axis=1)
    
    def calculate_volatility_factor(self, lookback_periods=[1, 3, 6, 12]):
        """
        Calculate volatility factors (low volatility anomaly).
        
        Parameters:
        -----------
        lookback_periods : list
            List of lookback periods in months
            
        Returns:
        --------
        pd.DataFrame: Volatility factors for each period
        """
        logger.info("Calculating volatility factors...")
        
        volatility_factors = {}
        
        for period in lookback_periods:
            lookback_days = period * 21
            volatility = self.return_data.rolling(window=lookback_days, min_periods=int(lookback_days*0.7)).std() * np.sqrt(252)
            
            # Invert volatility (lower volatility = higher factor score)
            volatility_factors[f'low_vol_{period}m'] = -volatility
            
        return pd.concat(volatility_factors, axis=1)
    
    def calculate_mean_reversion_factor(self, short_period=1, long_period=12):
        """
        Calculate mean reversion factor (short-term reversal).
        
        Parameters:
        -----------
        short_period : int
            Short-term period in months
        long_period : int
            Long-term period in months
            
        Returns:
        --------
        pd.DataFrame: Mean reversion factors
        """
        logger.info("Calculating mean reversion factors...")
        
        short_days = short_period * 21
        long_days = long_period * 21
        
        # Short-term returns
        short_returns = self.return_data.rolling(window=short_days, min_periods=int(short_days*0.7)).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Long-term returns
        long_returns = self.return_data.rolling(window=long_days, min_periods=int(long_days*0.7)).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Mean reversion = negative short-term performance relative to long-term
        mean_reversion = -short_returns / (long_returns + 1e-8)  # Add small epsilon to avoid division by zero
        
        return pd.DataFrame({'mean_reversion': mean_reversion.mean(axis=1)}, index=self.price_data.index)
    
    def calculate_size_factor(self):
        """
        Calculate size factor using market capitalization (if available) or price as proxy.
        
        Returns:
        --------
        pd.DataFrame: Size factors (negative log of market cap/price)
        """
        logger.info("Calculating size factors...")
        
        # Use price as a proxy for market cap (in absence of shares outstanding data)
        # In practice, you would use actual market cap data
        size_proxy = np.log(self.price_data + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Invert size (smaller companies = higher factor score)
        size_factor = -size_proxy.mean(axis=1)
        
        return pd.DataFrame({'size_factor': size_factor}, index=self.price_data.index)
    
    def calculate_quality_factor(self):
        """
        Calculate quality factor using price-based metrics.
        In practice, this would use fundamental data like ROE, ROA, etc.
        
        Returns:
        --------
        pd.DataFrame: Quality factors
        """
        logger.info("Calculating quality factors...")
        
        # Calculate return on investment as a quality proxy
        # This is a simplified version - in practice you'd use fundamental data
        
        # Calculate rolling Sharpe ratio as quality metric
        rolling_returns = self.return_data.rolling(window=252, min_periods=126)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        
        quality_factor = rolling_sharpe.mean(axis=1)
        
        return pd.DataFrame({'quality_factor': quality_factor}, index=self.price_data.index)
    
    def calculate_value_factor(self):
        """
        Calculate value factor using price-based metrics.
        In practice, this would use fundamental ratios like P/E, P/B, etc.
        
        Returns:
        --------
        pd.DataFrame: Value factors
        """
        logger.info("Calculating value factors...")
        
        # Use price performance relative to long-term average as value proxy
        long_term_avg = self.price_data.rolling(window=252*2, min_periods=252).mean()
        current_price = self.price_data
        
        # Value factor = negative of price relative to long-term average
        # (lower relative price = higher value score)
        value_factor = -(current_price / long_term_avg).mean(axis=1)
        
        return pd.DataFrame({'value_factor': value_factor}, index=self.price_data.index)
    
    def calculate_all_factors(self):
        """
        Calculate all implemented factors and return as a combined DataFrame.
        
        Returns:
        --------
        pd.DataFrame: All calculated factors
        """
        logger.info("Calculating all factors...")
        
        all_factors = {}
        
        # Calculate momentum factors
        momentum_factors = self.calculate_momentum_factor()
        
        # Calculate volatility factors
        volatility_factors = self.calculate_volatility_factor()
        
        # Calculate other factors
        mean_reversion = self.calculate_mean_reversion_factor()['mean_reversion']
        size_factor = self.calculate_size_factor()['size_factor']
        quality_factor = self.calculate_quality_factor()['quality_factor']
        value_factor = self.calculate_value_factor()['value_factor']
        
        # Combine all factors and convert column names to strings
        all_factors_list = [momentum_factors, volatility_factors]
        other_factors = pd.DataFrame({
            'mean_reversion': mean_reversion,
            'size_factor': size_factor,
            'quality_factor': quality_factor,
            'value_factor': value_factor
        }, index=self.price_data.index)
        
        all_factors_list.append(other_factors)
        
        # Concatenate all factors
        factor_df = pd.concat(all_factors_list, axis=1)
        
        # Convert column names to strings (handle MultiIndex if present)
        if isinstance(factor_df.columns, pd.MultiIndex):
            factor_df.columns = [f"{col[0]}_{col[1]}" for col in factor_df.columns]
        else:
            factor_df.columns = factor_df.columns.astype(str)
        
        # Clean column names to remove special characters
        clean_columns = []
        for col in factor_df.columns:
            # Remove special characters and replace with underscores
            clean_col = str(col).replace('(', '').replace(')', '').replace("'", '').replace(',', '_').replace(' ', '_')
            clean_columns.append(clean_col)
        
        factor_df.columns = clean_columns
        
        # Handle missing values
        factor_df = factor_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Generated {len(factor_df.columns)} factors")
        return factor_df
    
    def normalize_factors(self, factors_df, method='zscore'):
        """
        Normalize factors using specified method.
        
        Parameters:
        -----------
        factors_df : pd.DataFrame
            DataFrame with factors to normalize
        method : str
            Normalization method ('zscore', 'minmax', 'rank')
            
        Returns:
        --------
        pd.DataFrame: Normalized factors
        """
        logger.info(f"Normalizing factors using {method} method...")
        
        if method == 'zscore':
            return factors_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        elif method == 'minmax':
            return factors_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        elif method == 'rank':
            return factors_df.rank(axis=0, pct=True)
        else:
            logger.warning(f"Unknown normalization method: {method}. Returning original factors.")
            return factors_df
    
    def get_factor_statistics(self, factors_df):
        """
        Get descriptive statistics for all factors.
        
        Parameters:
        -----------
        factors_df : pd.DataFrame
            DataFrame with factors
            
        Returns:
        --------
        pd.DataFrame: Factor statistics
        """
        stats_dict = {
            'mean': factors_df.mean(),
            'std': factors_df.std(),
            'min': factors_df.min(),
            'max': factors_df.max(),
            'skew': factors_df.skew(),
            'kurtosis': factors_df.kurtosis()
        }
        
        return pd.DataFrame(stats_dict).T
