"""
Backtesting framework for evaluating smart beta portfolio strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtest:
    """
    Framework for backtesting portfolio strategies.
    """
    
    def __init__(self, prices, weights, risk_free_rate=0.01):
        """
        Initialize Backtest with price data and portfolio weights.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data (columns: tickers, index: dates)
        weights : pd.DataFrame
            DataFrame with portfolio weights (columns: tickers, index: dates)
        risk_free_rate : float
            Risk-free rate for performance metrics
        """
        self.prices = prices
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        
    def run_backtest(self):
        """
        Run backtest and calculate portfolio returns.
        
        Returns:
        --------
        pd.DataFrame: Portfolio returns
        """
        logger.info("Running backtest...")
        
        # Resample to ensure alignment
        self.prices = self.prices.resample('D').ffill()
        self.weights = self.weights.resample('D').ffill()
        
        # Calculate returns
        returns = self.prices.pct_change().fillna(0)
        portfolio_returns = (returns * self.weights.shift()).sum(axis=1)
        
        logger.info("Backtest completed")
        return portfolio_returns
    
    def calculate_performance_metrics(self, portfolio_returns):
        """
        Calculate performance metrics for the backtested portfolio.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series of portfolio returns
            
        Returns:
        --------
        dict: Performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        total_return = portfolio_returns.sum()
        annualized_return = np.prod(1 + portfolio_returns) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        logger.info("Performance metrics calculated")
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_returns):
        """
        Calculate the maximum drawdown of the portfolio.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series of portfolio returns
            
        Returns:
        --------
        float: Maximum drawdown
        """
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def plot_performance(self, portfolio_returns):
        """
        Plot portfolio performance and drawdown.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series of portfolio returns
        """
        logger.info("Plotting portfolio performance...")
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_returns, label='Cumulative Returns')
        plt.title('Portfolio Performance')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdown, color='red', label='Drawdown')
        plt.title('Drawdown')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        logger.info("Performance plot generated")
