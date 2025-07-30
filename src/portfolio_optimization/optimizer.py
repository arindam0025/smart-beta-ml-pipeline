"""
Portfolio Optimizer module implementing various optimization techniques.
"""

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    A portfolio optimizer for constructing portfolios with maximum Sharpe ratio,
    minimum variance, and other criteria.
    """
    
    def __init__(self, returns_data=None, cov_matrix=None):
        """
        Initialize PortfolioOptimizer with returns data or covariance matrix.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame, optional
            Historical returns data for assets
        cov_matrix : pd.DataFrame, optional
            Covariance matrix of asset returns
        """
        if returns_data is not None:
            self.returns_data = returns_data
            self.cov_matrix = returns_data.cov()
        elif cov_matrix is not None:
            self.cov_matrix = cov_matrix
        else:
            logger.error("Either returns_data or cov_matrix must be provided.")
            raise ValueError("Invalid initialization: Provide either returns_data or cov_matrix.")
    
    def maximum_sharpe_ratio(self, risk_free_rate=0.01):
        """
        Calculate portfolio weights with maximum Sharpe ratio.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
            
        Returns:
        --------
        np.array: Portfolio weights
        """
        logger.info("Calculating maximum Sharpe ratio portfolio...")
        
        excess_returns = self.returns_data.mean() - risk_free_rate
        
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        
        weights = inv_cov_matrix @ excess_returns
        
        # Normalize weights
        weights /= np.sum(weights)
        logger.info(f"Maximum Sharpe ratio portfolio weights:\n{weights}")
        
        return weights
    
    def minimum_variance(self):
        """
        Calculate portfolio weights with minimum variance.
        
        Returns:
        --------
        np.array: Portfolio weights
        """
        logger.info("Calculating minimum variance portfolio...")
        
        num_assets = len(self.cov_matrix)
        
        # Constraint matrices
        P = matrix(self.cov_matrix.values)
        q = matrix(np.zeros(num_assets))
        G = matrix(-np.eye(num_assets))
        h = matrix(np.zeros(num_assets))
        A = matrix(1.0, (1, num_assets))
        b = matrix(1.0)
        
        # Solve quadratic program
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        weights = np.array(solution['x']).flatten()
        
        logger.info(f"Minimum variance portfolio weights:\n{weights}")
        
        return weights
    
    def efficient_frontier(self, num_points=50, risk_free_rate=0.01):
        """
        Calculate points on the efficient frontier.
        
        Parameters:
        -----------
        num_points : int
            Number of points to calculate on the frontier
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
            
        Returns:
        --------
        pd.DataFrame: Efficient frontier with returns, volatility, sharpe ratio
        """
        logger.info("Calculating efficient frontier...")
        
        frontier_data = []
        target_returns = np.linspace(-0.1, 0.3, num_points)
        
        for target_return in target_returns:
            try:
                # Constraint matrices
                num_assets = len(self.cov_matrix)
                P = matrix(self.cov_matrix.values)
                q = matrix(np.zeros(num_assets))
                G = matrix(-np.eye(num_assets))
                h = matrix(np.zeros(num_assets))
                A = matrix(np.vstack((self.returns_data.mean().values, np.ones(num_assets))))
                b = matrix([target_return, 1.0])
                
                # Solve quadratic program
                solvers.options['show_progress'] = False
                solution = solvers.qp(P, q, G, h, A, b)
                weights = np.array(solution['x']).flatten()
                
                # Portfolio statistics
                portfolio_return = np.sum(weights * self.returns_data.mean())
                portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                frontier_data.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                })
            except Exception as e:
                frontier_data.append({'return': target_return, 'volatility': None, 'sharpe_ratio': None})
                logger.warning(f"Failed to calculate portfolio for return {target_return}: {e}")
        
        logger.info("Efficient frontier calculated")
        return pd.DataFrame(frontier_data)
