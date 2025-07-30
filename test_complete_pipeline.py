#!/usr/bin/env python3
"""
Complete Pipeline Test Script for Smart Beta Portfolio Strategy
Runs the entire pipeline from data collection to backtesting and shows results.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collection.data_fetcher import DataFetcher
from factor_construction.factor_builder import FactorBuilder
from models.ml_models import MLModels

print("="*80)
print("SMART BETA PORTFOLIO STRATEGY - COMPLETE PIPELINE EXECUTION")
print("="*80)

def main():
    """
    Main function to execute the complete pipeline.
    """
    
    # Step 1: Data Collection
    print("\n📊 STEP 1: DATA COLLECTION")
    print("-" * 40)
    
    try:
        fetcher = DataFetcher()
        
        # Get S&P 500 tickers
        sp500_tickers = fetcher.get_sp500_universe()
        print(f"✅ Fetched {len(sp500_tickers)} S&P 500 tickers")
        
        # Fetch stock data (using first 20 for demonstration to speed up execution)
        selected_tickers = sp500_tickers[:20]
        print(f"📈 Fetching data for {len(selected_tickers)} selected stocks...")
        
        stock_data = fetcher.get_stock_data(selected_tickers)
        print(f"✅ Stock data shape: {stock_data.shape}")
        
        # Fetch benchmark data
        benchmark_data = fetcher.get_benchmark_data('SPY')
        print(f"✅ Benchmark data shape: {benchmark_data.shape}")
        
        # Calculate returns
        stock_returns = fetcher.calculate_returns(stock_data)
        benchmark_returns = fetcher.calculate_returns(benchmark_data)
        
        print("✅ Data collection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return
    
    # Step 2: Factor Construction
    print("\n🔧 STEP 2: FACTOR CONSTRUCTION")
    print("-" * 40)
    
    try:
        factor_builder = FactorBuilder(stock_data, stock_returns)
        
        # Calculate all factors
        factors_df = factor_builder.calculate_all_factors()
        print(f"✅ Factors calculated: {factors_df.shape}")
        print(f"📊 Factor columns: {list(factors_df.columns)}")
        
        # Normalize factors
        factors_normalized = factor_builder.normalize_factors(factors_df, method='zscore')
        print("✅ Factors normalized using z-score method")
        
        # Get factor statistics
        factor_stats = factor_builder.get_factor_statistics(factors_normalized)
        print("\n📈 Factor Statistics (Top 5 factors):")
        print(factor_stats.round(4).head())
        
    except Exception as e:
        print(f"❌ Error in factor construction: {e}")
        return
    
    # Step 3: Machine Learning Models
    print("\n🤖 STEP 3: MACHINE LEARNING MODELS")
    print("-" * 40)
    
    try:
        # Prepare target variable (using benchmark returns)
        # benchmark_returns is a Series, not a DataFrame
        target_returns = benchmark_returns
        
        # Initialize ML models
        ml_models = MLModels(factors_normalized, target_returns, test_size=0.2)
        
        # Train all models
        print("🚀 Training all ML models...")
        ml_models.train_all_models()
        
        # Get model comparison
        model_comparison = ml_models.get_model_comparison()
        print("\n📊 Model Comparison Results:")
        print(model_comparison.round(4))
        
        # Get feature importance for best model
        best_model = model_comparison.index[0]
        feature_importance = ml_models.get_feature_importance(best_model)
        print(f"\n🏆 Best Model: {best_model}")
        print(f"📊 Top 5 Important Features:")
        if feature_importance is not None:
            print(feature_importance.head().round(4))
        
    except Exception as e:
        print(f"❌ Error in machine learning: {e}")
        return
    
    # Step 4: Portfolio Construction (Simplified)
    print("\n💼 STEP 4: PORTFOLIO CONSTRUCTION")
    print("-" * 40)
    
    try:
        # Get predictions from best ML model
        predictions = ml_models.predict_returns(best_model)
        print(f"✅ Generated {len(predictions)} return predictions")
        
        # Simple equal-weight portfolio for demonstration
        num_assets = len(selected_tickers)
        equal_weights = np.ones(num_assets) / num_assets
        
        print(f"📊 Portfolio construction: Equal-weight portfolio with {num_assets} assets")
        print(f"✅ Weight per asset: {equal_weights[0]:.4f}")
        
    except Exception as e:
        print(f"❌ Error in portfolio construction: {e}")
        return
    
    # Step 5: Performance Evaluation
    print("\n📈 STEP 5: PERFORMANCE EVALUATION")
    print("-" * 40)
    
    try:
        # Calculate portfolio returns
        portfolio_returns = (stock_returns * equal_weights).sum(axis=1)
        
        # Calculate performance metrics
        def calculate_metrics(returns, risk_free_rate=0.02):
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                'Total Return': total_return,
                'Annualized Return': annualized_return,
                'Annualized Volatility': annualized_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown
            }
        
        # Portfolio metrics
        portfolio_metrics = calculate_metrics(portfolio_returns)
        
        # Benchmark metrics
        benchmark_metrics = calculate_metrics(benchmark_returns)
        
        # Create comparison table
        print("\n📊 PERFORMANCE COMPARISON:")
        print("=" * 60)
        comparison_data = {
            'Metric': list(portfolio_metrics.keys()),
            'Portfolio': [f"{v:.2%}" if 'Return' in k or 'Volatility' in k or 'Drawdown' in k 
                         else f"{v:.3f}" for k, v in portfolio_metrics.items()],
            'Benchmark (SPY)': [f"{v:.2%}" if 'Return' in k or 'Volatility' in k or 'Drawdown' in k 
                               else f"{v:.3f}" for k, v in benchmark_metrics.items()]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("=" * 60)
        
        # Calculate excess returns
        excess_return = portfolio_metrics['Annualized Return'] - benchmark_metrics['Annualized Return']
        print(f"\n🎯 KEY RESULTS:")
        print(f"   • Excess Return vs Benchmark: {excess_return:.2%}")
        print(f"   • Risk-Adjusted Performance: {portfolio_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:.3f}")
        
    except Exception as e:
        print(f"❌ Error in performance evaluation: {e}")
        return
    
    # Step 6: Summary
    print("\n🎉 STEP 6: PIPELINE SUMMARY")
    print("-" * 40)
    
    print(f"✅ Data Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"✅ Number of Assets: {len(selected_tickers)}")
    print(f"✅ Number of Factors: {len(factors_df.columns)}")
    print(f"✅ Best ML Model: {best_model}")
    print(f"✅ Portfolio Strategy: Equal-weight (simplified)")
    
    # Data quality summary
    print(f"\n📊 DATA QUALITY:")
    print(f"   • Stock data points: {len(stock_data)}")
    print(f"   • Missing data: {stock_data.isnull().sum().sum()}")
    print(f"   • Factor completeness: {(~factors_normalized.isnull()).sum().sum()}/{factors_normalized.size}")
    
    print("\n" + "="*80)
    print("🎊 SMART BETA PORTFOLIO PIPELINE COMPLETED SUCCESSFULLY! 🎊")
    print("="*80)
    
    return {
        'stock_data': stock_data,
        'factors': factors_normalized,
        'model_comparison': model_comparison,
        'portfolio_metrics': portfolio_metrics,
        'benchmark_metrics': benchmark_metrics
    }

if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n📝 Results saved to variable 'results'")
        print(f"   Available keys: {list(results.keys())}")
        
        # Optional: Save results to files
        try:
            results['stock_data'].to_csv('data/processed/pipeline_stock_data.csv')
            results['factors'].to_csv('data/processed/pipeline_factors.csv')
            results['model_comparison'].to_csv('data/processed/pipeline_model_comparison.csv')
            print("💾 Results saved to CSV files in data/processed/")
        except Exception as e:
            print(f"⚠️  Could not save results to files: {e}")
    
    print("\n🚀 Pipeline execution completed!")
import matplotlib.pyplot as plt

# Example cumulative returns
portfolio_cumulative = (1 + portfolio_returns).cumprod()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(portfolio_cumulative, label='Smart Beta Portfolio', linewidth=2)
plt.plot(benchmark_cumulative, label='S&P 500 (SPY)', linewidth=2)
plt.title('Cumulative Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()