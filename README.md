<<<<<<< HEAD
# Smart Beta Portfolio Strategy

A comprehensive quantitative portfolio strategy that dynamically times factor exposures using machine learning models and macroeconomic signals.

## 🎯 Objective

Build an intelligent factor-timing system that:
- Predicts factor performance using macroeconomic indicators
- Dynamically allocates weights across multiple factors
- Outperforms traditional static factor portfolios
- Provides robust risk-adjusted returns with attribution analysis

## 📋 Features

- **Multi-Factor Framework**: Implementation of Fama-French 5 + Momentum + Quality + Low Volatility factors
- **Machine Learning Models**: LSTM, XGBoost, and LightGBM ensemble for factor timing
- **Macro Signal Integration**: Fed data (rates, inflation, employment) for regime detection
- **Dynamic Portfolio Optimization**: Mean-variance optimization with transaction cost modeling
- **Comprehensive Backtesting**: Full performance attribution and risk analysis
- **Interactive Dashboard**: Real-time monitoring and visualization

## 🏗️ Project Structure

```
smart_beta_portfolio/
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── data_collection/       # Data fetching modules
│   ├── factor_construction/   # Factor calculation and orthogonalization
│   ├── models/               # ML model implementations
│   ├── portfolio_optimization/ # Portfolio construction
│   ├── backtesting/          # Strategy simulation
│   └── utils/                # Helper utilities
├── notebooks/                # Jupyter notebooks for analysis
├── data/
│   ├── raw/                  # Raw data files
│   └── processed/            # Cleaned and processed data
├── tests/                    # Unit tests
├── dashboard/                # Dashboard application
├── docs/                     # Documentation
└── results/                  # Output files and reports
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd smart_beta_portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Configuration

Set environment variables for data sources:

```bash
export FRED_API_KEY="your_fred_api_key"
export QUANDL_API_KEY="your_quandl_api_key"  # Optional
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"  # Optional
```

### 3. Run the Strategy

```python
from src.main import SmartBetaStrategy

# Initialize strategy
strategy = SmartBetaStrategy()

# Run full pipeline
strategy.run_pipeline()

# View results
strategy.generate_report()
```

## 📊 Strategy Overview

### Factor Universe
- **Market**: Market beta exposure
- **Size**: Small-cap vs large-cap (SMB)
- **Value**: Book-to-market ratio (HML)
- **Profitability**: Operating profitability (RMW)
- **Investment**: Conservative vs aggressive investment (CMA)
- **Momentum**: Price momentum (UMD)
- **Quality**: Earnings quality and stability
- **Low Volatility**: Low-risk anomaly

### Machine Learning Pipeline
1. **Feature Engineering**: Macro indicators, factor returns, cross-sectional features
2. **Model Training**: LSTM for time series, XGBoost/LightGBM for cross-sectional
3. **Ensemble Learning**: Alpha stacking for improved predictions
4. **Portfolio Construction**: Dynamic weight allocation based on predictions

### Risk Management
- Transaction cost modeling (20 bps assumed)
- Maximum position limits per factor
- Drawdown controls and stop-loss mechanisms
- Regular rebalancing (monthly)

## 📈 Performance Metrics

The strategy tracks comprehensive performance metrics:
- **Returns**: Absolute and risk-adjusted returns vs benchmark
- **Risk**: Volatility, max drawdown, VaR, CVaR
- **Attribution**: Factor contribution analysis
- **Efficiency**: Sharpe ratio, information ratio, Sortino ratio
- **Turnover**: Transaction costs and portfolio stability

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/ tests/
flake8 src/ tests/
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📝 Configuration

Key parameters can be adjusted in `config/config.py`:
- Data date ranges and frequency
- Factor definitions and calculations
- ML model hyperparameters
- Portfolio optimization constraints
- Risk management rules

## 🔍 Research Notes

### Academic Foundation
- Fama-French factor models
- Dynamic factor timing literature
- Machine learning in finance
- Alternative risk premia strategies

### Implementation Details
- Monthly rebalancing frequency
- 36-month rolling window for covariance estimation
- Z-score normalization for macro variables
- Orthogonalized factors to reduce multicollinearity

## 📊 Dashboard

Launch the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

Features:
- Real-time portfolio weights
- Performance attribution charts
- Factor return predictions
- Risk decomposition analysis
- Macro regime indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is a research and educational project. Past performance does not guarantee future results. All investment strategies carry risk of loss. Please conduct your own due diligence before making any investment decisions.

## 📚 References

1. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
2. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere.
3. Frazzini, A., & Pedersen, L. H. (2014). Betting against beta.
4. Gu, S., Kelly, B., & Xiu, D. (2020). Machine learning in asset pricing.

---

For questions or support, please open an issue on GitHub or contact the development team.
=======
# smart-beta-ml-pipeline
>>>>>>> 84a68def9e4e5d0ea43cc47e6703e25ff47077f2
