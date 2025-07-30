"""
Smart Beta Portfolio Strategy Configuration
"""
import os
from datetime import datetime, timedelta

# Data Settings
DATA_START_DATE = "2000-01-01"
DATA_END_DATE = datetime.now().strftime("%Y-%m-%d")
LOOKBACK_PERIOD = 252  # Trading days for lookback
REBALANCE_FREQUENCY = "M"  # Monthly rebalancing

# Universe Settings
BENCHMARK = "SPY"
UNIVERSE_SIZE = 500  # Top 500 stocks by market cap
MIN_MARKET_CAP = 1e9  # $1B minimum market cap
MIN_TRADING_VOLUME = 1e6  # $1M daily trading volume

# Factor Settings
FACTORS = [
    "Market",
    "Size", 
    "Value",
    "Profitability",
    "Investment",
    "Momentum",
    "Quality",
    "Low_Volatility"
]

# Macro Variables for Timing
MACRO_VARIABLES = [
    "T10Y2Y",      # 10Y-2Y Treasury Spread
    "T10Y3M",      # 10Y-3M Treasury Spread
    "CPIAUCSL",    # CPI All Urban Consumers
    "INDPRO",      # Industrial Production Index
    "UNRATE",      # Unemployment Rate
    "HOUST",       # Housing Starts
    "VIX",         # Volatility Index
    "DGS10",       # 10-Year Treasury Rate
    "DGS3MO",      # 3-Month Treasury Rate
]

# Model Parameters
ML_MODELS = {
    "LSTM": {
        "units": 64,
        "dropout": 0.2,
        "lookback_window": 12,
        "epochs": 100,
        "batch_size": 32
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    "LightGBM": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
}

# Portfolio Optimization
OPTIMIZATION_PARAMS = {
    "method": "mean_variance",  # Options: mean_variance, risk_parity, black_litterman
    "max_weight": 0.4,          # Maximum weight per factor
    "min_weight": -0.2,         # Minimum weight (allow some shorting)
    "transaction_cost": 0.002,   # 20 bps transaction cost
    "risk_aversion": 1.0,       # Risk aversion parameter
    "lookback_months": 36       # Months for covariance estimation
}

# Backtesting Parameters
BACKTEST_PARAMS = {
    "initial_capital": 1000000,  # $1M initial capital
    "commission": 0.001,         # 10 bps commission
    "slippage": 0.001,          # 10 bps slippage
    "benchmark": "SPY",
    "risk_free_rate": 0.02      # 2% risk-free rate
}

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Keys (set as environment variables)
FRED_API_KEY = os.getenv("FRED_API_KEY")
QUANDL_API_KEY = os.getenv("QUANDL_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'smart_beta_portfolio.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
