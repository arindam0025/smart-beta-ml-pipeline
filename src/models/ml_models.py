"""
ML Models module for stock return prediction using various machine learning algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class MLModels:
    """
    A comprehensive machine learning models class for predicting stock returns
    using various algorithms and factors.
    """
    
    def __init__(self, factors_data, returns_data, test_size=0.2, random_state=42):
        """
        Initialize MLModels with factors and returns data.
        
        Parameters:
        -----------
        factors_data : pd.DataFrame
            DataFrame with factors (features)
        returns_data : pd.DataFrame or pd.Series
            Target returns data
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random state for reproducibility
        """
        self.factors_data = factors_data
        self.returns_data = returns_data
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Prepare and clean data for modeling.
        """
        logger.info("Preparing data for modeling...")
        
        # Align indices
        common_index = self.factors_data.index.intersection(self.returns_data.index)
        self.factors_data = self.factors_data.loc[common_index]
        
        if isinstance(self.returns_data, pd.DataFrame):
            self.returns_data = self.returns_data.loc[common_index]
            # If multiple return series, use the mean
            if self.returns_data.shape[1] > 1:
                self.returns_data = self.returns_data.mean(axis=1)
        else:
            self.returns_data = self.returns_data.loc[common_index]
        
        # Remove NaN values
        mask = ~(self.factors_data.isna().any(axis=1) | self.returns_data.isna())
        self.factors_data = self.factors_data[mask]
        self.returns_data = self.returns_data[mask]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.factors_data, self.returns_data, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=False  # Preserve time series order
        )
        
        # Scale features
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            index=self.X_train.index,
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            index=self.X_test.index,
            columns=self.X_test.columns
        )
        
        logger.info(f"Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
    
    def train_linear_models(self):
        """
        Train linear models (Linear Regression, Ridge, Lasso).
        """
        logger.info("Training linear models...")
        
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        for name, model in linear_models.items():
            model.fit(self.X_train_scaled, self.y_train)
            self.models[name] = model
            
            # Predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Metrics
            self.results[name] = {
                'train_mse': mean_squared_error(self.y_train, train_pred),
                'test_mse': mean_squared_error(self.y_test, test_pred),
                'train_r2': r2_score(self.y_train, train_pred),
                'test_r2': r2_score(self.y_test, test_pred),
                'train_mae': mean_absolute_error(self.y_train, train_pred),
                'test_mae': mean_absolute_error(self.y_test, test_pred)
            }
    
    def train_tree_models(self):
        """
        Train tree-based models (Random Forest, Gradient Boosting).
        """
        logger.info("Training tree-based models...")
        
        tree_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        for name, model in tree_models.items():
            model.fit(self.X_train, self.y_train)  # Tree models don't require scaling
            self.models[name] = model
            
            # Predictions
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            # Metrics
            self.results[name] = {
                'train_mse': mean_squared_error(self.y_train, train_pred),
                'test_mse': mean_squared_error(self.y_test, test_pred),
                'train_r2': r2_score(self.y_train, train_pred),
                'test_r2': r2_score(self.y_test, test_pred),
                'train_mae': mean_absolute_error(self.y_train, train_pred),
                'test_mae': mean_absolute_error(self.y_test, test_pred)
            }
    
    def train_xgboost(self):
        """
        Train XGBoost model.
        """
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # Predictions
        train_pred = xgb_model.predict(self.X_train)
        test_pred = xgb_model.predict(self.X_test)
        
        # Metrics
        self.results['xgboost'] = {
            'train_mse': mean_squared_error(self.y_train, train_pred),
            'test_mse': mean_squared_error(self.y_test, test_pred),
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_mae': mean_absolute_error(self.y_train, train_pred),
            'test_mae': mean_absolute_error(self.y_test, test_pred)
        }
    
    def train_lightgbm(self):
        """
        Train LightGBM model.
        """
        logger.info("Training LightGBM model...")
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        self.models['lightgbm'] = lgb_model
        
        # Predictions
        train_pred = lgb_model.predict(self.X_train)
        test_pred = lgb_model.predict(self.X_test)
        
        # Metrics
        self.results['lightgbm'] = {
            'train_mse': mean_squared_error(self.y_train, train_pred),
            'test_mse': mean_squared_error(self.y_test, test_pred),
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_mae': mean_absolute_error(self.y_train, train_pred),
            'test_mae': mean_absolute_error(self.y_test, test_pred)
        }
    
    def train_all_models(self):
        """
        Train all available models.
        """
        logger.info("Training all models...")
        
        self.train_linear_models()
        self.train_tree_models()
        self.train_xgboost()
        self.train_lightgbm()
        
        logger.info(f"Trained {len(self.models)} models successfully")
    
    def get_model_comparison(self):
        """
        Get comparison of all trained models.
        
        Returns:
        --------
        pd.DataFrame: Model comparison metrics
        """
        if not self.results:
            logger.warning("No models trained yet. Please run train_all_models() first.")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('test_r2', ascending=False)
        
        return comparison_df
    
    def get_feature_importance(self, model_name='xgboost'):
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to get feature importance for
            
        Returns:
        --------
        pd.DataFrame: Feature importance
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} does not have feature importance.")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.factors_data.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_returns(self, model_name='xgboost', new_factors=None):
        """
        Predict returns using a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use for prediction
        new_factors : pd.DataFrame, optional
            New factors data for prediction. If None, uses test set.
            
        Returns:
        --------
        np.array: Predicted returns
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        if new_factors is None:
            # Use test set
            if model_name in ['linear_regression', 'ridge', 'lasso']:
                predictions = model.predict(self.X_test_scaled)
            else:
                predictions = model.predict(self.X_test)
        else:
            # Use new factors
            if model_name in ['linear_regression', 'ridge', 'lasso']:
                new_factors_scaled = self.scaler.transform(new_factors)
                predictions = model.predict(new_factors_scaled)
            else:
                predictions = model.predict(new_factors)
        
        return predictions
    
    def cross_validate_model(self, model_name='xgboost', cv_folds=5):
        """
        Perform cross-validation on a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to cross-validate
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict: Cross-validation results
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        # Use appropriate data (scaled for linear models)
        if model_name in ['linear_regression', 'ridge', 'lasso']:
            X_data = self.X_train_scaled
        else:
            X_data = self.X_train
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_data, self.y_train, cv=cv_folds, scoring='r2')
        
        cv_results = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"Cross-validation for {model_name}: Mean R² = {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
        
        return cv_results
