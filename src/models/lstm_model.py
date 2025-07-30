"""
LSTM Model module for time series prediction of stock returns using deep learning.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class LSTMModel:
    """
    LSTM model for predicting stock returns using time series data.
    """
    
    def __init__(self, sequence_length=60, test_size=0.2, random_state=42):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences for LSTM
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random state for reproducibility
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def prepare_data(self, factors_data, returns_data):
        """
        Prepare data for LSTM training.
        
        Parameters:
        -----------
        factors_data : pd.DataFrame
            DataFrame with factors (features)
        returns_data : pd.DataFrame or pd.Series
            Target returns data
        """
        logger.info("Preparing data for LSTM...")
        
        # Align indices
        common_index = factors_data.index.intersection(returns_data.index)
        factors_data = factors_data.loc[common_index]
        
        if isinstance(returns_data, pd.DataFrame):
            returns_data = returns_data.loc[common_index]
            if returns_data.shape[1] > 1:
                returns_data = returns_data.mean(axis=1)
        else:
            returns_data = returns_data.loc[common_index]
        
        # Remove NaN values
        mask = ~(factors_data.isna().any(axis=1) | returns_data.isna())
        factors_data = factors_data[mask]
        returns_data = returns_data[mask]
        
        # Convert to numpy arrays
        X = factors_data.values
        y = returns_data.values.reshape(-1, 1)
        
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled)
        
        # Split data
        split_idx = int(len(X_sequences) * (1 - self.test_size))
        
        self.X_train = X_sequences[:split_idx]
        self.X_test = X_sequences[split_idx:]
        self.y_train = y_sequences[:split_idx]
        self.y_test = y_sequences[split_idx:]
        
        logger.info(f"Data prepared: {len(self.X_train)} training sequences, {len(self.X_test)} test sequences")
        logger.info(f"Input shape: {self.X_train.shape}, Output shape: {self.y_train.shape}")
    
    def _create_sequences(self, X, y):
        """
        Create sequences for LSTM input.
        
        Parameters:
        -----------
        X : np.array
            Features array
        y : np.array
            Target array
            
        Returns:
        --------
        tuple: X_sequences, y_sequences
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        lstm_units : list
            List of LSTM units for each layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        """
        logger.info("Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.X_train.shape[2])
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        
        logger.info("LSTM model built successfully")
        logger.info(f"Model summary:\n{model.summary()}")
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1, patience=10):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Proportion of training data to use for validation
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        dict: Training history
        """
        if self.model is None:
            logger.error("Model not built. Please call build_model() first.")
            return None
        
        logger.info("Training LSTM model...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        callbacks = [early_stopping, reduce_lr]
        
        # Train model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        logger.info("LSTM model training completed")
        
        return history.history
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained. Please call train_model() first.")
            return None
        
        logger.info("Evaluating LSTM model...")
        
        # Make predictions
        train_pred_scaled = self.model.predict(self.X_train)
        test_pred_scaled = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        train_pred = self.scaler_y.inverse_transform(train_pred_scaled)
        test_pred = self.scaler_y.inverse_transform(test_pred_scaled)
        y_train_actual = self.scaler_y.inverse_transform(self.y_train)
        y_test_actual = self.scaler_y.inverse_transform(self.y_test)
        
        # Calculate metrics
        results = {
            'train_mse': mean_squared_error(y_train_actual, train_pred),
            'test_mse': mean_squared_error(y_test_actual, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, test_pred)),
            'train_mae': mean_absolute_error(y_train_actual, train_pred),
            'test_mae': mean_absolute_error(y_test_actual, test_pred),
            'train_r2': r2_score(y_train_actual, train_pred),
            'test_r2': r2_score(y_test_actual, test_pred)
        }
        
        logger.info("LSTM Model Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.6f}")
        
        return results
    
    def predict(self, new_data=None):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        new_data : np.array, optional
            New data for prediction. If None, uses test set.
            
        Returns:
        --------
        np.array: Predictions
        """
        if self.model is None:
            logger.error("Model not trained. Please call train_model() first.")
            return None
        
        if new_data is None:
            # Use test set
            predictions_scaled = self.model.predict(self.X_test)
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
        else:
            # Use new data
            new_data_scaled = self.scaler_X.transform(new_data)
            new_sequences, _ = self._create_sequences(new_data_scaled, np.zeros((len(new_data_scaled), 1)))
            predictions_scaled = self.model.predict(new_sequences)
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def get_training_history(self):
        """
        Get training history for visualization.
        
        Returns:
        --------
        dict: Training history
        """
        if self.history is None:
            logger.warning("No training history available.")
            return None
        
        return self.history.history
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            logger.error("No model to save.")
            return
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model architecture summary.
        
        Returns:
        --------
        str: Model summary
        """
        if self.model is None:
            logger.warning("No model built yet.")
            return None
        
        return self.model.summary()
