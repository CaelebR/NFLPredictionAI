import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os


class NFLEnsembleModel:
    """
    Ensemble model for NFL game prediction using:
    - Random Forest
    - XGBoost
    - Decision Tree
    - Logistic Regression
    
    Combines predictions using soft voting (averaged probabilities).
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ensemble model with base estimators.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Initialize base models with reasonable hyperparameters
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight='balanced'
        )
        
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('xgb', self.xgb_model),
                ('dt', self.dt_model),
                ('lr', self.lr_model)
            ],
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, X, y, feature_names=None):
        """
        Train the ensemble model.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target labels (1 = home win, 0 = away win)
            feature_names: List of feature names (optional, for interpretability)
        
        Returns:
            self
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features (important for Logistic Regression)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble
        print("Training ensemble model...")
        self.ensemble.fit(X_scaled, y)
        
        self.is_fitted = True
        print("Model training complete!")
        
        return self
    
    def predict(self, X):
        """
        Predict game outcomes (0 or 1).
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predictions (1 = home win, 0 = away win)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.ensemble.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict win probabilities for each class.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, 2) with probabilities [away_win, home_win]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.ensemble.predict_proba(X_scaled)
    
    def get_individual_predictions(self, X):
        """
        Get predictions from each base model individually.
        Useful for understanding model agreement/disagreement.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary with predictions from each model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.ensemble.named_estimators_.items():
            predictions[name] = model.predict(X_scaled)
            probabilities[name] = model.predict_proba(X_scaled)[:, 1]  # Probability of home win
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importances from tree-based models.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance!")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Pass feature_names during fit().")
        
        # Get importances from tree-based models
        rf_importance = self.ensemble.named_estimators_['rf'].feature_importances_
        xgb_importance = self.ensemble.named_estimators_['xgb'].feature_importances_
        dt_importance = self.ensemble.named_estimators_['dt'].feature_importances_
        
        # Average importances
        avg_importance = (rf_importance + xgb_importance + dt_importance) / 3
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'random_forest': rf_importance,
            'xgboost': xgb_importance,
            'decision_tree': dt_importance,
            'average': avg_importance
        })
        
        # Sort by average importance
        importance_df = importance_df.sort_values('average', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath='models/nfl_ensemble_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump({
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/nfl_ensemble_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded NFLEnsembleModel instance
        """
        # Load saved data
        saved_data = joblib.load(filepath)
        
        # Create new instance
        model = cls(random_state=saved_data['random_state'])
        model.ensemble = saved_data['ensemble']
        model.scaler = saved_data['scaler']
        model.feature_names = saved_data['feature_names']
        model.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        
        return model


if __name__ == "__main__":
    # Test with dummy data
    print("Testing NFLEnsembleModel...")
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.randn(100, 20)
    
    # Create and train model
    model = NFLEnsembleModel()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:5, 1]}")  # Home win probabilities
    
    # Test save/load
    model.save('models/test_model.pkl')
    loaded_model = NFLEnsembleModel.load('models/test_model.pkl')
    
    print("\n✓ All tests passed!")