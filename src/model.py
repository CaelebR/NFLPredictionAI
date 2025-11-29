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
        print("✓ Model training complete!")
        
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
        
        print(f"✓ Model saved to {filepath}")
    
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
        
        print(f"✓ Model loaded from {filepath}")
        
        return model


class StackingEnsemble:
    """
    Alternative ensemble approach using stacking.
    Base models make predictions, then a meta-learner combines them.
    """
    
    def __init__(self, random_state=42):
        """Initialize stacking ensemble."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Base models
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1
            ),
            'xgb': XGBClassifier(
                n_estimators=200, max_depth=6, random_state=random_state, n_jobs=-1
            ),
            'dt': DecisionTreeClassifier(
                max_depth=10, random_state=random_state
            ),
            'lr': LogisticRegression(
                max_iter=1000, random_state=random_state
            )
        }
        
        # Meta-learner (learns how to combine base model predictions)
        self.meta_learner = LogisticRegression(random_state=random_state)
        
        self.is_fitted = False
    
    def fit(self, X, y, feature_names=None):
        """
        Train the stacking ensemble.
        Uses cross-validation to generate meta-features.
        """
        from sklearn.model_selection import cross_val_predict
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training base models...")
        
        # Train base models and generate meta-features
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training {name}...")
            model.fit(X_scaled, y)
            
            # Get out-of-fold predictions for meta-learner
            meta_features[:, idx] = cross_val_predict(
                model, X_scaled, y, cv=5, method='predict_proba', n_jobs=-1
            )[:, 1]
        
        print("Training meta-learner...")
        self.meta_learner.fit(meta_features, y)
        
        self.is_fitted = True
        print("✓ Stacking ensemble training complete!")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using stacking."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))
        for idx, model in enumerate(self.base_models.values()):
            meta_features[:, idx] = model.predict_proba(X_scaled)[:, 1]
        
        # Meta-learner combines base predictions
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict classes using stacking."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# Utility function for quick model creation
def create_default_model(model_type='voting', random_state=42):
    """
    Create a default ensemble model.
    
    Args:
        model_type: 'voting' or 'stacking'
        random_state: Random seed
    
    Returns:
        NFLEnsembleModel or StackingEnsemble instance
    """
    if model_type == 'voting':
        return NFLEnsembleModel(random_state=random_state)
    elif model_type == 'stacking':
        return StackingEnsemble(random_state=random_state)
    else:
        raise ValueError("model_type must be 'voting' or 'stacking'")


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