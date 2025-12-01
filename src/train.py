import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import os
import sys

# Import your custom modules
from data_loader import load_data
from feature_engineering import create_feature_dataframe, get_feature_names
from model import NFLEnsembleModel 


def prepare_data(seasons, use_cache=True):
    # ... (unchanged) ...
    
    # Load raw data
    schedules, team_stats = load_data(seasons, use_cache=use_cache)
    
    # Create features
    features_df = create_feature_dataframe(schedules, team_stats)
    
    # Filter to only completed games (those with labels)
    completed_games = features_df[features_df['home_win'].notna()].copy()
    
    # Prepare features and target
    feature_cols = get_feature_names()
    
    X = completed_games[feature_cols]
    y = completed_games['home_win']
    
    
    return X, y, feature_cols, completed_games


def train_model(X, y, feature_names, random_state=42, test_size=0.2, cv_folds=5):
    """
    Train the ensemble model with train/test split and cross-validation.
    Always uses NFLEnsembleModel (voting ensemble).
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    
    # Create and train model (always NFLEnsembleModel)
    model = NFLEnsembleModel(random_state=random_state)
    
    model.fit(X_train, y_train, feature_names=feature_names)
    
    
    # Perform cross-validation on training set
    
    # For cross-validation, manually handle scaling
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        model.ensemble, 
        model.scaler.transform(X_train.values), 
        y_train.values, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    # ... unchanged ...
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Baseline (always predict home team wins)
    baseline_accuracy = y_test.mean()
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Calculate additional metrics
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'confusion_matrix': cm,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }



def save_model_and_results(model, results, output_dir='models'):
    # ... unchanged ...
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'nfl_ensemble_model.pkl')
    model.save(model_path)
    
    results_path = os.path.join(output_dir, 'training_results.txt')
    with open(results_path, 'w') as f:
        f.write("NFL Ensemble Model - Training Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train Accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy:  {results['test_accuracy']:.4f}\n")
        f.write(f"Baseline:       {results['baseline_accuracy']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{results['confusion_matrix']}\n")
    
    print(f"Results saved to {results_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train NFL game prediction model')
    parser.add_argument('--seasons', nargs='+', type=int, 
                       default=[2019, 2020, 2021, 2022, 2023],
                       help='Seasons to include in training (e.g., 2020 2021 2022)')
    # Removed model-type arg
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force re-download of data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained model')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("NFL GAME PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Seasons: {args.seasons}")
    print(f"  Test Size: {args.test_size}")
    print(f"  Random State: {args.random_state}")
    print(f"  Use Cache: {not args.no_cache}")
    
    try:
        # Prepare data
        X, y, feature_names, games_df = prepare_data(
            args.seasons, 
            use_cache=not args.no_cache
        )
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(
            X, y, feature_names,
            random_state=args.random_state,
            test_size=args.test_size
        )
        
        # Evaluate
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        
        # Save everything
        save_model_and_results(model, results, args.output_dir)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nModel saved to: {args.output_dir}/nfl_ensemble_model.pkl")
        print(f"Test Accuracy: {results['test_accuracy']:.2%}")
        print("\nYou can now use this model for predictions with predict.py")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
