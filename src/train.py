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
from model import NFLEnsembleModel, StackingEnsemble


def prepare_data(seasons, use_cache=True):
    """
    Load and prepare data for training.
    
    Args:
        seasons: List of seasons to include in training
        use_cache: Whether to use cached data
    
    Returns:
        X, y, feature_names, games_df (for analysis)
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    # Load raw data
    schedules, team_stats = load_data(seasons, use_cache=use_cache)
    
    print(f"\nLoaded {len(schedules)} games from {len(seasons)} seasons")
    print(f"Seasons: {min(seasons)} - {max(seasons)}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    
    # Create features
    features_df = create_feature_dataframe(schedules, team_stats)
    
    # Filter to only completed games (those with labels)
    completed_games = features_df[features_df['home_win'].notna()].copy()
    
    print(f"\nCompleted games with labels: {len(completed_games)}")
    print(f"Home wins: {completed_games['home_win'].sum()}")
    print(f"Away wins: {len(completed_games) - completed_games['home_win'].sum()}")
    print(f"Home win rate: {completed_games['home_win'].mean():.1%}")
    
    # Prepare features and target
    feature_cols = get_feature_names()
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in completed_games.columns]
    if missing_cols:
        print(f"\nWarning: Missing feature columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in completed_games.columns]
    
    X = completed_games[feature_cols]
    y = completed_games['home_win']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y, feature_cols, completed_games


def train_model(X, y, feature_names, model_type='voting', random_state=42, 
                test_size=0.2, cv_folds=5):
    """
    Train the ensemble model with train/test split and cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        model_type: 'voting' or 'stacking'
        random_state: Random seed
        test_size: Proportion of data for testing
        cv_folds: Number of cross-validation folds
    
    Returns:
        Trained model, X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("STEP 3: Train/Test Split")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} games ({len(X_train)/len(X):.1%})")
    print(f"Test set:  {len(X_test)} games ({len(X_test)/len(X):.1%})")
    print(f"\nTrain home win rate: {y_train.mean():.1%}")
    print(f"Test home win rate:  {y_test.mean():.1%}")
    
    print("\n" + "=" * 60)
    print("STEP 4: Model Training")
    print("=" * 60)
    
    # Create and train model
    if model_type == 'voting':
        model = NFLEnsembleModel(random_state=random_state)
    elif model_type == 'stacking':
        model = StackingEnsemble(random_state=random_state)
    else:
        raise ValueError("model_type must be 'voting' or 'stacking'")
    
    print(f"\nTraining {model_type} ensemble...")
    model.fit(X_train, y_train, feature_names=feature_names)
    
    print("\n" + "=" * 60)
    print("STEP 5: Cross-Validation")
    print("=" * 60)
    
    # Perform cross-validation on training set
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    # For cross-validation, we need to manually handle scaling
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        model.ensemble, 
        model.scaler.transform(X_train.values), 
        y_train.values, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Args:
        model: Trained model
        X_train, X_test: Feature matrices
        y_train, y_test: Target labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "-" * 60)
    print("ACCURACY")
    print("-" * 60)
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy:.2%})")
    
    # Baseline (always predict home team wins)
    baseline_accuracy = y_test.mean()
    print(f"Baseline (always predict home win): {baseline_accuracy:.4f} ({baseline_accuracy:.2%})")
    print(f"Improvement over baseline: {(test_accuracy - baseline_accuracy):.4f} ({(test_accuracy - baseline_accuracy):.2%})")
    
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Away Win', 'Home Win'],
                                digits=4))
    
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX (Test Set)")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n                Predicted")
    print(f"              Away    Home")
    print(f"Actual Away   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Home   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Calculate additional metrics
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()
    
    print("\n" + "-" * 60)
    print("DETAILED METRICS")
    print("-" * 60)
    print(f"True Positives (correctly predicted home wins):  {true_positives}")
    print(f"True Negatives (correctly predicted away wins):  {true_negatives}")
    print(f"False Positives (predicted home, actually away): {false_positives}")
    print(f"False Negatives (predicted away, actually home): {false_negatives}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'confusion_matrix': cm,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


def show_feature_importance(model, top_n=15):
    """Display feature importance if available."""
    try:
        print("\n" + "=" * 60)
        print("STEP 7: Feature Importance")
        print("=" * 60)
        
        importance_df = model.get_feature_importance(top_n=top_n)
        print(f"\nTop {top_n} Most Important Features:")
        print("\n" + importance_df.to_string(index=False))
        
    except Exception as e:
        print(f"\nCould not generate feature importance: {e}")


def save_model_and_results(model, results, output_dir='models'):
    """Save trained model and results."""
    print("\n" + "=" * 60)
    print("STEP 8: Saving Model")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'nfl_ensemble_model.pkl')
    model.save(model_path)
    
    # Save results summary
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
    parser.add_argument('--model-type', type=str, default='voting',
                       choices=['voting', 'stacking'],
                       help='Type of ensemble model')
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
    print(f"  Model Type: {args.model_type}")
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
            model_type=args.model_type,
            random_state=args.random_state,
            test_size=args.test_size
        )
        
        # Evaluate
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Feature importance
        show_feature_importance(model)
        
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