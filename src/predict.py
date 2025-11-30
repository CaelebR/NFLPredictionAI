import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# Import your custom modules
from data_loader import load_data
from feature_engineering import create_feature_dataframe, get_feature_names
from model import NFLEnsembleModel


def load_model(model_path='models/nfl_ensemble_model.pkl'):
    """Load trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train a model first using: python train.py"
        )
    
    print(f"Loading model from {model_path}...")
    model = NFLEnsembleModel.load(model_path)
    return model


def get_upcoming_games(season, week=None):
    """
    Get upcoming games for prediction.
    
    Args:
        season: Season year
        week: Specific week (optional, if None gets all future games)
    
    Returns:
        DataFrame with upcoming games
    """
    print(f"\nFetching games for {season} season...")
    
    # Load schedule
    schedules, team_stats = load_data([season], use_cache=True)
    
    # Filter to games without scores (upcoming games)
    upcoming = schedules[schedules['home_score'].isna()].copy()
    
    if week is not None:
        upcoming = upcoming[upcoming['week'] == week]
        print(f"Found {len(upcoming)} games in Week {week}")
    else:
        print(f"Found {len(upcoming)} upcoming games")
    
    if len(upcoming) == 0:
        print("No upcoming games found!")
        return None
    
    return upcoming, team_stats


def get_single_game(home_team, away_team, season, week=1, is_playoff=False, is_neutral=False):
    """
    Create a single game for prediction (manual entry).
    
    Args:
        home_team: Home team abbreviation (e.g., 'KC')
        away_team: Away team abbreviation (e.g., 'BUF')
        season: Season year
        week: Week number
        is_playoff: Whether it's a playoff game
        is_neutral: Whether it's at a neutral site
    
    Returns:
        DataFrame with single game
    """
    print(f"\nCreating custom matchup: {away_team} @ {home_team}")
    
    # Load team stats for the season
    _, team_stats = load_data([season], use_cache=True)
    
    # Create game entry
    game = pd.DataFrame({
        'game_id': [f'custom_{home_team}_{away_team}'],
        'home_team': [home_team],
        'away_team': [away_team],
        'week': [week],
        'season': [season],
        'playoff': [1 if is_playoff else 0],
        'neutral_site': [1 if is_neutral else 0],
        'home_score': [None],
        'away_score': [None]
    })
    
    return game, team_stats


def make_predictions(model, games_df, team_stats):
    """
    Make predictions for a set of games.
    
    Args:
        model: Trained model
        games_df: DataFrame with games to predict
        team_stats: Team statistics DataFrame
    
    Returns:
        DataFrame with predictions
    """
    print("\nGenerating features...")
    
    # Create features for games
    features_df = create_feature_dataframe(games_df, team_stats)
    
    # Get feature columns
    feature_cols = get_feature_names()
    feature_cols = [col for col in feature_cols if col in features_df.columns]
    
    X = features_df[feature_cols]
    
    print(f"Making predictions for {len(X)} games...")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Get individual model predictions
    individual = model.get_individual_predictions(X)
    
    # Compile results
    results = features_df[['game_id', 'home_team', 'away_team', 'week', 'season']].copy()
    results['predicted_winner'] = results.apply(
        lambda row: row['home_team'] if predictions[row.name] == 1 else row['away_team'],
        axis=1
    )
    results['home_win_prob'] = probabilities[:, 1]
    results['away_win_prob'] = probabilities[:, 0]
    results['confidence'] = np.maximum(probabilities[:, 0], probabilities[:, 1])
    
    # Add individual model probabilities
    results['rf_home_prob'] = individual['probabilities']['rf']
    results['xgb_home_prob'] = individual['probabilities']['xgb']
    results['dt_home_prob'] = individual['probabilities']['dt']
    results['lr_home_prob'] = individual['probabilities']['lr']
    
    return results


def print_predictions(results, show_details=True):
    """
    Print predictions in a readable format.
    
    Args:
        results: DataFrame with prediction results
        show_details: Whether to show individual model predictions
    """
    print("\n" + "=" * 80)
    print("GAME PREDICTIONS")
    print("=" * 80)
    
    for idx, game in results.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        predicted_winner = game['predicted_winner']
        confidence = game['confidence']
        
        print(f"\nWeek {game['week']}: {away_team} @ {home_team}")
        print(f"Predicted Winner: {predicted_winner} ({confidence:.1%} confidence)")
    
    print("\n" + "=" * 80)


def save_predictions(results, output_path='predictions/predictions.csv'):
    """Save predictions to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add timestamp
    results['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    total = len(results)
    home_wins = (results['predicted_winner'] == results['home_team']).sum()
    away_wins = total - home_wins
    
    print(f"\nTotal Games: {total}")
    print(f"Predicted Home Wins: {home_wins} ({home_wins/total:.1%})")
    print(f"Predicted Away Wins: {away_wins} ({away_wins/total:.1%})")
    
    # Confidence breakdown
    print(f"\nConfidence Breakdown:")
    high_conf = (results['confidence'] >= 0.70).sum()
    med_conf = ((results['confidence'] >= 0.60) & (results['confidence'] < 0.70)).sum()
    low_conf = (results['confidence'] < 0.60).sum()
    
    print(f"  High Confidence (≥70%): {high_conf} games")
    print(f"  Medium Confidence (60-70%): {med_conf} games")
    print(f"  Low Confidence (<60%): {low_conf} games")
    
    # Average confidence
    avg_conf = results['confidence'].mean()
    print(f"\nAverage Confidence: {avg_conf:.1%}")


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Make NFL game predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all upcoming games in current season
  python predict.py --season 2024
  
  # Predict specific week
  python predict.py --season 2024 --week 15
  
  # Predict specific matchup
  python predict.py --home-team KC --away-team BUF --season 2024
  
  # Save predictions to file
  python predict.py --season 2024 --week 15 --output predictions/week15.csv
        """
    )
    
    parser.add_argument('--season', type=int, required=True,
                       help='Season year (e.g., 2024)')
    parser.add_argument('--week', type=int, default=None,
                       help='Specific week to predict (optional)')
    parser.add_argument('--home-team', type=str, default=None,
                       help='Home team abbreviation for single game prediction')
    parser.add_argument('--away-team', type=str, default=None,
                       help='Away team abbreviation for single game prediction')
    parser.add_argument('--playoff', action='store_true',
                       help='Mark as playoff game (for single game prediction)')
    parser.add_argument('--neutral', action='store_true',
                       help='Mark as neutral site (for single game prediction)')
    parser.add_argument('--model-path', type=str, default='models/nfl_ensemble_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    parser.add_argument('--no-details', action='store_true',
                       help='Hide individual model predictions')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("NFL GAME PREDICTION")
    print("=" * 80)
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Get games to predict
        if args.home_team and args.away_team:
            # Single game prediction
            games_df, team_stats = get_single_game(
                args.home_team,
                args.away_team,
                args.season,
                week=args.week or 1,
                is_playoff=args.playoff,
                is_neutral=args.neutral
            )
        else:
            # Upcoming games from schedule
            result = get_upcoming_games(args.season, args.week)
            if result is None:
                return
            games_df, team_stats = result
        
        # Make predictions
        results = make_predictions(model, games_df, team_stats)
        
        # Display results
        print_predictions(results, show_details=not args.no_details)
        
        # Save if requested
        if args.output:
            save_predictions(results, args.output)
        
        print("\n" + "=" * 80)
        print("PREDICTION COMPLETE!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()