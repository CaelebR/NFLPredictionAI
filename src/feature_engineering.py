import pandas as pd
import numpy as np


def safe_get(stats: dict, key: str, default=0):
    """
    A helper that safely gets values from a stats dict.
    Returns a default if the key does not exist or is NaN.
    """
    value = stats.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


def get_team_stats(stats_df: pd.DataFrame, team: str, season: int) -> dict:
    """
    Safely retrieve team stats for a given season.
    Returns empty dict if team/season not found.
    """
    team_data = stats_df[(stats_df['team'] == team) & (stats_df['season'] == season)]
    
    if team_data.empty:
        print(f"Warning: No stats found for {team} in {season}")
        return {}
    
    return team_data.iloc[0].to_dict()


def build_features_for_game(home_stats: dict, away_stats: dict,
                            is_playoff: bool, is_neutral: bool,
                            week: int, season: int):
    """
    Builds the feature dictionary for a single game using stats for home and away teams.
    
    Args:
        home_stats: Dictionary of home team statistics
        away_stats: Dictionary of away team statistics
        is_playoff: Whether this is a playoff game
        is_neutral: Whether this is at a neutral site
        week: Week number of the season
        season: Season year
    
    Returns:
        Dictionary of engineered features
    """
    # Handle both possible column name formats (with and without _pg suffix)
    def get_stat(stats, base_name):
        """Try both column name formats"""
        if f"{base_name}_pg" in stats:
            return safe_get(stats, f"{base_name}_pg")
        return safe_get(stats, base_name)
    
    home_passing = get_stat(home_stats, 'passing_yards')
    home_rushing = get_stat(home_stats, 'rushing_yards')
    home_total = get_stat(home_stats, 'total_yards')
    home_points = get_stat(home_stats, 'points_scored') or get_stat(home_stats, 'points')
    home_pass_tds = get_stat(home_stats, 'passing_touchdowns') or get_stat(home_stats, 'passing_tds')
    home_fumbles = get_stat(home_stats, 'fumbles_lost')
    home_ints = get_stat(home_stats, 'interceptions_thrown') or get_stat(home_stats, 'interceptions')
    
    away_passing = get_stat(away_stats, 'passing_yards')
    away_rushing = get_stat(away_stats, 'rushing_yards')
    away_total = get_stat(away_stats, 'total_yards')
    away_points = get_stat(away_stats, 'points_scored') or get_stat(away_stats, 'points')
    away_pass_tds = get_stat(away_stats, 'passing_touchdowns') or get_stat(away_stats, 'passing_tds')
    away_fumbles = get_stat(away_stats, 'fumbles_lost')
    away_ints = get_stat(away_stats, 'interceptions_thrown') or get_stat(away_stats, 'interceptions')
    
    features = {
        # Basic team stats (Home)
        'home_passing_ypg': home_passing,
        'home_rushing_ypg': home_rushing,
        'home_total_ypg': home_total,
        'home_points_pg': home_points,
        'home_passing_tds_pg': home_pass_tds,
        'home_turnovers_pg': home_fumbles + home_ints,

        # Basic team stats (Away)
        'away_passing_ypg': away_passing,
        'away_rushing_ypg': away_rushing,
        'away_total_ypg': away_total,
        'away_points_pg': away_points,
        'away_passing_tds_pg': away_pass_tds,
        'away_turnovers_pg': away_fumbles + away_ints,

        # Advantage stats (differential features)
        'passing_advantage': home_passing - away_passing,
        'rushing_advantage': home_rushing - away_rushing,
        'scoring_advantage': home_points - away_points,
        'total_yards_advantage': home_total - away_total,
        'turnover_advantage': (away_fumbles + away_ints) - (home_fumbles + home_ints),

        # Ratio features (can be powerful for ML models)
        'passing_ratio': home_passing / (away_passing + 1),  # +1 to avoid division by zero
        'scoring_ratio': home_points / (away_points + 1),
        
        # Game metadata
        'home_field_advantage': 0 if is_neutral else 2.5,
        'is_playoff': int(is_playoff),
        'is_neutral': int(is_neutral),
        'week': int(week),
        'season': int(season)
    }

    return features


def create_feature_dataframe(games_df: pd.DataFrame, stats_df: pd.DataFrame):
    """
    Create a full feature dataframe for ALL games in the schedule.

    Args:
        games_df: DataFrame with columns ['game_id', 'home_team', 'away_team', 
                  'week', 'season', 'playoff', 'neutral_site', 'home_score', 'away_score']
        stats_df: DataFrame with team statistics (one row per team per season)
    
    Returns:
        DataFrame with engineered features for each game
    """
    feature_rows = []
    skipped_games = 0

    for idx, game in games_df.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        season = game['season']

        # Safely get team stats
        home_stats = get_team_stats(stats_df, home_team, season)
        away_stats = get_team_stats(stats_df, away_team, season)

        # Skip if either team has no stats
        if not home_stats or not away_stats:
            skipped_games += 1
            continue

        # Build features
        row = build_features_for_game(
            home_stats,
            away_stats,
            is_playoff=bool(game.get('playoff', False)),
            is_neutral=bool(game.get('neutral_site', False)),
            week=game['week'],
            season=season
        )

        # Add identifiers
        row['game_id'] = game['game_id']
        row['home_team'] = home_team
        row['away_team'] = away_team

        # Add label (1 = home win, 0 = away win)
        if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
            row['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
        else:
            row['home_win'] = None  # Future games

        feature_rows.append(row)

    if skipped_games > 0:
        print(f"Warning: Skipped {skipped_games} games due to missing team stats")

    df = pd.DataFrame(feature_rows)
    
    print(f"\nFeature Engineering Summary:")
    print(f"  Total games processed: {len(df)}")
    print(f"  Games with labels: {df['home_win'].notna().sum()}")
    print(f"  Games without labels (future): {df['home_win'].isna().sum()}")
    print(f"  Feature columns: {len([col for col in df.columns if col not in ['game_id', 'home_team', 'away_team', 'home_win']])}")
    
    return df


def get_feature_names(exclude_meta=True):
    """
    Returns list of feature column names (useful for model training).
    
    Args:
        exclude_meta: If True, excludes game_id, team names, and target variable
    
    Returns:
        List of feature column names
    """
    all_features = [
        'home_passing_ypg', 'home_rushing_ypg', 'home_total_ypg',
        'home_points_pg', 'home_passing_tds_pg', 'home_turnovers_pg',
        'away_passing_ypg', 'away_rushing_ypg', 'away_total_ypg',
        'away_points_pg', 'away_passing_tds_pg', 'away_turnovers_pg',
        'passing_advantage', 'rushing_advantage', 'scoring_advantage',
        'total_yards_advantage', 'turnover_advantage',
        'passing_ratio', 'scoring_ratio',
        'home_field_advantage', 'is_playoff', 'is_neutral', 'week', 'season'
    ]
    
    return all_features


# Test function
if __name__ == "__main__":
    # Test with sample data
    from data_loader import load_data
    
    seasons = [2022, 2023]
    schedules, team_stats = load_data(seasons, use_cache=True)
    
    print("Creating features...")
    features_df = create_feature_dataframe(schedules, team_stats)
    
    print("\n=== Sample Features ===")
    print(features_df.head())
    
    print("\n=== Feature Statistics ===")
    print(features_df[get_feature_names()].describe())