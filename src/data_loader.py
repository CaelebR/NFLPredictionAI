import os
import pandas as pd
import nflreadpy as nfl

CACHE_DIR = 'data/raw'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_schedules(seasons, use_cache=True):
    """
    Load NFL game schedules for specified seasons using nflreadpy.
    """
    cache_file = f"{CACHE_DIR}/schedules_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached schedules from {cache_file}")
        return pd.read_csv(cache_file)
    
    print(f"Downloading schedules for seasons: {seasons}")
    
    # nflreadpy returns Polars DataFrame, convert to pandas
    df = nfl.load_schedules(seasons=seasons).to_pandas()

    # Add playoff indicator
    df['playoff'] = df['game_type'].isin(['WC', 'DIV', 'CON', 'SB']).astype(int)

    # Add result column for completed games
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['game_completed'] = df['home_score'].notna() & df['away_score'].notna()

    if use_cache:
        df.to_csv(cache_file, index=False)
        print(f"Cached schedules to {cache_file}")
    
    return df


def load_team_stats(seasons, use_cache=True):
    """
    Load team statistics using nflreadpy.
    """
    cache_file = f"{CACHE_DIR}/team_stats_{min(seasons)}_{max(seasons)}.csv"

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached team stats → {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Loading team stats for seasons {seasons} ...")

    # Load player stats and aggregate to team level
    player_stats = nfl.load_player_stats(seasons=seasons).to_pandas()
    
    # Print columns to debug
    print(f"Available columns: {player_stats.columns.tolist()[:20]}...")
    
    # Try different possible team column names
    team_col = None
    for possible_name in ['recent_team', 'team', 'team_abbr', 'posteam']:
        if possible_name in player_stats.columns:
            team_col = possible_name
            break
    
    if team_col is None:
        raise ValueError("Could not find team column in player stats. Available columns: " + str(player_stats.columns.tolist()))
    
    print(f"Using team column: {team_col}")
    
    # Aggregate to team level per season
    agg_dict = {
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'passing_tds': 'sum',
        'rushing_tds': 'sum',
        'receiving_tds': 'sum',
        'interceptions': 'sum',
        'completions': 'sum',
        'attempts': 'sum',
        'carries': 'sum'
    }
    
    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in player_stats.columns}
    
    print(f"Aggregating columns: {list(agg_dict.keys())}")
    
    team_stats = player_stats.groupby([team_col, 'season']).agg(agg_dict).reset_index()
    team_stats = team_stats.rename(columns={team_col: 'team'})
    
    # Calculate per-game averages
    games_per_season = 17
    
    team_stats['passing_yards_pg'] = team_stats.get('passing_yards', 0) / games_per_season
    team_stats['rushing_yards_pg'] = team_stats.get('rushing_yards', 0) / games_per_season
    team_stats['total_yards_pg'] = (team_stats.get('passing_yards', 0) + team_stats.get('rushing_yards', 0)) / games_per_season
    team_stats['points_pg'] = ((team_stats.get('passing_tds', 0) + team_stats.get('rushing_tds', 0) + team_stats.get('receiving_tds', 0)) * 7) / games_per_season
    team_stats['passing_touchdowns_pg'] = team_stats.get('passing_tds', 0) / games_per_season
    team_stats['fumbles_lost_pg'] = 0  # Calculate if data available
    team_stats['interceptions_thrown_pg'] = team_stats.get('interceptions', 0) / games_per_season
    
    if use_cache:
        team_stats.to_csv(cache_file, index=False)
        print(f"Saved team stats cache → {cache_file}")

    return team_stats


def load_data(seasons, use_cache=True):
    """Load schedules + team stats together."""
    schedules = load_schedules(seasons, use_cache=use_cache)
    team_stats = load_team_stats(seasons, use_cache=use_cache)

    missing_seasons = set(seasons) - set(team_stats["season"].unique())
    if missing_seasons:
        print(f"Warning: Missing team stats for seasons: {missing_seasons}")

    print(f"\nLoaded {len(schedules)} games and stats for {len(team_stats)} team-seasons")
    
    return schedules, team_stats